#!/usr/bin/env python3
"""
Set up and run Kosmulator MCMC across models and observation groups.

Main entry: `main(...)` — called by Kosmulator.py
"""
from __future__ import annotations

import os
import logging
import time
from copy import deepcopy
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np

# ── MPI is optional; utils.init_mpi() handles details ──────────────────────────
try:
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover
    MPI = None  # utils.init_mpi() will fall back to serial

import User_defined_modules as UDM
from Kosmulator_main import Config
from Kosmulator_main.utils import acquire_run_lock
from Kosmulator_main.utils import (
    parse_cli_args,
    build_plot_settings,
    init_mpi,
    mpi_broadcast,
    compute_pantheon_cov,
    print_completion_banner,
    print_model_banner,
    detect_vectorisation,
    format_elapsed_time,
    apply_pantheon_cov,
    cleanup_pantheon_cov,
    get_pool,
    get_parallel_flag,
    set_main_threads,
    mpi_node_ranks,
    mpi_abort_on_error,
    mpi_wait_all,
    set_post_pool,
    MPI_END_POLL_MAX_S,
    prepare_output,
    load_or_run_chain,
    init_summary_context,
    issue_observation_warnings,   # moved from here into utils.py
)
import Plots.Plots as MP
from Kosmulator_main import Class_run as CR
from Kosmulator_main import rd_helpers as RDH
import Kosmulator_main.constants as K   # live access to flags
from Kosmulator_main.utils import generate_label as cfg_generate_label

# Optional Zeus import for printout (not strictly required to run emcee)
try:
    import zeus  # noqa: F401
except Exception:
    zeus = None

log = logging.getLogger("MCMC_setup")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


class _DropExactMessages(logging.Filter):
    """
    Logging filter that completely drops messages whose text matches one of a
    small set of tokens (e.g., 'fσ8 run solo').

    Used to:
      * Allow higher-level log collectors to still see the tokens.
      * Avoid cluttering the console for typical users.
    """
    def __init__(self, blocked):
        super().__init__()
        self.blocked = {b.strip() for b in blocked}

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage().strip()
        return msg not in self.blocked


# Drop the solo-dataset token lines globally (any logger, any time)
root = logging.getLogger()
root.addFilter(_DropExactMessages(blocked=[
    "f solo",
    "fσ8 run solo",
    "fσ₈ run solo",
    "fsigma8 run solo",
    "JLA run solo",
    "Pantheon+ (uncal) run solo",
    "Union3 run solo",
    "DESY5 run solo",
]))


# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────

def _required_params_for_block(obs_block: Sequence[str]) -> Set[str]:
    """
    Compute the union of required parameters for an observation block.
    This uses OBS_REQUIRED_PARAMS defined in rd_helpers or this module.
    """
    from Kosmulator_main import rd_helpers as RDH  # adjust if needed

    required: Set[str] = set()
    for obs in obs_block:
        required |= RDH.OBS_REQUIRED_PARAMS.get(obs, set())
    return required


def _log_parameter_injections(logger, obs_blocks: Iterable[Sequence[str]]) -> None:
    """
    Pretty-print the parameter injections per observation block.
    """
    # Build (block, params) pairs
    rows: list[Tuple[Sequence[str], Set[str]]] = []
    for block in obs_blocks:
        params = _required_params_for_block(block)
        if params:
            rows.append((block, params))

    if not rows:
        return

    logger.info("Parameter injections (auto-added, identical for all models)")
    header = f"{'Obs block':<44} | Injected parameters"
    sep    = "-" * len(header)
    logger.info("  " + header)
    logger.info("  " + sep)

    for obs_block, params in rows:
        block_str  = "[" + ", ".join(obs_block) + "]"
        params_str = ", ".join(sorted(params))
        # WARNING level so it stands out visually, like before
        logger.warning("  %s | %s", f"{block_str:<44}", params_str)
        
def _model_has_any_cmb_or_bbn(CONFIG, model_name: str) -> bool:
    """
    Return True if *any* observation group for this model
    involves CMB or BBN-type data.
    """
    obs_types_list = CONFIG[model_name].get("observation_types", [])
    for obs_types in obs_types_list:
        for t in obs_types:
            t = str(t)
            if t.startswith("CMB") or ("BBN" in t):
                return True
    return False
    
def main(
    model_names: List[str],
    observations: List[List[str]],
    reference_model: str,
    prior_limits: Dict[str, tuple],
    reference_values: Dict[str, Any],
    nwalkers: int,
    nsteps: int,
    burn: int,
    convergence: float,
    pantheonp_mode: str = "PplusSH0ES",
    sample_H0_uncalibrated_SNe: bool = False,
    cmb_nuisance: str = "baseline",
    waic_include_cmb: bool = False,
    neutrinos: str = "massive",
    nu_mass_eV: float = 0.06,
    nu_N_eff: float = 3.044,
    cmb_nonlinear=None,
    hmcode_version: str = "2020",
    growth_model: str = "gamma",
    emcee_moves: str = "de_snooker",
    zeus_light_mode: bool = False,
    rd_background: str = "model",
    lensing_only_priors: bool = True,
):
    """
    Orchestrates config creation, data prep, MPI broadcast, and the per-model MCMC.

    Returns
    -------
    samples : dict
        Nested dictionary: samples[model_name][obs_key] → chain / metadata.
    """
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1) CLI
    # ------------------------------------------------------------------
    args = parse_cli_args()
    if getattr(args, "peek", False):
        # Item 54: look at the run --output_suffix (while it goes on, or afterwards); no sampling
        from Kosmulator_main.peek import peek
        peek((getattr(args, "output_suffix", "") or "").strip(), burn=getattr(args, "peek_burn", None),
             all_params=bool(getattr(args, "corner_show_all_cmb_params", False)),
             latex_enabled=bool(getattr(args, "latex_enabled", False)))
        return {}
    K.set_engine_overrides(
        force_vec=getattr(args, "force_vectorisation", False),
        disable_vec=getattr(args, "disable_vectorisation", False),
        force_z=getattr(args, "force_zeus", False),
        force_e=getattr(args, "force_emcee", False),
        mode=getattr(args, "engine_mode", "mixed"),
    )

    # Planck high-l likelihood: plik with Planck's nuisance parameters (baseline 21, extended 30,
    # full 47), or plik_lite. Set before the config is built (the sampled nuisances and the
    # files depend on it).
    _mode = str(cmb_nuisance).strip().lower()
    if _mode not in K.CMB_NUISANCE_MODES:
        raise ValueError(f"cmb_nuisance must be one of {K.CMB_NUISANCE_MODES}, not {cmb_nuisance!r}")
    K.CMB_NUISANCE = _mode
    # WAIC of groups with Planck data: one term per Planck likelihood, or n/a (item 53)
    K.WAIC_INCLUDE_CMB = bool(waic_include_cmb)
    # Neutrinos in every CLASS call and in the late-time E(z) (items 27, 13). Before any pool
    # exists: spawned workers inherit them through the KOSM_* variables set here.
    K.set_neutrinos(neutrinos, nu_mass_eV, nu_N_eff)
    # Non-linear correction of the CMB lensing (one setting for every model) and the growth
    # model for f and f sigma_8 (gamma, or the model's G_eff): also before any pool, and
    # before the config (with "geff" gamma is not sampled)
    K.set_cmb_nonlinear(cmb_nonlinear, hmcode_version)
    K.set_growth_model(growth_model)
    # Proposal moves of the samplers (item 55): emcee stretch or DE + snooker, zeus light mode
    K.set_sampler_moves(emcee_moves, zeus_light_mode)
    # r_d of calibrated BAO/DESI groups: the model's own E(z) (rd_Class) or flat LCDM (item 12).
    # Before any pool: spawned workers read KOSM_RD_BACKGROUND.
    K.set_rd_background(rd_background)
    # Lensing without primary CMB (item 62): n_s prior and tau fixed. Before the config (tau is
    # then not sampled) and before any pool (workers read KOSM_LENSING_ONLY_PRIORS).
    K.set_lensing_only_priors(lensing_only_priors)
    # l_max of the lensed CLASS spectra: 3000 for Planck, 8501 with ACT, 4095 with SPT (item 57).
    # One value for the run (every CMB group of it), before any pool: workers read KOSM_CMB_CLASS_LMAX.
    K.set_cmb_class_lmax(K.cmb_lmax_for_groups(observations))
    if _mode == "lite":
        # figures and tables name the likelihood that was used
        for _t in ("CMB_hil", "CMB_hil_TT"):
            _tex, _txt = K.OBS_PRETTY_MAP[_t]
            if not _txt.endswith(" lite"):
                K.OBS_PRETTY_MAP[_t] = (_tex + " lite", _txt + " lite")

    # ------------------------------------------------------------------
    # 2) Plot settings
    # ------------------------------------------------------------------
    PLOT_SETTINGS = build_plot_settings(
        observations, args.output_suffix, args.latex_enabled, args.plot_table
    )

    PLOT_SETTINGS["corner_show_all_cmb_params"] = bool(
        getattr(args, "corner_show_all_cmb_params", False)
    )
    PLOT_SETTINGS["corner_table_full_params"] = bool(
        getattr(args, "corner_table_full", False)
    )

    # Same root as corner and best-fit plots (constants.DEFAULT_PLOTS_BASE);
    # "Saved_plots" here split the output into two folders on Linux/macOS.
    base_plots = PLOT_SETTINGS.get("base_plots_dir", K.DEFAULT_PLOTS_BASE)
    suffix = (getattr(args, "output_suffix", "") or "").strip()
    auto_base = os.path.join(base_plots, suffix) if suffix else base_plots
    PLOT_SETTINGS["autocorr_save_path"] = auto_base
    os.makedirs(PLOT_SETTINGS["autocorr_save_path"], exist_ok=True)

    PLOT_SETTINGS["autocorr_check_every"] = int(
        getattr(args, "autocorr_check_every", 100) or 100
    )
    _user_buf = getattr(args, "autocorr_buffer", None)
    PLOT_SETTINGS["autocorr_buffer_after_burn"] = int(
        _user_buf if _user_buf is not None else 0
    )
    PLOT_SETTINGS["tau_consecutive"] = max(1, int(args.consecutive_required))
    # Tolerance on the relative change of tau_max (Kosmulator.py `convergence`),
    # used by the stopping rule and quoted in the convergence report
    PLOT_SETTINGS["conv_tau_rtol"] = float(convergence)
    PLOT_SETTINGS["callback_ncheck"] = PLOT_SETTINGS["autocorr_check_every"]

    # ------------------------------------------------------------------
    # 3) MPI init
    # ------------------------------------------------------------------
    comm, rank = init_mpi()
    # An error on any rank ends the whole MPI job instead of leaving the others waiting (item 50)
    mpi_abort_on_error(comm)
    # MPI ranks on rank 0's machine (they sleep while rank 0 works alone, item 49): rank 0's
    # threads for its own CLASS work (item 45). Counted here, while every rank is present.
    K.MPI_NODE_RANKS = mpi_node_ranks(comm) if bool(getattr(args, "use_mpi", False)) else 1

    # ------------------------------------------------------------------
    # 4) Master builds CONFIG & data
    # ------------------------------------------------------------------
    if rank == 0:
        style = getattr(args, "init_log", "terse")
        with init_summary_context(style=style):
            models = UDM.Get_model_names(model_names)
            CONFIG, data = Config.create_config(
                models=models,
                reference_values=reference_values,
                prior_limits=prior_limits,
                restrictions=UDM.Get_model_restrictions(model_names),
                coupled_restrictions=UDM.Get_model_coupled_restrictions(model_names),
                observation=observations,
                nwalkers=nwalkers,
                nsteps=nsteps,
                burn=burn,
                model_name=model_names,
                pantheonp_mode=pantheonp_mode,
                sample_H0_uncalibrated_SNe=sample_H0_uncalibrated_SNe,
                logger=log,
            )
            issue_observation_warnings(CONFIG, models, token_mode=True)
        from Kosmulator_main.utils import warn_data_overlaps
        warn_data_overlaps(CONFIG, models)                        # item 35: same data counted twice
        announce_growth_and_lensing(CONFIG, list(model_names))
    else:
        models = None
        CONFIG = None
        data = None

    # ------------------------------------------------------------------
    # 5) Broadcast
    # ------------------------------------------------------------------
    shared = mpi_broadcast(
        comm, rank, {"models": models, "CONFIG": CONFIG, "data": data}
    )
    models, CONFIG, data = (
        shared["models"],
        shared["CONFIG"],
        shared["data"],
    )

    # ------------------------------------------------------------------
    # 6) Pantheon covariance
    # ------------------------------------------------------------------
    for tag in ("PantheonP", "PantheonPS"):
        pant_cov = compute_pantheon_cov(
            data,
            CONFIG[reference_model],
            comm,
            rank,
            os.path.join(K.OBSERVATIONS_BASE, "PantheonP.cov"),
            obs_tag=tag,
        )
        if pant_cov is not None:
            data[tag]["cov"] = pant_cov
            data[tag]["type_data_error"] = np.sqrt(
                np.sum(pant_cov ** 2, axis=1)
            )

    # ------------------------------------------------------------------
    # 7) Vectorisation detection
    # ------------------------------------------------------------------
    vectorised = detect_vectorisation(
        models=model_names,
        get_model_fn=UDM.Get_model_function,
        config=CONFIG,
        data=data,
    )

    if getattr(K, "disable_vectorisation", False):
        vectorised = {m: False for m in vectorised}
        log.warning(
            "Vectorisation explicitly disabled by user (--disable_vectorisation). "
            "Models will run in scalar mode."
        )
    
    if getattr(K, "force_vectorisation", False):
        vectorised = {m: True for m in vectorised}
        log.info("Vectorisation forced via CLI; treating all models as vectorised.")

    # ------------------------------------------------------------------
    # 8) Engine decision per model (THIS feeds the banner)
    # ------------------------------------------------------------------
    engine_mode = getattr(K, "engine_mode", "mixed")
    force_emcee = bool(getattr(K, "force_emcee", False))
    force_zeus  = bool(getattr(K, "force_zeus", False))

    K.engine_for_model = {}
    any_needs_pool = False

    for m in model_names:
        can_vec = bool(vectorised.get(m, False))
        touches_cmb_bbn = _model_has_any_cmb_or_bbn(CONFIG, m)
        # Expensive late-time likelihoods (official JLA) run zeus through the pool
        if any(o in getattr(K, "POOL_PREFERRED_DATASETS", set())
               for grp in CONFIG[m].get("observations", []) for o in grp):
            any_needs_pool = True

        if engine_mode in ("single", "mixed"):
            # (A rule here used to send every wowaCDM model to emcee because zeus
            # walkers got stuck at the w0 + wa < 0 wall; fixed by the initial ball
            # and warm-up in Kosmulator_MCMC, so w0waCDM follows the general rule.)
            if force_emcee:
                eng = "emcee"
            elif force_zeus and (zeus is not None):
                eng = "zeus"
            else:
                if touches_cmb_bbn:
                    eng = "emcee"
                else:
                    eng = "zeus" if (can_vec and zeus is not None) else "emcee" 

            K.engine_for_model[m] = eng

            # emcee needs workers only for CMB/BBN groups or non-vectorisable models;
            # other emcee groups run vectorised in this process (Kosmulator_MCMC).
            if (eng == "emcee" and (touches_cmb_bbn or not can_vec)) or (eng == "zeus" and not can_vec):
                any_needs_pool = True

        elif engine_mode == "fastest":
            if (not can_vec) or touches_cmb_bbn:
                any_needs_pool = True

        else:
            eng = "emcee" if touches_cmb_bbn else (
                "zeus" if (can_vec and zeus is not None) else "emcee"
            )
            K.engine_for_model[m] = eng
            if eng == "emcee" or (eng == "zeus" and not can_vec):
                any_needs_pool = True

    # ------------------------------------------------------------------
    # 9) MPI early-exit if no pool is ever needed
    # ------------------------------------------------------------------
    if bool(getattr(args, "use_mpi", False)) and not any_needs_pool:
        # The extra ranks wait for rank 0 in a sleeping barrier (MPI_Finalize spins on
        # MPICH ch3, a whole core per rank for the whole run); rank 0 joins it at exit
        if rank != 0:
            try:
                if comm is not None:
                    from mpi4py import MPI as _MPI
                    mpi_wait_all(comm, MPI_END_POLL_MAX_S)
                    _MPI.Finalize()
            finally:
                os._exit(0)
        else:
            if comm is not None:
                import atexit
                atexit.register(mpi_wait_all, comm)
            print(
                "MPI requested but every group runs vectorised in this process (zeus or emcee) → "
                "proceeding on rank 0; the extra ranks wait idle until it ends."
            )

    # ------------------------------------------------------------------
    # 10) Engine + parallel plan summary (rank 0)
    # ------------------------------------------------------------------
    if rank == 0:
        log.info("Engine mode: %s", engine_mode)

        if force_zeus or force_emcee:
            log.info(
                "Engine CLI overrides: --force_zeus=%s, --force_emcee=%s",
                force_zeus,
                force_emcee,
            )

        for m in model_names:
            log.info(
                "  Model '%s': main engine=%s (vectorised=%s, touches CMB/BBN=%s)",
                m,
                K.engine_for_model.get(m, "auto"),
                bool(vectorised.get(m, False)),
                _model_has_any_cmb_or_bbn(CONFIG, m),
            )

        use_mpi   = bool(getattr(args, "use_mpi", False))
        num_cores = int(getattr(args, "num_cores", 1))
        parallel_flag = get_parallel_flag(use_mpi=use_mpi, num_cores=num_cores)

        if any_needs_pool and parallel_flag:
            if use_mpi:
                log.info(
                    "Parallel plan: MPI Pool for emcee / non-vectorised Zeus. "
                    "Vectorised Zeus remains single-core."
                )
            else:
                log.info(
                    "Parallel plan: local multiprocessing Pool with up to %d workers "
                    "for emcee / non-vectorised Zeus chains.",
                    num_cores,
                )
        elif any_needs_pool:
            log.info(
                "Parallel plan: engines could use a Pool, but parallelism disabled → serial."
            )
        else:
            log.info(
                "Parallel plan: selected engines do not require a worker Pool "
                "(vectorised zeus and emcee run in this process)."
            )

        # Item 42: a CMB group without a pool evaluates its walkers one after another
        cmb_groups = [f"{m}: {'+'.join(g)}" for m in model_names
                      for g in CONFIG[m].get("observations", []) if any(o in K.CMB_TAGS for o in g)]
        nw = max([int(CONFIG[m].get("nwalker", nwalkers)) for m in model_names] or [nwalkers])
        if cmb_groups and not parallel_flag:
            import platform as _platform
            why = ("native Windows has no worker pool (run Kosmulator in WSL)"
                   if _platform.system() == "Windows" else f"--num_cores {num_cores}")
            log.warning(
                "Serial CMB run (%s): %s. The walkers' CLASS + Planck evaluations (seconds each) "
                "run one after another on one core, so a converged chain takes days. Use "
                "--num_cores N (up to nwalker/2 = %d) or --use_mpi.",
                why, "; ".join(cmb_groups), max(1, nw // 2))
        if any_needs_pool and parallel_flag:
            n_workers = ((comm.Get_size() - 1) if comm is not None else 0) if use_mpi else num_cores
            if n_workers > max(1, nw // 2):
                log.info(
                    "%d workers for %d walkers: emcee and zeus move half the walkers at a time, "
                    "so only %d workers are busy and %d wait at each half-step.",
                    n_workers, nw, nw // 2, n_workers - nw // 2)

        # Item 12: r_d with the model's own E(z) (rd_Class, Kosmulator_main/rd_class.py) for
        # BAO/DESI groups calibrated by BBN or Planck, for models other than LCDM. Compiled once
        # and cached (Class/.cache/rd_Class); loaded here, before any worker pool exists, so the
        # workers only read the cached build.
        _bao = {"BAO", "DESI_DR1", "DESI_DR2"}
        _rdc_models = set()
        if K.RD_BACKGROUND == "model":
            _early = set(K.CMB_PRIMARY_TAGS)
            rdc_groups = [f"{m}: {'+'.join(g)}" for m in model_names if m not in K.RD_LCDM_MODELS
                          for g in CONFIG[m].get("observations", [])
                          if set(g) & _bao and (any(str(o).startswith("BBN") for o in g) or set(g) & _early)]
            if rdc_groups:
                try:
                    from Kosmulator_main import rd_class as RDC
                    RDC.ensure_ready(announce=True)
                    _rdc_models = {m for m in model_names if m not in K.RD_LCDM_MODELS}
                    log.info("r_d from CLASS with each model's own E(z) (rd_Class) for: %s",
                             "; ".join(rdc_groups))
                except Exception as e:
                    log.warning("rd_Class could not be built or loaded (%s): r_d from CLASS's flat-LCDM "
                                "background for every model instead (rd_background = \"lcdm\")", e)
                    K.set_rd_background("lcdm")

        # Item 15: BAO/DESI calibrated by BBN (no Planck data, so no CLASS build of
        # Kosmulator's is loaded) take r_d from CLASS if `import classy` works, otherwise
        # from the EH98 fit. The run goes on either way; say so before it starts.
        _bao = {"BAO", "DESI_DR1", "DESI_DR2"}
        rd_groups = [f"{m}: {'+'.join(g)}" for m in model_names if m not in _rdc_models
                     for g in CONFIG[m].get("observations", [])
                     if set(g) & _bao and any(str(o).startswith("BBN") for o in g)
                     and not set(g) & set(K.CMB_TAGS)]
        if rd_groups and RDH.classy_import_problem():
            # No classy loaded or installed: Kosmulator's own LCDM_v build (compiled once, then
            # cached) gives the flat-LCDM r_d; built here, before any worker pool exists, and only
            # with a compiler (without one every run would try and fail)
            import shutil as _sh
            _path = (CR._build_env() or os.environ).get("PATH")
            if all(_sh.which(t, path=_path) for t in ("gcc", "g++", "make", "ar")):
                try:
                    CR.ensure_class_ready(RDH.OWN_CLASS_MODEL, announce=True)
                    RDH._OWN_CLASS_FAILED = False
                except Exception as e:
                    log.warning("CLASS for %s could not be built or loaded: %s", RDH.OWN_CLASS_MODEL, e)
        _rd_problem = RDH.classy_import_problem() if rd_groups else None
        if _rd_problem:
            bar = "#" * 78
            log.warning(bar)
            log.warning("r_d WILL COME FROM THE EH98 FIT, NOT CLASS, for: %s", "; ".join(rd_groups))
            log.warning("  Reason: %s.", _rd_problem)
            log.warning("  Effect: %s. The run continues; the statistics notes repeat this.",
                        RDH.EH98_EFFECT)
            log.warning("  Fix: Kosmulator builds its own CLASS (Class/%s) when a C compiler is there; "
                        "kosmulator-doctor --build-class %s shows what is missing. Then rerun.",
                        RDH.OWN_CLASS_MODEL, RDH.OWN_CLASS_MODEL)
            log.warning(bar)

        log.info("────────────────────────────────────────────────────────")

    # ------------------------------------------------------------------
    # 11) Run MCMC
    # ------------------------------------------------------------------
    # One run per output folder: refuse to start if another live run is
    # writing to MCMC_Chains/<suffix> (it would interleave chain rows).
    if rank == 0:
        acquire_run_lock((getattr(args, "output_suffix", "") or "").strip())
        # Item 54: what this run is, for --peek (written after the lock: a run refused
        # because another one holds the folder must not overwrite that run's settings).
        # Chains already in the folder are reused (or resumed) unless --overwrite: refuse when
        # they were made with other likelihood settings, warn when that is unknown.
        from Kosmulator_main.peek import write_run_settings, check_run_settings
        _check = {
            "neutrinos": K.NEUTRINOS, "nu_mass_eV": K.NU_MASS_EV, "nu_N_eff": K.NU_N_EFF,
            "growth_model": K.GROWTH_MODEL, "cmb_nonlinear": K.CMB_NONLINEAR, "hmcode_version": K.HMCODE_VERSION,
            "cmb_nuisance": K.CMB_NUISANCE, "sample_H0_uncalibrated_SNe": bool(sample_H0_uncalibrated_SNe),
        }
        if any(m not in K.RD_LCDM_MODELS for m in model_names):   # r_d is the same for LCDM models
            _check["rd_background"] = K.RD_BACKGROUND
        if any(K.group_is_lensing_only(g) for g in observations):    # item 62
            _check["lensing_only_priors"] = bool(K.LENSING_ONLY_PRIORS)
        check_run_settings((getattr(args, "output_suffix", "") or "").strip(), _check,
                           overwrite=bool(getattr(args, "overwrite", False)))
        write_run_settings((getattr(args, "output_suffix", "") or "").strip(), {
            "models": list(model_names), "observations": [list(g) for g in observations],
            "nwalkers": nwalkers, "nsteps": nsteps, "burn": burn, "convergence": convergence,
            "cmb_nuisance": K.CMB_NUISANCE, "sample_H0_uncalibrated_SNe": bool(sample_H0_uncalibrated_SNe),
            "waic_include_cmb": bool(K.WAIC_INCLUDE_CMB), "neutrinos": K.NEUTRINOS,
            "nu_mass_eV": K.NU_MASS_EV, "nu_N_eff": K.NU_N_EFF, "growth_model": K.GROWTH_MODEL,
            "cmb_nonlinear": K.CMB_NONLINEAR, "hmcode_version": K.HMCODE_VERSION,
            "num_cores": int(getattr(args, "num_cores", 1)), "use_mpi": bool(getattr(args, "use_mpi", False)),
            "emcee_moves": K.EMCEE_MOVES, "zeus_light_mode": bool(K.ZEUS_LIGHT_MODE),
            "rd_background": K.RD_BACKGROUND, "lensing_only_priors": bool(K.LENSING_ONLY_PRIORS),
        })

    samples = run_mcmc_for_all_models(
        models=models,
        observations=observations,
        CONFIG=CONFIG,
        data=data,
        overwrite=bool(getattr(args, "overwrite", False)),
        convergence=convergence,
        PLOT_SETTINGS=PLOT_SETTINGS,
        use_mpi=bool(getattr(args, "use_mpi", False)),
        num_cores=int(getattr(args, "num_cores", 1)),
        vectorised=vectorised,
        suffix=getattr(args, "output_suffix", ""),
        resumeChains=bool(getattr(args, "resume", False)),
    )

    # ------------------------------------------------------------------
    # 12) Post-processing
    # ------------------------------------------------------------------
    try:
        if rank == 0:
            # What to cite for this run: Kosmulator, the data sets of its groups and the codes used
            # (written before the plots, so a plotting error does not lose it)
            from Kosmulator_main import citations as CIT
            _cite = CIT.write_run_citations(
                PLOT_SETTINGS.get("output_suffix", "default_run"), list(model_names),
                [list(g) for g in observations],
                engines=CIT.ENGINES_USED or set(getattr(K, "engine_for_model", {}).values()),
                settings={"emcee_moves": K.EMCEE_MOVES, "cmb_nonlinear": K.CMB_NONLINEAR,
                          "hmcode_version": K.HMCODE_VERSION},
                bbn_backends={o: d.get("bbn_model_effective") for o, d in (data or {}).items()
                              if isinstance(d, dict) and d.get("bbn_model_effective")})
            MP.generate_plots(samples, CONFIG, PLOT_SETTINGS, data, reference_model)
            if _cite:
                print(f"\nWhat to cite for this run (data, codes, acknowledgements): {_cite}")
            elapsed = format_elapsed_time(time.time() - t0)
            print_completion_banner(elapsed)
    finally:
        close_open_pools()

    return samples



def run_mcmc_for_all_models(
    models,
    observations: List[List[str]],
    CONFIG: Dict[str, Any],
    data: Dict[str, Any],
    overwrite: bool,
    convergence: float,
    PLOT_SETTINGS: Dict[str, Any],
    use_mpi: bool,
    num_cores: int,
    vectorised: Dict[str, bool],
    suffix: str = "",
    pool=None,
    pantheon_cov: Optional[np.ndarray] = None,
    resumeChains: bool = False,
):
    """
    Run MCMC simulations for all models and observation sets,
    choosing between Zeus and emcee based on vectorisation and Config.force_zeus.
    """
    All_Samples: Dict[str, Dict[str, Any]] = {}

    # Rank detection (safe if MPI is unavailable)
    try:
        comm = MPI.COMM_WORLD if MPI is not None else None  # type: ignore[attr-defined]
        rank = comm.Get_rank() if comm is not None else 0
    except Exception:  # pragma: no cover
        rank = 0

    # Determine which clik likelihoods we need
    need_hil     = any("CMB_hil"     in grp for grp in observations)
    need_hilTT   = any("CMB_hil_TT"  in grp for grp in observations)
    need_lowl    = any("CMB_lowl"    in grp for grp in observations)
    need_lowlTT  = any("CMB_lowl_TT" in K.cmb_group_components(grp) for grp in observations)   # P-ACT, SPA too
    need_lensing = any("CMB_lensing" in grp for grp in observations)

    # Define ALL paths first (plik or plik_lite: constants.CMB_NUISANCE)
    def _planck_path(tag: str, needed: bool):
        return os.path.join(K.OBSERVATIONS_BASE, K.planck_clik_file(tag)) if needed else None

    hil_path    = _planck_path("CMB_hil", need_hil)
    hiltt_path  = _planck_path("CMB_hil_TT", need_hilTT)
    lowl_path   = _planck_path("CMB_lowl", need_lowl)
    lowltt_path = _planck_path("CMB_lowl_TT", need_lowlTT)
    # Lensing files (Config.load_all_data provides both)
    lens_raw     = data.get("CMB_lensing_RAW") if need_lensing else None
    lens_cmbmarg = data.get("CMB_lensing_CMBMARGED") if need_lensing else None

    # Backward-compatible fallback (only if Config didn't populate them)
    if need_lensing and not lens_raw:
        lens_raw = os.path.join(
            K.OBSERVATIONS_BASE,
            "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing",
        )
    if need_lensing and not lens_cmbmarg:
        lens_cmbmarg = os.path.join(
            K.OBSERVATIONS_BASE,
            "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_CMBmarged.clik_lensing",
        )

    if rank == 0:
        _ground = sorted({o for g in observations for o in g if o in K.GROUND_CMB_TAGS})
        if _ground:
            # ACT, SPT and the Planck + ACT (+ SPT) combinations (item 57): load them here, so a
            # missing package stops the run now rather than in a worker
            from Kosmulator_main import ground_cmb as GC
            GC.check_available(_ground)
            for t in _ground:
                log.info("%s: %s (%d data points)", t, " + ".join(K.cmb_components(t)), GC.n_data(t))
            log.info("CLASS spectra to l = %d for every CMB group of this run (ACT needs 8501, SPT 4095, "
                     "Planck 3000): a CMB point costs more than with Planck alone", K.CMB_CLASS_LMAX)
            for grp in observations:
                tp = K.cmb_tau_prior(grp)
                if tp:
                    log.info("Group %s has no low-l EE likelihood: tau prior %.4f +- %.4f (the one of "
                             "%s's own likelihood)", grp, tp[0], tp[1],
                             "SPT-lite" if "CMB_SPT" in K.cmb_group_components(grp) else "ACT-lite")
            if not K.CMB_NONLINEAR:
                log.warning("cmb_nonlinear = None with %s: ACT and SPT reach l = 8500 and 4000, where "
                            "non-linear lensing matters (ACT DR6 and SPT-3G D1 used HMcode 2020); set "
                            "cmb_nonlinear = \"hmcode\" in Kosmulator.py", ", ".join(_ground))
        _lens = sorted({o for g in observations for o in g if o in K.EXT_LENSING_TAGS})
        if _lens:
            # ACT DR6 + Planck PR4 and SPT-3G MUSE lensing (item 61): load the data now (ACT + Planck
            # per group's mode; the corrected one writes its cache here, before any pool)
            from Kosmulator_main import ground_cmb as GC
            for grp in observations:
                for t in grp:
                    if t == "CMB_lensing_ACTPlanck":
                        lens_only = not K.group_has_primary_cmb(grp)
                        GC.check_lensing(t, lens_only)
                        log.info("%s in %s: %s, %d band powers, %s", t, grp, K.ACT_LENS_VARIANT, GC.n_data(t),
                                 "CMB-marginalised covariance (no primary CMB in the group)" if lens_only else
                                 "likelihood corrections from the group's own CMB spectra")
                    elif t in K.EXT_LENSING_TAGS:
                        GC.check_lensing(t, False)
                        log.info("%s in %s: MUSE phi phi, %d band powers to L = %d (CLASS spectra to l = %d)",
                                 t, grp, GC.n_data(t), GC.spt_muse().lmax, K.CMB_CLASS_LMAX)
            if not K.CMB_NONLINEAR:
                log.warning("cmb_nonlinear = None with %s: the lensing power spectrum depends on non-linear "
                            "growth at the multipoles these data use (at the Planck 2018 best fit, linear "
                            "instead of HMcode raises chi^2 by 2.4 - 3.6 for ACT + Planck and 19 for SPT "
                            "MUSE); set cmb_nonlinear = \"hmcode\" in Kosmulator.py", ", ".join(_lens))
        _lens_only = [list(g) for g in observations if K.group_is_lensing_only(g)]
        if _lens_only and not K.LENSING_ONLY_PRIORS:
            log.warning("lensing_only_priors = False: %s sample n_s and tau_reio within their top-hat priors; "
                        "lensing alone leaves n_s nearly free, and tau_reio enters none of these likelihoods "
                        "(item 62)", _lens_only)

    # Planck likelihood code (clik, or clipy when clik is missing; see Class_run)
    if need_hil or need_hilTT or need_lowl or need_lowlTT or need_lensing:
        if CR.clik is None:
            raise ImportError(
                "CMB likelihoods need clik or clipy. Easiest: pip install \"clipy-like>=0.15\" "
                "(pure Python); or build CLIK as in the README's advanced installation."
            )
        if need_lowl:
            problem = CR.clipy_lowl_problem()
            if problem:
                raise ImportError(problem)
        # Compiled clik holds one plik_lite likelihood per process ("plik_cmbonly
        # already initialized"); clipy has no such limit.
        if K.CMB_NUISANCE == "lite" and need_hil and need_hilTT and CR.CLIK_BACKEND == "clik":
            raise RuntimeError(
                "cmb_nuisance = \"lite\" with both CMB_hil and CMB_hil_TT in one run: compiled clik "
                "can load only one plik_lite likelihood per process. Run them separately, or use "
                "clipy (pip install \"clipy-like>=0.15\" and make clik unimportable)."
            )
        if rank == 0:
            log.info("Planck likelihood code: %s (%s)", CR.CLIK_BACKEND, getattr(CR.clik, "__file__", ""))
            if need_hil or need_hilTT:
                log.info("Planck high-l likelihood: %s (cmb_nuisance = %s; %d nuisance parameters sampled for "
                         "TT,TE,EE, %d for TT)", "plik_lite" if K.CMB_NUISANCE == "lite" else "plik", K.CMB_NUISANCE,
                         len(K.planck_sampled_nuisances("CMB_hil")), len(K.planck_sampled_nuisances("CMB_hil_TT")))
                if K.CMB_NUISANCE == "extended":
                    log.warning("cmb_nuisance = \"extended\": the three polarisation efficiencies and the six EE dust "
                                "amplitudes that Planck's baseline fixes are sampled with their stated Gaussian priors "
                                "(a robustness check of the baseline, not Planck's own analysis).")
                elif K.CMB_NUISANCE == "full":
                    log.warning("cmb_nuisance = \"full\": all 47 plik inputs are sampled. 17 of them (A_pol, cib_index, "
                                "the dust indices, the correlated-noise and subpixel template amplitudes) have no published "
                                "uncertainty; their flat boxes are Kosmulator's choice (constants.PLANCK_NUISANCE_FULL_EXTRA). "
                                "Planck never published this analysis, so there is no reference result; expect slow "
                                "convergence and prior-dominated nuisances.")
            # Planck's "TT" includes the low-l TT likelihood (Commander)
            for grp in observations:
                if ("CMB_hil" in grp or "CMB_hil_TT" in grp) and "CMB_lowl_TT" not in grp:
                    log.warning("Group %s has Planck high-l without CMB_lowl_TT (Commander, l = 2 - 29); "
                                "Planck 2018's TT,TE,EE+lowE and TT+lowE combinations include it.", grp)

    # Sanity-check clik files early (rank 0 only)
    if rank == 0:
        for name, path in [
            ("Plik TTTEEE", hil_path),
            ("Plik TT",     hiltt_path),
            ("SimAll EE",   lowl_path),
            ("Commander TT", lowltt_path),
            ("Lensing RAW",      lens_raw),
            ("Lensing CMBmarg",  lens_cmbmarg),
        ]:
            if path and not os.path.exists(path):
                raise FileNotFoundError(f"{name} clik file not found at: {path}")

    # Preload clik likelihoods once per process
    if hil_path:
        CR.preload_clik_hil(hil_path)
    if hiltt_path:
        CR.preload_clik_hilTT(hiltt_path)
    if lowl_path:
        CR.preload_clik_lowl(lowl_path)
    if lowltt_path:
        CR.preload_clik_lowlTT(lowltt_path)

    if lens_raw:
        CR.preload_clik_lensing_raw(lens_raw)
    if lens_cmbmarg:
        CR.preload_clik_lensing_cmbmarg(lens_cmbmarg)

    # Decide per-model whether we actually need a pool (lazy creation)
    parallel_flag = get_parallel_flag(use_mpi, num_cores)
    pool_cache = {"mpi": None, "local": None}

    # Threads of this process's own CLASS/BLAS work (item 45): the cores the run was given on
    # this machine, which wait while the main process works: --num_cores with a local pool,
    # the ranks on rank 0's machine under MPI, 1 for a serial run
    if parallel_flag and use_mpi:
        n_main = int(getattr(K, "MPI_NODE_RANKS", 1) or 1)
    elif parallel_flag:
        n_main = num_cores
    else:
        n_main = 1
    main_threads = set_main_threads(n_main)
    if rank == 0:
        log.info("Threads for CLASS and BLAS in the main process: %s", main_threads)
        if K.NEUTRINOS == "massive":
            log.info("Neutrinos: one massive of %g eV plus massless ones, N_eff = %g in total",
                     K.NU_MASS_EV, K.NU_N_EFF)
        else:
            log.info("Neutrinos: all massless, N_eff = %g", K.NU_N_EFF)

    # Build the model list from either dict or list
    model_list = list(models.keys()) if isinstance(models, dict) else list(models)
    engine_mode = getattr(K, "engine_mode", "mixed")

    def _model_needs_pool(m: str) -> bool:
        """
        A group of model m evaluates its likelihood in the worker pool: emcee for
        CMB/BBN groups (also in engine mode "fastest", where main records no
        model-level engine), a model that cannot be vectorised, or a pool-preferred
        dataset (official JLA). The same rule as any_needs_pool in main.
        """
        cv = bool(vectorised.get(m, False))
        eng = (getattr(K, "engine_for_model", {}) or {}).get(m)
        if eng not in ("zeus", "emcee"):
            eng = "zeus" if (cv and (zeus is not None)) else "emcee"
        touches = _model_has_any_cmb_or_bbn(CONFIG, m)
        return bool(
            (touches and (eng == "emcee" or engine_mode == "fastest"))
            or not cv
            or any(o in getattr(K, "POOL_PREFERRED_DATASETS", set())
                   for grp in CONFIG[m].get("observations", []) for o in grp)
        )

    any_model_needs_pool = any(_model_needs_pool(m) for m in model_list)

    for model_name in model_list:
        # -------------------------------
        # 0) Model capabilities (always define)
        # -------------------------------
        can_vec = bool(vectorised.get(model_name, False))

        # -------------------------------
        # 1) Decide sampling engine (single source of truth)
        # -------------------------------
        eng_map = getattr(K, "engine_for_model", {}) or {}
        engine = eng_map.get(model_name)

        if engine not in ("zeus", "emcee"):
            # Ultra-safe fallback (should never happen)
            engine = "zeus" if (can_vec and (zeus is not None)) else "emcee"

        # -------------------------------
        # 2) Decide if this model needs a pool
        # -------------------------------
        # In mixed/fastest/single engine modes, the per-observation engine can
        # differ from this model-level guess. For example, the model may be
        # vectorised overall (can_vec=True) but a BBN+DESI group will still use
        # emcee (and really wants a Pool).
        #
        # To avoid starving those runs of parallelism, we create a Pool whenever
        # parallelism is allowed for this model. run_mcmc() will then decide
        # per observation whether to actually use it (vectorised Zeus ignores it).
        # Only create workers when this model has a group that uses them: emcee,
        # non-vectorised zeus, or a pool-preferred dataset (official JLA).
        # Vectorised zeus never uses the pool, so creating one there only left
        # idle worker processes. Under MPI the extra ranks must wait in the pool
        # from the first model on (otherwise they would walk the group loop of the
        # models rank 0 runs alone), so the pool is built at the first model when
        # any model needs it; when none does, the extra ranks have already exited
        # (step 9 of main) and no MPIPool is built for ranks that are gone.
        model_needs_pool = _model_needs_pool(model_name)
        need_pool = parallel_flag and (model_needs_pool or (use_mpi and any_model_needs_pool))

        # BLAS/OpenMP threads for groups that sample in this process without a pool
        # are set per group in Kosmulator_MCMC (constants.MAIN_BLAS_THREADS_NO_POOL).

        pool_for_model = None
        if need_pool:
            key = "mpi" if use_mpi else "local"
            if pool_cache[key] is None:
                # Prepare lensing paths once from the data dict
                lens_raw     = data.get("CMB_lensing_RAW")
                lens_cmbmarg = data.get("CMB_lensing_CMBMARGED")

                pool_cache[key] = get_pool(
                    use_mpi=use_mpi,
                    num_cores=num_cores,
                    initializer=CR.init_clik_worker,
                    initargs=(
                        hil_path,
                        lowl_path,
                        hiltt_path,
                        lens_raw,
                        lens_cmbmarg,
                        lowltt_path,
                    ),
                )
            pool_for_model = pool_cache[key]

        # Helpful warning if user forced Zeus but it isn't installed
        if getattr(K, "force_zeus", False) and (zeus is None) and (rank == 0):
            log.warning("--force_zeus requested but 'zeus' is not installed; falling back to emcee.")


        # -------------------------------
        # 3) Banner for this model
        # -------------------------------
        if rank == 0:
            touches_cmb_bbn = _model_has_any_cmb_or_bbn(CONFIG, model_name)
            print_model_banner(
                model_name=model_name,
                engine_mode=engine_mode,
                can_vec=bool(vectorised.get(model_name, False)),
                touches_cmb_bbn=_model_has_any_cmb_or_bbn(CONFIG, model_name),
                single_engine_map=K.engine_for_model,
                pool_kind=("serial" if pool_for_model is None else ("mpi" if use_mpi else "local")),
            )
            #print(bar, flush=True)

        MODEL = UDM.Get_model_function(model_name)
        obs_list: List[List[str]] = CONFIG[model_name]["observations"]
        Samples: Dict[str, Any] = {}

        # Prepare CLASS once per model if any CMB likelihood is involved
        did_class_prep = False

        for i, obs_set in enumerate(obs_list):
            if rank == 0:
                print("-" * 66)

            # One-time CLASS run for CMB pipelines (avoid repeated heavy prep)
            if not did_class_prep and any(x in obs_set for x in K.CMB_TAGS):
                no_rebuild = str(os.environ.get("KOSM_NO_CLASS_REBUILD", "")).strip().lower() in ("1", "true", "yes", "on")
                ok = CR.ensure_class_ready(
                    model_name,
                    force=False,
                    no_rebuild=no_rebuild,
                    announce=(rank == 0),
                    recheck=True,   # hash the sources once per model; per-point calls skip it
                )
                if not ok and no_rebuild:
                    raise RuntimeError("classy missing and rebuild forbidden (KOSM_NO_CLASS_REBUILD=1).")
                if not ok:
                    # ultra-conservative fallback (should be rare)
                    CR.run_model(model_name)
                    # and load it explicitly so the right binary is active
                    CR.ensure_class_ready(model_name, force=False, no_rebuild=False)
                did_class_prep = True

            # Work on a copy so per-obs modifications don't leak into the next set
            data_work = deepcopy(data)
            data_work = apply_pantheon_cov(data_work, obs_set, pantheon_cov)

            if "PantheonP" in data_work:
                p = data_work["PantheonP"]
                n_tot  = int(np.asarray(p.get("zHD", [])).size)
                idx    = np.asarray(
                    p.get("indices", np.where(p.get("mask", np.ones(n_tot, dtype=bool)))[0]),
                    int,
                )
                n_used = int(idx.size)
                n_cal  = int(np.sum(np.asarray(p.get("IS_CALIBRATOR", []), int)))
                covN   = None if p.get("cov", None) is None else p["cov"].shape[0]
                # Debug hook if needed:
                # print(f"[diag][Pantheon+] N_total={n_tot}  N_used={n_used}  N_calibrators={n_cal}  cov_N={covN}")

            key = cfg_generate_label(
                obs_set,
                config_model=CONFIG[model_name],
                obs_index=i,
            )  # e.g. "CC+PantheonP_SH0ES" or "CC+PantheonP"
            # plik_lite, extended and full chains go to their own folders (_plik_lite, ...), so a
            # run never loads or resumes a chain of the same group made with another Planck set
            out_key = key.replace("+", "_")
            if K.CMB_NUISANCE != "baseline" and any(o in ("CMB_hil", "CMB_hil_TT") for o in obs_set):
                out_key += f"_plik_{K.CMB_NUISANCE}"
            output_dir = prepare_output(
                model_name,
                out_key,
                suffix,
            )  # .../LCDM_v/CC_PantheonP_SH0ES

            if rank == 0:
                # Use the resolved key to decide Pantheon display (Pantheon+ or Pantheon+SH0ES)
                resolved_parts = key.split("+")
                for a, obs_name in enumerate(obs_set):
                    disp_name = obs_name
                    if obs_name == "PantheonP":
                        # Find the corresponding resolved token for PantheonP
                        rp = next((p for p in resolved_parts if p.startswith("PantheonP")), "PantheonP")
                        disp_name = "Pantheon+SH0ES" if rp == "PantheonP_SH0ES" else "Pantheon+"
                    obs_type = CONFIG[model_name]["observation_types"][i][a]
                    print(
                        "Observations:             "
                        f"\033[34m{disp_name}\033[0m data (aka "
                        f"\033[34m{obs_type}\033[0m data)"
                    )

            # Run (or load) the chain
            if rank == 0:
                last_obs = (i == len(obs_list) - 1)
                Samples[key] = load_or_run_chain(
                    output_dir=output_dir,
                    chain_file=f"{key}.h5",
                    overwrite=overwrite,
                    CONFIG_model=CONFIG[model_name],
                    data=data_work,
                    MODEL_func=MODEL,
                    convergence=convergence,
                    parallel=parallel_flag,
                    pool=pool_for_model,
                    vectorised=can_vec,
                    resumeChains=resumeChains,
                    model_name=model_name,
                    obs=obs_set,
                    Type=CONFIG[model_name]["observation_types"][i],
                    colors=PLOT_SETTINGS["color_schemes"][i],
                    last_obs=last_obs,
                    PLOT_SETTINGS=PLOT_SETTINGS,
                    obs_index=i,
                    use_mpi=use_mpi,
                    num_cores=num_cores,
                    obs_key=key,
                )
                # A chain made with other settings can have other parameters (with growth_model
                # "gamma", f and f_sigma_8 groups sample gamma; with "geff" they do not)
                _s = Samples[key].get("samples") if isinstance(Samples[key], dict) else Samples[key]
                _pars = list(CONFIG[model_name]["parameters"][i])
                if _s is not None and getattr(_s, "ndim", 0) == 2 and _s.shape[1] != len(_pars):
                    raise RuntimeError(
                        f"{model_name} {key}: the chain in {output_dir} has {_s.shape[1]} parameters, but this "
                        f"run's group has {len(_pars)} ({', '.join(_pars)}). It was made with other settings "
                        "(for example growth_model). Use --overwrite or another --output_suffix.")
                # Where this group's chain lives, for the convergence report
                PLOT_SETTINGS.setdefault("chain_index", {})[(model_name, key)] = {
                    "dir": output_dir, "key": key, "obs_index": i,
                }

            if rank == 0:
                print("-" * 66)
                print("\n")

            # Clean up per-obs temporary covariance attachments
            data_work = cleanup_pantheon_cov(data_work)

        All_Samples[model_name] = Samples

    # Keep the workers for the statistics and plots (item 52): their CLASS runs (CMB
    # band draws, CMB WAIC draws) go through the pool; main() closes it afterwards
    pools = [p for p in getattr(locals().get("pool_cache", {}), "values", lambda: [])() if p is not None]
    _OPEN_POOLS.extend(pools)
    if rank == 0 and pools:
        set_post_pool(pools[0])

    return All_Samples


#: Pools kept open after sampling for the statistics and plots (item 52)
_OPEN_POOLS: list = []


def announce_growth_and_lensing(CONFIG, model_names) -> None:
    """
    Start-up lines on how each model's growth data (f, f sigma_8) and lensed CMB spectra are
    computed (growth_model, cmb_nonlinear in Kosmulator.py), with warnings where that is an
    approximation for the model: GR growth for a model without <model>_Geff, non-linear
    fits calibrated on GR simulations, G_eff in the growth but GR perturbations in CLASS.
    """
    import User_defined_modules as UDM

    def _tags(m):
        cfg = CONFIG.get(m, {})
        return ({str(o) for g in cfg.get("observations", []) for o in g}
                | {str(t) for ts in cfg.get("observation_types", []) for t in ts})

    growth = [m for m in model_names if m in CONFIG and {"f", "f_sigma_8"} & _tags(m)]
    cmb = [m for m in model_names if m in CONFIG and any(t.startswith("CMB") for t in _tags(m))]
    bar = "#" * 78
    if growth and K.GROWTH_MODEL == "gamma":
        log.info("Growth (f, f sigma_8): growth index, f = Omega_m(z)^gamma (growth_model = 'gamma')")
        unused = [m for m in growth if UDM.get_geff_function(m) is not None]
        if unused:
            log.info("  G_eff functions of %s are not used with growth_model = 'gamma' ('geff' uses them)",
                     ", ".join(unused))
    elif growth:
        for m in growth:
            if UDM.get_geff_function(m) is not None:
                log.info("Growth of %s: linear growth equation with G_eff from %s_Geff", m, m)
            elif m in K.GR_GRAVITY_MODELS:
                log.info("Growth of %s: linear growth equation, GR (G_eff = G)", m)
            else:
                log.warning(bar)
                log.warning("NO G_eff FUNCTION FOR %s: its growth (f, f sigma_8) is computed with G_eff = G,", m)
                log.warning("  that is the GR (LCDM) growth equation on %s's own expansion history.", m)
                log.warning("  If %s changes gravity, its true f and f sigma_8 can differ from these, and", m)
                log.warning("  so can the fit and the model comparison. Fix: add %s_Geff(z, p), returning", m)
                log.warning("  G_eff/G, to User_defined_modules.py (example: f1CDM_v_Geff); or, if %s is a", m)
                log.warning("  GR model with conserved matter (dark energy only, no interaction), add it to")
                log.warning("  constants.GR_GRAVITY_MODELS. The growth equation assumes conserved matter")
                log.warning("  (rho_m ~ a^-3): interacting dark-sector models need their own growth equation.")
                log.warning(bar)
        for m in growth:
            if m in cmb and UDM.get_geff_function(m) is not None:
                log.warning("%s: f and f sigma_8 use %s_Geff, but its CLASS build computes the CMB with GR "
                            "perturbations, so the two parts of the fit treat gravity differently", m, m)
    if cmb:
        log.info("Non-linear correction of the CMB lensing: %s, the same for every model (cmb_nonlinear)",
                 K.nonlinear_description())
        mg = [m for m in cmb if m not in K.GR_GRAVITY_MODELS]
        if K.CMB_NONLINEAR and mg:
            log.warning("%s for %s: these fits are calibrated on simulations with GR, so for a model that "
                        "changes gravity on small scales they are an approximation (cmb_nonlinear = None "
                        "in Kosmulator.py: linear lensing for every model)", K.nonlinear_description(), ", ".join(mg))


def close_open_pools() -> None:
    """Close the pools kept for the statistics and plots (item 52)."""
    set_post_pool(None)
    while _OPEN_POOLS:
        p = _OPEN_POOLS.pop()
        try:
            if hasattr(p, "close"):
                p.close()
            if hasattr(p, "join"):
                p.join()
        except Exception:
            pass
