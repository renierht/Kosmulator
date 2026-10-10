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
    prepare_output,
    load_or_run_chain,
    init_summary_context,
    issue_observation_warnings,   # moved from here into utils.py
)
import Plots.Plots as MP
from Kosmulator_main import Class_run as CR
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
    K.set_engine_overrides(
        force_vec=getattr(args, "force_vectorisation", False),
        disable_vec=getattr(args, "disable_vectorisation", False),
        force_z=getattr(args, "force_zeus", False),
        force_e=getattr(args, "force_emcee", False),
        mode=getattr(args, "engine_mode", "mixed"),
    )

    # Planck high-l likelihood: plik with Planck's nuisance parameters, or plik_lite.
    # Set before the config is built (the sampled nuisances and the files depend on it).
    _mode = str(cmb_nuisance).strip().lower()
    if _mode not in K.CMB_NUISANCE_MODES:
        raise ValueError(f"cmb_nuisance must be one of {K.CMB_NUISANCE_MODES}, not {cmb_nuisance!r}")
    K.CMB_NUISANCE = _mode
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
        if rank != 0:
            try:
                if comm is not None:
                    from mpi4py import MPI as _MPI
                    _MPI.Finalize()
            finally:
                os._exit(0)
        else:
            print(
                "MPI requested but every group runs vectorised in this process (zeus or emcee) → "
                "proceeding on rank 0; extra ranks exited."
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

        if any_needs_pool and parallel_flag and num_cores > 1:
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

        log.info("────────────────────────────────────────────────────────")

    # ------------------------------------------------------------------
    # 11) Run MCMC
    # ------------------------------------------------------------------
    # One run per output folder: refuse to start if another live run is
    # writing to MCMC_Chains/<suffix> (it would interleave chain rows).
    if rank == 0:
        acquire_run_lock((getattr(args, "output_suffix", "") or "").strip())

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
    if rank == 0:
        MP.generate_plots(samples, CONFIG, PLOT_SETTINGS, data, reference_model)
        elapsed = format_elapsed_time(time.time() - t0)
        print_completion_banner(elapsed)

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
    need_lowlTT  = any("CMB_lowl_TT" in grp for grp in observations)
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
                log.info("Planck high-l likelihood: %s (cmb_nuisance = %s)",
                         "plik_lite" if K.CMB_NUISANCE == "lite" else "plik", K.CMB_NUISANCE)
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
            # plik_lite chains go to their own folder, so a lite run never loads or
            # resumes a plik chain of the same group (and the reverse)
            out_key = key.replace("+", "_")
            if K.CMB_NUISANCE == "lite" and any(o in ("CMB_hil", "CMB_hil_TT") for o in obs_set):
                out_key += "_plik_lite"
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

    # Tidy up any workers we created
    for p in getattr(locals().get("pool_cache", {}), "values", lambda: [])():
        if p is not None:
            try:
                if hasattr(p, "close"):
                    p.close()
                if hasattr(p, "join"):
                    p.join()
            except Exception:
                pass

    return All_Samples
