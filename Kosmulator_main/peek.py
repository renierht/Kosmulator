"""
Look at a run while it is going, or afterwards (item 54).

    python Kosmulator.py --peek --output_suffix <suffix>
    python Kosmulator.py --peek --output_suffix <suffix> --peek_burn 500 --corner-show-all-cmb-params

For every model and group of the run that has a chain so far, the chain file is
copied and only the copy is read: opening the running file would take an HDF5
lock, and the run could then fail at its next write. The newest step is dropped
(it may have been half written when the file was copied). Kosmulator's own corner
plot goes to Plots/Saved_Plots/Peek/<suffix>/<model>/corner_plots/, so the run's
own plots are never overwritten; the means and standard deviations so far and the
convergence table so far (the same rule the run stops on) are printed.

The run writes MCMC_Chains/<suffix>/run_settings.json at start-up (models, groups,
walkers, steps, burn-in, switches), so a peek needs only the run's name. For runs
started before that file existed, the models, groups and Planck likelihood set are
read from the chain folders and the burn-in defaults to half the steps so far.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

SETTINGS_FILE = "run_settings.json"
COPY_PREFIX = "kosm_peek_"
#: Copies older than this are from a peek that was killed outright (a running peek's copies
#: are minutes old); they are removed at the start of the next peek
STALE_COPY_AGE_H = 12.0


def _chain_root(suffix: str) -> str:
    return os.path.join("MCMC_Chains", suffix) if suffix else "MCMC_Chains"


def write_run_settings(suffix: str, settings: Dict[str, Any]) -> Optional[str]:
    """Write the run's settings next to its chains (rank 0, at start-up)."""
    try:
        folder = _chain_root(suffix)
        os.makedirs(folder, exist_ok=True)
        info = dict(settings)
        info.setdefault("started", time.strftime("%Y-%m-%d %H:%M:%S"))
        info.setdefault("argv", list(sys.argv))
        try:
            info.setdefault("git_commit", subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip() or None)
        except Exception:
            info.setdefault("git_commit", None)
        path = os.path.join(folder, SETTINGS_FILE)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(info, fh, indent=2, default=str)
        os.replace(tmp, path)
        return path
    except Exception as e:                      # a peek convenience; never stops a run
        log.warning("Could not write %s for --peek: %s", SETTINGS_FILE, e)
        return None


#: Run settings that change the likelihood: chains made with other values must not be reused
LIKELIHOOD_SETTINGS = ("neutrinos", "nu_mass_eV", "nu_N_eff", "growth_model", "cmb_nonlinear",
                       "hmcode_version", "cmb_nuisance", "sample_H0_uncalibrated_SNe", "rd_background",
                       "lensing_only_priors")


def check_run_settings(suffix: str, settings: Dict[str, Any], overwrite: bool = False) -> None:
    """
    Before a run reuses (or resumes) the chains already in MCMC_Chains/<suffix>: stop if they
    were made with other likelihood settings (run_settings.json), warn if that is unknown (runs
    started before run_settings.json existed). Nothing to check with --overwrite or no chains.
    """
    root = _chain_root(suffix)
    if overwrite or not glob.glob(os.path.join(root, "**", "*.h5"), recursive=True):
        return
    path = os.path.join(root, SETTINGS_FILE)
    if not os.path.exists(path):
        log.warning("The chains in %s were made before Kosmulator recorded its run settings. Kosmulator "
                    "reuses them with the current settings (neutrinos %s, CMB lensing %s, growth_model %s), "
                    "but runs before October 2026 used massless neutrinos and linear CMB lensing: to "
                    "reprocess such chains consistently set neutrinos = \"massless\" and nu_N_eff = 3.044 "
                    "in Kosmulator.py (and keep cmb_nonlinear = None); to start new chains use --overwrite "
                    "or another --output_suffix.", root, settings.get("neutrinos"),
                    settings.get("cmb_nonlinear") or "linear", settings.get("growth_model"))
        return
    try:
        with open(path) as fh:
            old = json.load(fh)
    except Exception as e:
        log.warning("Could not read %s (%s); the run settings of its chains are not checked", path, e)
        return
    diff = []
    old = dict(old)
    if "rd_background" not in old:
        old["rd_background"] = "lcdm"     # runs before item 12: flat-LCDM r_d for every model
    old.setdefault("lensing_only_priors", False)   # runs before item 62 sampled n_s and tau freely
    for k in LIKELIHOOD_SETTINGS:
        if k not in old or k not in settings:
            continue
        if k == "hmcode_version" and not (old.get("cmb_nonlinear") == settings.get("cmb_nonlinear") == "hmcode"):
            continue
        if k == "nu_mass_eV" and not (old.get("neutrinos") == settings.get("neutrinos") == "massive"):
            continue
        a, b = old[k], settings[k]
        same = (abs(float(a) - float(b)) < 1e-12) if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) else (a == b)
        if not same:
            diff.append(f"{k}: chains {a!r}, now {b!r}")
    if diff:
        from Kosmulator_main.utils import refuse          # CT-33: no traceback
        refuse(
            f"The chains in {root} were made with other settings ({'; '.join(diff)}), so reusing or "
            "resuming them would mix two likelihoods. Set these back in Kosmulator.py, or start new "
            "chains with --overwrite or another --output_suffix.")


def _remove_stale_copies(max_age_h: float = STALE_COPY_AGE_H) -> int:
    """Remove chain copies left in the temp folder by a peek that was killed outright
    (a peek removes its own copies when it ends, also after an error or Ctrl-C)."""
    n = 0
    for d in glob.glob(os.path.join(tempfile.gettempdir(), COPY_PREFIX + "*")):
        try:
            if os.path.isdir(d) and time.time() - os.path.getmtime(d) > max_age_h * 3600.0:
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        except OSError:
            pass
    if n:
        log.info("--peek: removed %d chain copies left by an earlier peek", n)
    return n


def _settings_from_folders(suffix: str) -> Dict[str, Any]:
    """Models, groups and Planck set of a run without run_settings.json, from its folders."""
    root = _chain_root(suffix)
    models, groups, mode = [], {}, "baseline"
    for mdir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(mdir):
            continue
        model = os.path.basename(mdir)
        for gdir in sorted(glob.glob(os.path.join(mdir, "*"))):
            for f in sorted(glob.glob(os.path.join(gdir, "*.h5"))):
                key = os.path.basename(f)[:-3]
                key = key[:-5] if key.endswith("_zeus") else key
                tokens = key.split("+")
                if model not in models:
                    models.append(model)
                groups.setdefault(model, [])
                if tokens not in groups[model]:
                    groups[model].append(tokens)
                for _m in ("lite", "extended", "full"):
                    if os.path.basename(gdir).endswith(f"_plik_{_m}"):
                        mode = _m
    obs = []
    for m in models:
        for g in groups[m]:
            if g not in obs:
                obs.append(g)
    return {"models": models, "observations": obs, "cmb_nuisance": mode}


def _copy_and_read(src: str, tmpdir: str, engine: str):
    """Copy a chain file, read the copy without its newest step; also trim the copy so
    the convergence summary reads the same steps. Returns (chain [steps, walkers, dim], path)."""
    last = None
    for _ in range(5):                         # the run may be writing a step at this moment
        dst = os.path.join(tmpdir, os.path.basename(src))
        shutil.copyfile(src, dst)
        try:
            import h5py
            with h5py.File(dst, "r+") as h:
                if engine == "zeus":
                    n = int(h["samples"].shape[0])
                    n = n - 1 if n > 1 else n
                    chain = np.asarray(h["samples"][:n], float)
                    for name in ("samples", "logprob", "log_prob_chain"):
                        if name in h and h[name].shape[0] > n:
                            h[name].resize(n, axis=0)
                else:
                    g = h["mcmc"]
                    n = int(g.attrs["iteration"])
                    n = n - 1 if n > 1 else n
                    chain = np.asarray(g["chain"][:n], float)
                    g.attrs["iteration"] = n
            return chain, dst
        except Exception as e:
            last = e
            time.sleep(3)
    raise RuntimeError(f"could not read a consistent copy of {src}: {last}")


#: Per-group lists of a model's CONFIG (Config.create_config); dicts keyed by group index end in "_by_group"
_PER_GROUP_LISTS = ("observations", "observation_types", "parameters", "reference_values", "prior_limits",
                    "ndim", "nwalker_by_obs")


def _subset_groups(cfg: dict, keep: List[int]) -> dict:
    """A copy of a model's CONFIG with only the groups `keep` (per-group lists, and the dicts keyed by
    group index renumbered), for a corner plot of the groups that have chains so far (CT-28)."""
    n = len(cfg["observations"])
    out = dict(cfg)
    for k, v in cfg.items():
        if k in _PER_GROUP_LISTS and isinstance(v, list) and len(v) == n:
            out[k] = [v[j] for j in keep]
        elif k.endswith("_by_group") and isinstance(v, dict):
            out[k] = {i: v[j] for i, j in enumerate(keep) if j in v}
    return out


def peek(suffix: str, burn: Optional[int] = None, all_params: bool = False,
         latex_enabled: bool = False) -> List[str]:
    """Corner plot, means and convergence so far of every chain of run <suffix>. Returns the plot paths."""
    # one thread: the run's workers keep the cores
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")    # only our copies are opened
    _remove_stale_copies()

    import Kosmulator as KZ
    import User_defined_modules as UDM
    from Kosmulator_main import Config as CF, constants as K, utils as U, Class_run as CR
    from Plots import Plots as P
    from Plots.Plot_functions import normalize_save_roots

    root = _chain_root(suffix)
    if not os.path.isdir(root):
        raise SystemExit(f"--peek: no chains for --output_suffix {suffix!r} (looked in {root})")
    sfile = os.path.join(root, SETTINGS_FILE)
    if os.path.exists(sfile):
        with open(sfile) as fh:
            st = json.load(fh)
        source = SETTINGS_FILE
    else:
        st = _settings_from_folders(suffix)
        source = "the chain folders (no run_settings.json: run started before --peek existed)"
    models = list(st.get("models") or [])
    observations = [list(g) for g in (st.get("observations") or [])]
    if not models or not observations:
        raise SystemExit(f"--peek: no chain files yet under {root} (has sampling started?)")

    if st.get("neutrinos"):           # the run's neutrinos (r_d column of the corner table)
        K.set_neutrinos(st["neutrinos"], st.get("nu_mass_eV", K.NU_MASS_EV), st.get("nu_N_eff", K.NU_N_EFF))
    # the run's growth model decides whether gamma is a parameter of its f / f_sigma_8 groups
    K.set_growth_model(st.get("growth_model", "gamma"))
    # runs whose settings file has no cmb_nonlinear were made with linear lensing
    K.set_cmb_nonlinear(st.get("cmb_nonlinear"), st.get("hmcode_version", K.HMCODE_VERSION))
    K.CMB_NUISANCE = str(st.get("cmb_nuisance", "baseline")).lower()
    if K.CMB_NUISANCE == "lite":
        for _t in ("CMB_hil", "CMB_hil_TT"):
            _tex, _txt = K.OBS_PRETTY_MAP[_t]
            if not _txt.endswith(" lite"):
                K.OBS_PRETTY_MAP[_t] = (_tex + " lite", _txt + " lite")

    print(f"Peek at run {suffix!r} (settings from {source})")
    with U.quiet_cstdio():
        CONFIG, _ = CF.create_config(
            models=UDM.Get_model_names(models), reference_values=dict(KZ.reference_values),
            prior_limits=dict(KZ.prior_limits), restrictions=UDM.Get_model_restrictions(models),
            observation=observations, nwalkers=int(st.get("nwalkers", 32)), nsteps=int(st.get("nsteps", 1000)),
            burn=int(st.get("burn", 0) or 0), model_name=models, logger=logging.getLogger("peek"),
            sample_H0_uncalibrated_SNe=bool(st.get("sample_H0_uncalibrated_SNe", False)))

    PLOT_SETTINGS = U.build_plot_settings(observations, os.path.join("Peek", suffix), bool(latex_enabled), False)
    normalize_save_roots(PLOT_SETTINGS)
    PLOT_SETTINGS["corner_show_all_cmb_params"] = bool(all_params)
    PLOT_SETTINGS["corner_show_nuisance"] = bool(all_params) or bool(st.get("corner_show_nuisance", False))
    rules = U.convergence_rules(PLOT_SETTINGS, st.get("convergence"))

    out_paths: List[str] = []
    for model in models:
        c = CONFIG[model]
        Samples, shown, rows = {}, [], []
        tmpdir = tempfile.mkdtemp(prefix=COPY_PREFIX)
        try:
            for j, g in enumerate(c["observations"]):
                key = U.generate_label(g, config_model=c, obs_index=j)
                cands = []
                for eng, pat in (("zeus", f"{key}_zeus.h5"), ("emcee", f"{key}.h5")):
                    cands += [(f, eng) for f in glob.glob(os.path.join(root, model, "*", pat))]
                if K.CMB_NUISANCE != "baseline":
                    cands = [x for x in cands if os.path.dirname(x[0]).endswith(f"_plik_{K.CMB_NUISANCE}")] or cands
                if not cands:
                    print(f"  {model} {key}: no chain yet")
                    continue
                src, eng = max(cands, key=lambda x: os.path.getmtime(x[0]))
                chain, copy = _copy_and_read(src, tmpdir, eng)
                it = chain.shape[0]
                b_run = int(st.get("burn", 0) or 0)
                b = burn if burn is not None else (b_run if (b_run and it > 2 * b_run) else it // 2)
                if it - b < 2:
                    print(f"  {model} {key}: only {it} steps so far; nothing left after a burn-in of {b}")
                    continue
                flat = chain[b:].reshape(-1, chain.shape[-1])
                Samples[key] = flat
                shown.append((key, it, b, flat, c["parameters"][j]))
                try:
                    rows.append(U.chain_convergence_summary(tmpdir, key, b, list(c["parameters"][j]), rules))
                except Exception as e:
                    log.debug("peek: convergence summary failed for %s: %s", key, e)
            if not Samples:
                continue
            # CT-28: groups without a chain yet (the run has not reached them) are left out of the
            # corner plot; make_CornerPlot needs samples for every group of the CONFIG it gets
            keep = [j for j, g in enumerate(c["observations"])
                    if U.generate_label(g, config_model=c, obs_index=j) in Samples]
            if len(keep) < len(c["observations"]):
                c = _subset_groups(c, keep)
            if any(str(t) == "CMB" for ts in c["observation_types"] for t in ts):
                with U.quiet_cstdio():
                    CR.ensure_class_ready(model)    # r_d column of the table, as in the run
            step = max(it for _, it, _, _, _ in shown)
            name = f"{model}_step{step}" + ("_all" if all_params else "")
            P.make_CornerPlot(Samples, c, model, name, PLOT_SETTINGS)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        found = glob.glob(os.path.join("Plots", "**", "Peek", suffix, model, "corner_plots", f"corner_{name}.png"),
                          recursive=True)
        print()
        for key, it, b, flat, names in shown:
            print(f"{model} {key}: {it} steps so far, burn-in {b}, {flat.shape[0]} samples")
            for k, p in enumerate(names):
                print(f"    {p:14s} {flat[:, k].mean():.6g} +/- {flat[:, k].std():.3g}")
        if rows:
            print()
            print(U.format_convergence_table(model, rows, rules))
        if found:
            print(f"\nCorner plot: {os.path.abspath(found[0])}")
            out_paths.append(os.path.abspath(found[0]))
    return out_paths
