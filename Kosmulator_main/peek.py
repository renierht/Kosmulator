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


def _settings_from_folders(suffix: str) -> Dict[str, Any]:
    """Models, groups and Planck set of a run without run_settings.json, from its folders."""
    root = _chain_root(suffix)
    models, groups, lite = [], {}, False
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
                lite = lite or os.path.basename(gdir).endswith("_plik_lite")
    obs = []
    for m in models:
        for g in groups[m]:
            if g not in obs:
                obs.append(g)
    return {"models": models, "observations": obs, "cmb_nuisance": "lite" if lite else "baseline"}


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
    rules = U.convergence_rules(PLOT_SETTINGS, st.get("convergence"))

    out_paths: List[str] = []
    for model in models:
        c = CONFIG[model]
        Samples, shown, rows = {}, [], []
        tmpdir = tempfile.mkdtemp(prefix="kosm_peek_")
        try:
            for j, g in enumerate(c["observations"]):
                key = U.generate_label(g, config_model=c, obs_index=j)
                cands = []
                for eng, pat in (("zeus", f"{key}_zeus.h5"), ("emcee", f"{key}.h5")):
                    cands += [(f, eng) for f in glob.glob(os.path.join(root, model, "*", pat))]
                if K.CMB_NUISANCE == "lite":
                    cands = [x for x in cands if os.path.dirname(x[0]).endswith("_plik_lite")] or cands
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
