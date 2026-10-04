#!/usr/bin/env python3
"""Validate the integrated statistics against the four existing paper chains.

Run from the repository root. Output is written to a new temporary directory
unless --output is supplied. The default chain root is this repository's
MCMC_Chains directory; use --chain-root to point to an external archive.
No chains, CLASS sources or run configuration
files are edited, and CLASS rebuilding is prohibited in this process.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, default=Path.cwd())
parser.add_argument("--chain-root", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
root = args.repo.resolve()
sys.path.insert(0, str(root))
os.chdir(root)

import emcee
import numpy as np
import Kosmulator as kc
import User_defined_modules as udm
from Kosmulator_main import Class_run as cr
from Kosmulator_main import Model_comparison as mc
from Kosmulator_main import Post_processing as pp

for module in (kc, udm, cr, mc, pp):
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError(f"Unexpected module: {module.__file__}")

kc.observations = [["PantheonPS", "DESI_DR2"]]
kc.nwalkers, kc.nsteps, kc.burn = 120, 8500, 1000
kc.pantheonp_mode = "PplusSH0ES"
kc.prior_limits = dict(kc.prior_limits)
kc.prior_limits.update({"Omega_m": (.1, .5), "H_0": (60., 90.),
                        "M_abs": (-20.5, -18.), "w": (-2., -.33), "delta": (-1., 1.)})
original_ensure = cr.ensure_class_ready
def no_rebuild(model, **kwargs):
    kwargs.update(force=False, no_rebuild=True)
    return original_ensure(model, **kwargs)
cr.ensure_class_ready = no_rebuild

helper_path = root / "reproducibility/SAIP2026_IDE/scripts/paper_context.py"
spec = importlib.util.spec_from_file_location("recovered_paper_helper", helper_path)
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)
if not Path(paper.__file__).resolve().is_relative_to(root):
    raise RuntimeError("Unexpected paper context module")

chain_root = (args.chain_root or root/"MCMC_Chains").resolve()
obs_folder = "DESI_DR2_PantheonP_SH0ES"
cases = [
    ("LCDM", "lcdm", "RD_CLASS_DIAG_5K", "LCDM_v",
     "DESI_DR2+PantheonP_SH0ES.h5", 1472.715196940941, 1478.7237553855),
    ("iwCDM", "iw", "RD_CLASS_SEEDED_2K", "NonLinear_IDE_2",
     "DESI_DR2+PantheonP_SH0ES_FINAL_8500.h5", 1461.416626033485, 1471.4023444719),
    ("+iwCDM", "plus_iw", "RD_CLASS_PLUS_IW_8K", "NonLinear_IDE_2",
     "DESI_DR2+PantheonP_SH0ES.h5", 1470.850483701247, 1480.8251998564),
    ("SiwCDM", "siw", "RD_CLASS_SIW_8K", "NonLinear_IDE_2",
     "DESI_DR2+PantheonP_SH0ES.h5", 1465.744453474332, 1475.6356745239),
]
results, failures = {}, []
for label, regime, folder, model, filename, target_chi2, target_dic in cases:
    print("\n"+"="*78, flush=True)
    print("VALIDATING", label, flush=True)
    path = chain_root/folder/model/obs_folder/filename
    if not path.is_file():
        raise FileNotFoundError(path)
    paper.set_regime(regime)
    if not cr.ensure_class_ready(model, announce=True):
        raise RuntimeError("Existing CLASS binary unavailable")
    config, data, index, obs, types, names, priors, func = paper.build_context(model)
    cfg = config[model]
    cfg["_postprocessing_switches"] = {key: bool(getattr(udm, key)) for key in mc.SWITCHES}
    backend = emcee.backends.HDFBackend(str(path), read_only=True)
    samples = backend.get_chain(discard=1000, flat=True)
    key = "+".join(obs)
    cfg["_postprocessing_sources"] = {key: str(path)}
    print("Chain:", path, "retained samples:", len(samples), flush=True)
    # Exercise the same public statistics entry point used by generate_plots.
    group = pp.statistical_analysis({}, data, config, model,
                                    posterior_samples={model: {key: samples}})[model][key]
    results[label] = {key: group}
    for name in ("N", "k", "Raw_chain_chi2", "Chi_squared", "D_bar",
                 "D_at_mean", "p_D", "DIC", "AIC", "AICc", "BIC"):
        print(f"{name:24s} = {group[name]}")
    print("Likelihood source:", group["Likelihood_source"])
    for run in group["Optimizer_runs"]:
        print(f"  {run['stage']}: success={run['success']}, nfev={run['nfev']}, "
              f"chi2={run['chi2']:.12f}, objective spread="
              f"{run['simplex_objective_spread']:.3e}")
    dchi, ddic = group["Chi_squared"]-target_chi2, group["DIC"]-target_dic
    print(f"Difference from paper chi2: {dchi:+.12e}")
    print(f"Difference from paper DIC:  {ddic:+.12e}")
    if group["N"] != 1670 or group["k"] != (3 if regime == "lcdm" else 5):
        failures.append(f"{label}: N or k mismatch")
    if abs(dchi) > 5e-4 or abs(ddic) > 1e-4:
        failures.append(f"{label}: statistics outside validation tolerances")
    if not group["Likelihood_source"].startswith("saved likelihood blobs"):
        failures.append(f"{label}: saved likelihood blobs were not used")
    if any(not run["success"] for run in group["Optimizer_runs"]):
        failures.append(f"{label}: optimiser diagnostics require review")

mc.compare_models(results, "LCDM")
output = args.output or Path(tempfile.mkdtemp(prefix="Kosmulator_postprocessing_validation_"))
mc.export_comparison(results, output)
print("\nOUTPUT:", output.resolve())
print("\nCORRECTED MODEL COMPARISON (N=1670)")
for label, groups in results.items():
    row = next(iter(groups.values()))
    print(label, " ".join(f"{name}={row[name]:+.6f}"
                           for name in ("dAIC", "dAICc", "dDIC", "dBIC")))
if failures:
    print("\nREQUIRES REVIEW:\n"+"\n".join(failures))
    raise SystemExit(1)
print("\nAll four chains passed the statistics and optimiser checks.")
print("No MCMC, chain edits, source edits, or CLASS rebuilding performed.")
