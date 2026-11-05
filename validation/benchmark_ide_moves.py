"""Matched short DES Model 9 proposal benchmark; read-only production input."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import sys, json, time, multiprocessing as mp
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import emcee, h5py
from Kosmulator_main import Config, constants as K, Kosmulator_MCMC as M, Class_run as CR
from Kosmulator_main.sampler_moves import emcee_moves
import User_defined_modules as U

def evaluate(theta):
    return M.emcee_prob(theta, DATA, CFG["observation_types"][0], CFG,
                        U.Get_model_function(NAME), NAME, CFG["observations"][0], 0)

if __name__ == "__main__":
    os.chdir(ROOT)
    NAME = "SignSwitch_IDE_1"
    paper = json.loads((ROOT/"reproducibility/SAIP2026_IDE/configs/paper.json").read_text())
    for key, value in paper["class_settings"].items():
        setattr(K, key, value)
    U.ALLOW_NEGATIVE_ENERGIES = U.ALLOW_BIG_RIP = U.ALLOW_DOOM_FACTOR_INSTABILITIES = True
    configs, DATA = Config.create_config(U.Get_model_names([NAME]), model_name=[NAME],
        observation=[["DES_Dovekie", "DESI_DR2"]], prior_limits=dict(paper["prior_limits"], r_d=[100,200]),
        free_w=True, free_w_dm=False, nwalkers=120)
    CFG = configs[NAME]
    CR.ensure_class_ready(NAME, no_rebuild=True, announce=False)
    source = ROOT.parent/"Kosmulator_followup/MCMC_Chains/DES_Model9_iw_dovekie/SignSwitch_IDE_1/DESI_DR2_DES_Dovekie/DESI_DR2+DES_Dovekie.h5"
    with h5py.File(source, "r") as handle:
        group = handle["mcmc"]
        initial = group["chain"][group.attrs["iteration"]-1]
    report = []
    for move in ("stretch", "de_snooker"):
        np.random.seed(173)
        with mp.get_context("fork").Pool(8) as pool:
            start = time.perf_counter()
            sampler = emcee.EnsembleSampler(120, 4, evaluate, pool=pool, moves=emcee_moves(move))
            sampler.run_mcmc(initial.copy(), 100, progress=False)
            elapsed = time.perf_counter()-start
        x = sampler.get_chain()
        tau = emcee.autocorr.integrated_time(x, tol=0, quiet=True)
        report.append(dict(move=move, steps=100, seconds=elapsed,
            acceptance=float(sampler.acceptance_fraction.mean()), tau_provisional=tau.tolist(),
            warning="100 steps cannot establish reliable ESS or convergence",
            jump_squared=np.mean(np.diff(x,axis=0)**2,axis=(0,1)).tolist()))
        (ROOT/"validation/ide_moves_benchmark.json").write_text(json.dumps(report,indent=2))
        print(report[-1], flush=True)
