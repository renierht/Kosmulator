"""Matched short DES Model 9 proposal benchmark; read-only production input."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import sys, json, time, argparse, multiprocessing as mp
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--output", default="ide_moves_benchmark.json")
    args = parser.parse_args()
    if args.steps < 1 or args.rounds < 1:
        parser.error("steps and rounds must be positive")
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
    schedule = [(r, move) for r in range(args.rounds)
                for move in (("stretch", "de_snooker") if r % 2 == 0
                             else ("de_snooker", "stretch"))]
    assert CFG["parameters"][0] == ["Omega_m", "w", "delta", "H_0"], CFG["parameters"]
    for round_index, move in schedule:
        np.random.seed(173 + round_index)
        print(f"Starting round {round_index + 1}: {move}, {args.steps} steps", flush=True)
        with mp.get_context("fork").Pool(8) as pool:
            start = time.perf_counter()
            sampler = emcee.EnsembleSampler(120, 4, evaluate, pool=pool, moves=emcee_moves(move))
            sampler.run_mcmc(initial.copy(), args.steps, progress=True)
            elapsed = time.perf_counter()-start
        x = sampler.get_chain()
        tau = emcee.autocorr.integrated_time(x, tol=0, quiet=True)
        report.append(dict(move=move, round=round_index + 1, steps=args.steps, seconds=elapsed,
            acceptance=float(sampler.acceptance_fraction.mean()), tau_provisional=tau.tolist(),
            warning="Autocorrelation estimates require sufficient chain length and consistency checks",
            jump_squared=np.mean(np.diff(x,axis=0)**2,axis=(0,1)).tolist()))
        output = ROOT/"validation"/args.output
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        os.replace(temporary, output)
        np.savez_compressed(output.with_name(output.stem + f"_{round_index}_{move}.npz"),
                            chain=x, log_prob=sampler.get_log_prob())
        print(report[-1], flush=True)
