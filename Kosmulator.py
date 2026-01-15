#!/usr/bin/env python3
"""
Kosmulator entry point.

Configure:
  - models to run
  - observation combinations
  - priors and reference values
  - basic sampler settings

Then call the main MCMC driver in Kosmulator_main.MCMC_setup.

Project layout (fixed):
  Kosmulator/
  ├─ Kosmulator.py                                       # (this file)
  ├─ User_defined_modules.py                # user models
  ├─ Kosmulator_main/
  │  ├─ Class_run.py
  │  ├─ Config.py
  │  ├─ constants.py
  │  ├─ Kosmulator_MCMC.py                   # main MCMC orchestrator 
  │  ├─ MCMC_setup.py                               # setup/adapter (fallback)
  │  ├─ Statistical_packages.py
  │  └─ utils.py                                                # parse_cli_args() lives here
  └─ Plots/
     ├─ Plots_functions.py
     └─ Plots.py
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Import path setup
# ----------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parent
KOSM_MAIN = PROJ_ROOT / "Kosmulator_main"
if str(KOSM_MAIN) not in sys.path:
    sys.path.insert(0, str(KOSM_MAIN))

# ----------------------------------------------------------------------
# Import main MCMC driver (with fallback)
# ----------------------------------------------------------------------
try:
    from Kosmulator_main.MCMC_setup import main as run_mcmc
except Exception:
    import importlib.util as ilu

    spec = ilu.spec_from_file_location("MCMC_setup", str(KOSM_MAIN / "MCMC_setup.py"))
    if spec is None or spec.loader is None:
        raise
    _mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    run_mcmc = _mod.main  # type: ignore[assignment]
    
from Kosmulator_main import constants as K

from Kosmulator_main.utils import install_rank0_logging  # type: ignore[import]
install_rank0_logging()
logging.basicConfig(level=logging.INFO)


# ----------------------------------------------------------------------
# User configuration
# ----------------------------------------------------------------------

# Models implemented in User_defined_modules.py
model_names: List[str] = ["LCDM_v"]

# Each inner list is a combined likelihood
observations: List[List[str]] = [
    #['JLA'],
    #['OHD'],
    ['CC'],
    #['PantheonP'],
    #['PantheonPS'],
    #['f_sigma_8'],
    #['f'],
    #['DESI_DR1'],
    #['DESI_DR2'],
    #['BAO'],
    #['Union3'],
    #['DESY5'],
    #["JLA","CC","OHD"],
    #["JLA","Pantheon","PantheonP","DESY5","Union3"],
    #["CMB_lowl", "CMB_hil_TT"],
    #["DESI_DR2", 'CC','PantheonP','f_sigma_8'],
    #["DESI_DR2",'BBN_PryMordial'],
    #["DESI_DR2","BBN_DH_AlterBBN"],
    #["DESI_DR2","CMB_lowl"],
    #["DESI_DR2","CMB_lowl","BBN_PryMordial"],
    #["CMB_lowl"],
    #['CMB_hil'],
    #['CMB_lensing'],
    #['CMB_hil_TT'],
    #["CC", "DESI_DR1"],
    #["JLA","DESY5","Union3"],
    #["DESI_DR1", "f_sigma_8", "CC"],
    #["f", "f_sigma_8"],
    #["f", "PantheonP"],
    #["f_sigma_8", "PantheonP"],
]

true_model: str = "LCDM_v"

# Sampler settings
nwalkers: int = 16
nsteps: int = 500
burn: int = 10
convergence: float = 0.01

# Top-hat priors
prior_limits: Dict[str, Tuple[float, float]] = {
    "Omega_m": (0.01, 1.0),
    "Omega_b": (0.001, 0.1),
    "H_0": (40.0, 100.0),
    "r_d": (0.01, 1000.0),
    "M_abs": (-30.0, -5.0),
    "gamma": (0.01, 1.0),
    "sigma_8": (0.01, 1.0),
    "n": (0.0, 0.6),
    "q0": (-0.8, -0.01),
    "q1": (-0.75, 1.0),
    "beta": (0.01, 5.0),
    "tau_reio": (0.03, 0.06),
    "Omega_dh^2": (0.05, 0.2),
    "Omega_bh^2": (0.015, 0.031),
    "ln10^10_As": (2.5, 3.5),
    "n_s": (0.9, 1.1),
    "100theta_s": (1.035, 1.047),
    "N_eff": (K.N_EFF_DEFAULT, K.N_EFF_DEFAULT),
    "tau_n": (K.TAU_N_DEFAULT, K.TAU_N_DEFAULT),
    "alpha": (0.00, 1.00),
    "B": (0.00, 0.333),
    "f1": (0.01, 100.0),
}

# Reference “true” values (for diagnostics/plots)
true_values: Dict[str, float] = {
    "Omega_m": 0.315,
    "H_0": 67.4,
    "gamma": K.GAMMA_FS8_SINGLETON,
    "sigma_8": 0.8,
    "q0": -0.537,
    "n": 0.25,
    "q1": 0.125,
    "beta": 2.505,
    "r_d": K.R_D_SINGLETON,
    "M_abs": -19.2,
    "tau_reio": 0.054,
    "Omega_dh^2": 0.12,
    "Omega_bh^2": 0.0224,
    "n_s": 0.9624,
    "ln10^10_As": 3.045,
    "Omega_de": 0.69,
    "100theta_s": 1.04110,
    "N_eff": K.N_EFF_DEFAULT,
    "tau_n": K.TAU_N_DEFAULT,
    "Omega_b": 0.05,
}

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _ensure_true_model_first(names: List[str], tm: str) -> List[str]:
    """Return `names` with `tm` (if present) moved to the front."""
    ordered = list(names)
    if tm in ordered:
        ordered.remove(tm)
    return [tm] + ordered

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    from Kosmulator_main.utils import print_init_banner  # type: ignore[import]

    try:
        from mpi4py import MPI  # type: ignore[import]
        rank = MPI.COMM_WORLD.Get_rank()
    except Exception:
        rank = 0

    ordered_models = _ensure_true_model_first(model_names, true_model)

    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
        print_init_banner("Initialising Kosmulator and setting up Safeguards.")
        logging.info("Models (ordered): %s", ordered_models)
        logging.info("Observations: %s", observations)
        logging.info(
            "Sampler: nwalkers=%d nsteps=%d burn=%d conv=%.5f",
            nwalkers,
            nsteps,
            burn,
            convergence,
        )

    run_mcmc(
        model_names=ordered_models,
        observations=observations,
        true_model=true_model,
        prior_limits=prior_limits,
        true_values=true_values,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burn=burn,
        convergence=convergence,
    )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
