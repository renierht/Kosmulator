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
model_names: List[str] = ["Linear_IDE_1"]

# Each inner list is a combined likelihood
observations: List[List[str]] = [
    #['JLA'],
    #['JLA','CC'],
    #['OHD'],
    #['CC'],
    ['PantheonPS', 'DESI_DR2', 'CC'],
    #['PantheonPS'],
    #['PantheonPS','CC'],
    #['f_sigma_8'],
    #['f'],
    #['DESI_DR1'],
    #['DESI_DR2'],
    #['BAO'],
    #['Union3'],
    #["DESY5"],
    #['Union3','CC'],
    #["DESY5",'CC'],
    #["JLA","CC","OHD"],
    #['PantheonPS','CC','DESI_DR2'],
    #['PantheonP','DESI_DR2','BBN_DH_AlterBBN'],
    #['PantheonPS','DESI_DR2','BBN_DH_AlterBBN'],
    #["JLA","Pantheon","PantheonP","DESY5","Union3"],
    #["CMB_lowl", "CMB_hil_TT"],
    #["DESI_DR2", 'CC','PantheonP','f_sigma_8'],
    #["DESI_DR2",'BBN_PryMordial'],
    #["DESI_DR2","BBN_DH_AlterBBN"],
    #["DESI_DR2","CMB_lowl"],
    #["DESI_DR2","CMB_lowl","BBN_PryMordial"],
    #["CMB_lowl"],
    #['CMB_hil'],
    #['CMB_lensing', 'CMB_lowl'],
    #['CMB_hil_TT'],
    #["CC", "DESI_DR1"],
    #["JLA","DESY5","Union3"],
    #["DESI_DR1", "f_sigma_8", "CC"],
    #["f", "f_sigma_8"],
    #["f", "PantheonP"],
    #["f_sigma_8", "PantheonP"],
]

true_model: str = "Linear_IDE_1"

# Sampler settings
nwalkers: int = 36
nsteps: int = 100000
burn: int = 30000
convergence: float = 0.01


prior_limits: Dict[str, Tuple[float, float]] = {
    # Core Cosmological Parameters (Table III limits)
    "Omega_dm": (0.001, 0.9),      # Fractional density of dark matter
    "Omega_b": (0.001, 0.3),       # Fractional density of baryons
    "Omega_m": (0.1, 0.5),       # Implied total matter bounds if sampled directly
    "H_0": (60.0, 90.0),          # Hubble constant bounds
    "M_abs": (-20.5, -18.0),       # Supernova absolute magnitude (M)
    "w": (-2.0, -0.33),           # Dark energy equation of state

    # Interacting Dark Energy Coupling Parameters
    # Approximating the (-∞, +∞) unbounded priors for linear models
    "delta_dm": (-2.0, 2.0),
    "delta_de": (-2.0, 2.0),
    "delta": (-10.0, 10.0),

    # Standard Nuisance / CMB Parameters (Unchanged for safety)
    "r_d": (0.01, 1000.0),
    "r_d": (147.499, 147.501),
    "gamma": (0.01, 1.0),
    "sigma_8": (0.01, 1.0),
    "n": (0.0, 0.6),
    "q0": (-0.8, -0.01),
    "q1": (-0.75, 1.0),
    "beta": (0.01, 5.0),
    "tau_reio": (0.04, 0.09),
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

# ----------------------------------------------------------------------
# Reference “true” values (Based on paper's best fits)
# ----------------------------------------------------------------------
true_values: Dict[str, float] = {
    # Aligned with the mean values from Table V (Pantheon+, DESI DR2, CC & BBN)
    "Omega_dm": 0.26,
    "Omega_b": 0.047,
    "Omega_m": 0.307,
    "H_0": 70.0,
    "M_abs": -19.35,
    "w": -1.0,
    "delta_dm": 0.0,
    "delta_de": 0.0,
    "delta": 0.0,

    # Standard fallbacks
    "gamma": K.GAMMA_FS8_SINGLETON,
    "sigma_8": 0.8,
    "q0": -0.537,
    "n": 0.25,
    "q1": 0.125,
    "beta": 2.505,
    "r_d": K.R_D_SINGLETON,
    "tau_reio": 0.054,
    "Omega_dh^2": 0.12,
    "Omega_bh^2": 0.0224,
    "n_s": 0.9624,
    "ln10^10_As": 3.045,
    "Omega_de": 0.69,
    "100theta_s": 1.04110,
    "N_eff": K.N_EFF_DEFAULT,
    "tau_n": K.TAU_N_DEFAULT,
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

    # --- IDE SAFETY CHECK: Completely reject CMB for IDE models ---
    # Check if any selected model is an IDE model
    is_ide_run = any("IDE" in str(m) for m in model_names)

    # Check if any selected observation group contains CMB data
    flat_obs = [obs for grp in observations for obs in grp]
    has_cmb = any("CMB" in str(obs) for obs in flat_obs)

    if is_ide_run and has_cmb:
        raise RuntimeError(
            "\n" + "="*85 + "\n"
            "   CONFIGURATION ERROR: IDE Models + CMB Not Supported!\n"
            "="*85 + "\n"
            "You have requested a CMB likelihood while running an Interacting Dark Energy (IDE) model.\n"
            "CMB perturbation theory requires a modified CLASS backend, which is currently disabled.\n\n"
            "FIX: Restrict your observations strictly to background cosmology (e.g., CC, SNe, BAO)\n"
            "when running 'IDE_de_v' or 'IDE_dm_v' models.\n"
            + "="*85 + "\n"
        )
    # --------------------------------------------------------------

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
