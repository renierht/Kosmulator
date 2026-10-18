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
model_names: List[str] = ["wowaCDM_v"]

# Each inner list is a combined likelihood
observations: List[List[str]] = [
    #['DESI_DR2', 'CMB_lowl']
    #['DESI_DR2', 'CMB_lowl', 'CMB_hil'],
    ['DESI_DR2'],
    #['DESI_DR2','DESY5'],
    #['BBN_PryMordial','DESI_DR2', 'PantheonP'],
    #['DESI_DR2', 'Union3'],
    #['DESI_DR2', 'PantheonP'],
    #['JLA'],
    #['JLA','CC'],
    #['OHD'],
    #['CC'],
    #['PantheonP'],
    #['DESI_DR2','PantheonPS'],
    #['BBN_PryMordial', 'CC', 'DESI_DR2', 'Pantheon'],
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
    # Ground-based CMB (candl; ACT DR6, SPT-3G D1) and the combinations used in the literature:
    #['CMB_SPA'],            # CMB-SPA: Planck (cut) + ACT DR6 + SPT-3G D1 + Planck low-l TT and SRoll2 EE
    #['CMB_PACT'],           # P-ACT: the same without SPT
    #['CMB_ACT'],            # ACT DR6 alone (tau prior 0.0566 +- 0.0058 unless low-l EE is in the group)
    #['CMB_SPT'],            # SPT-3G D1 alone (tau prior 0.051 +- 0.006 unless low-l EE is in the group)
    #['CMB_SPA', 'DESI_DR2'],
    #['CMB_PACT', 'CMB_lensing_ACTPlanck', 'DESI_DR2'],   # with ACT DR6 + Planck PR4 lensing (P-ACT-LB's)
    #['CMB_SPA', 'CMB_lensing_ACTPlanck', 'CMB_lensing_SPT'],   # with ACT + Planck and SPT-3G MUSE lensing
    #['CMB_lensing_ACTPlanck', 'DESI_DR2', 'BBN_DH'],    # lensing without primary CMB; BBN sets r_d
    #["CC", "DESI_DR1"],
    #["JLA","DESY5","Union3"],
    #["DESI_DR1", "f_sigma_8", "CC"],
    #["f", "f_sigma_8"],
    #["f", "PantheonP"],
    #["f_sigma_8", "PantheonP"],
]

reference_model: str = "LCDM_v"

# Sampler settings
nwalkers: int = 32
nsteps: int = 1500
burn: int = 100
# Convergence rule (zeus and emcee): the run stops once, after burn-in, the chain is
# >= 100 tau_max long, ESS >= 2000, split-Rhat < 1.01 and tau_max changed by less than
# `convergence` (relative) since the previous check (constants.CONV_*).
convergence: float = 0.05
# Uncalibrated supernovae without BAO/DESI or an H0 anchor (CC, OHD, SH0ES, CMB): once the
# SN offset is marginalised H_0 cancels from the likelihood. False (default, the usual
# choice): H_0 is not sampled and distances use reference_values["H_0"]. True: H_0 is
# sampled anyway; its posterior is then the prior and it is not counted in k.
sample_H0_uncalibrated_SNe: bool = False
# Planck high-l likelihood (CMB_hil, CMB_hil_TT). "baseline": plik, with Planck's nuisance
# parameters sampled (21 for TTTEEE, 15 for TT) under Planck's priors. "lite": plik_lite,
# Planck's foreground-marginalised likelihood with A_planck only (6 + 1 parameters for LCDM,
# much faster to converge). "extended": baseline plus the polarisation efficiencies and EE
# dust amplitudes Planck fixes, sampled with their stated priors (30 for TTTEEE). "full": all
# 47 inputs of plik TTTEEE (20 of TT); 17 of them have no published uncertainty and get flat
# boxes chosen by Kosmulator, so they are prior-dominated and convergence is slow. Chains of
# the other settings go to folders ending in _plik_lite, _plik_extended or _plik_full.
cmb_nuisance: str = "baseline"
# WAIC for groups with Planck data. The Planck likelihoods give one total log-likelihood,
# not one per data point, so WAIC can only count each of them as one term, and that costs
# one lensed CLASS run per WAIC draw (1000 draws per group, after sampling). False
# (default): WAIC is reported as n/a, with a note, for groups with Planck data. True: it is
# computed, if you are willing to wait.
waic_include_cmb: bool = False
# Neutrinos in every CLASS call (CMB spectra, r_d) and in the late-time expansion. "massive"
# (default): one massive neutrino of nu_mass_eV plus massless ones, nu_N_eff in total, as in
# Planck 2018 (0.06 eV, N_eff 3.046) and DESI DR2 (sum 0.06 eV, N_eff 3.044). "massless": all
# nu_N_eff massless (Kosmulator before October 2026). BBN keeps its own N_eff (prior_limits).
neutrinos: str = "massive"
nu_mass_eV: float = 0.06
nu_N_eff: float = 3.044
# Non-linear correction of the CMB lensing (lensed spectra and Planck lensing), as in Cobaya: one
# setting for every model, passed to CLASS. None (default): linear lensing. "halofit", or "hmcode"
# (hmcode_version "2020", CAMB's default, or "2015", the earlier HMcode that Planck 2018 used; use it
# to compare with Planck's tables). These fits are calibrated on simulations with GR; for models that
# change gravity on small scales they are an approximation, and Kosmulator warns for models not in
# constants.GR_GRAVITY_MODELS. HMcode costs about 70% more time per CMB point than None, halofit 8%.
cmb_nonlinear = None
hmcode_version: str = "2020"
# Growth data (f, f_sigma_8). "gamma" (default): f = Omega_m(z)^gamma with the growth index gamma
# sampled (fixed to 0.55 for f_sigma_8 alone), a phenomenological test of gravity. "geff": the model's
# own linear growth with G_eff/G from <model>_Geff(z, p) in User_defined_modules.py (gamma not
# sampled). A model without <model>_Geff is grown with G_eff = G, the GR (LCDM) growth equation on its
# own expansion history; for a theory that changes gravity that can differ from its true growth, and
# Kosmulator warns at start-up.
growth_model: str = "gamma"
# Proposal moves of the samplers. emcee_moves: "de_snooker" (default): differential evolution (80% of
# the steps) plus a snooker move (20%, ter Braak & Vrugt 2008), or "stretch" (emcee's own move). Both
# sample the same posterior; in Kosmulator's tests de_snooker needed 2 - 6 times fewer steps than
# stretch (about 4.5 times on a 27-parameter target shaped like a Planck baseline chain; Planck lite:
# 4900 against 11 600 steps, same posterior) at the same likelihood calls per step. zeus_light_mode: True
# (default) skips zeus's interval expansions after its tuning phase: fewer likelihood calls per step but
# more steps (Complete test T07: 1.1 - 1.9 times more steps, 1.3 - 2.3 times less time, same posteriors);
# False gives zeus's full slice step. Changing either between a run and its --resume is allowed.
emcee_moves: str = "de_snooker"
zeus_light_mode: bool = True
# r_d of BAO/DESI groups calibrated by BBN or Planck. "model" (default): CLASS with the model's own
# E(z) in place of the cosmological constant (rd_Class: compiled once, then cached), for every model
# other than LCDM; this matters when the model's dark sector is not negligible before recombination
# (e.g. w0wa with w0 + wa near 0). "lcdm": CLASS's flat-LCDM r_d for every model (as before).
rd_background: str = "model"
# Groups whose CMB data are lensing only (CMB_lensing, CMB_lensing_ACTPlanck, CMB_lensing_SPT without
# primary CMB). True (default): n_s gets a Gaussian prior N(0.96, 0.02) and tau_reio, which enters none
# of these likelihoods, is fixed at its reference value, as in lensing-only analyses (Madhavacheril
# et al. 2024, arXiv:2304.05203). Add a BBN tag (e.g. BBN_DH) for Omega_b h^2; it also calibrates r_d.
# False: n_s and tau_reio sampled within their top-hat priors (as before October 2026).
lensing_only_priors: bool = True
# Top-hat priors
prior_limits: Dict[str, Tuple[float, float]] = {
    "Omega_m": (0.01, 1.0),
    "Omega_b": (0.01, 0.06),
    "H_0": (40.0, 100.0),
    "r_d": (0.01, 1000.0),
    "M_abs": (-30.0, -5.0),
    "gamma": (0.01, 1.0),
    "sigma_8": (0.01, 1.0),
    "n": (-0.6, 0.6),   # f1CDM; n < 0 is phantom-like (w < -1), n = 0 is LCDM; n < 0.5 by restrict_f1CDM_v
    "q0": (-0.8, -0.01),
    "q1": (-0.75, 1.0),
    "beta": (0.01, 5.0),
    "tau_reio": (0.01, 0.8),      # Planck 2018's range; (0.04, 0.09) cut Planck's tau at about 2 sigma
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
    'w0': (-3.0, 1.0),
    'wa': (-3.0, 2.0),
    'w': (-2.0, -1.0),
    'delta':(0.0, 0.5),
}

# Reference “true” values (for diagnostics/plots)
reference_values: Dict[str, float] = {
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
    "w0": -1.0,
    "wa": 0.0,
}

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _ensure_reference_model_first(names: List[str], tm: str) -> List[str]:
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

    ordered_models = _ensure_reference_model_first(model_names, reference_model)

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
        reference_model=reference_model,
        prior_limits=prior_limits,
        reference_values=reference_values,
        nwalkers=nwalkers,
        nsteps=nsteps,
        burn=burn,
        convergence=convergence,
        sample_H0_uncalibrated_SNe=sample_H0_uncalibrated_SNe,
        cmb_nuisance=cmb_nuisance,
        waic_include_cmb=waic_include_cmb,
        neutrinos=neutrinos,
        nu_mass_eV=nu_mass_eV,
        nu_N_eff=nu_N_eff,
        cmb_nonlinear=cmb_nonlinear,
        hmcode_version=hmcode_version,
        growth_model=growth_model,
        emcee_moves=emcee_moves,
        zeus_light_mode=zeus_light_mode,
        rd_background=rd_background,
        lensing_only_priors=lensing_only_priors,
    )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
