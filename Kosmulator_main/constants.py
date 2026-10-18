# Kosmulator_main/constants.py
from __future__ import annotations

import os

from typing import Any, Dict, List, Set, Tuple

"""
Central place for global constants used throughout Kosmulator.

Sections:
  1. Physical / cosmological constants
  2. Data / path defaults
  3. Plotting & LaTeX helpers
  4. Corner-table layout heuristics
  5. Engine toggles (runtime switches)
"""

# ======================================================================
# 1. Physical / cosmological constants
# ======================================================================

#: Speed of light [km s^-1]
C_KM_S: float = 299_792.458

#: CMB monopole temperature [K] (Planck 2018)
T_CMB_DEFAULT: float = 2.7255

#: N_eff of a BBN grid file without an N_eff axis (the grids before October 2026). BBN itself uses the
#: run's NU_N_EFF when N_eff is not sampled (CT-39), as CLASS does.
N_EFF_DEFAULT: float = 3.046

#: Radiation in the late-time E(z) of the built-in models (LCDM, w0waCDM,
#: f1CDM, NonLinear_IDE_2): Omega_r h^2 = omega_gamma (1 + 0.22711 N_ur), with
#: photons at T_CMB_DEFAULT and N_ur = N_UR_LATE massless neutrino species (set by
#: the neutrinos below: 2.0308 with one massive neutrino, which is then in Omega_m).
#: Omega_DE is reduced by Omega_r so the model stays flat. A sampled or fixed
#: 'Omega_r' in the parameters overrides this.
LATE_TIME_RADIATION: bool = True

#: Neutrinos in every CLASS call (CMB spectra, r_d) and in the late-time E(z) (items 13, 27).
#: "massive": as Planck 2018 base LCDM (arXiv:1807.06209, Sect. 2.1: one massive eigenstate of
#: 0.06 eV and two massless ones), one massive eigenstate of NU_MASS_EV = 0.06 eV (in Omega_m at
#: late times) plus massless species, N_eff = NU_N_EFF = 3.044 in total (CLASS v3.3's default;
#: Planck 2018 used 3.046). "massless": all NU_N_EFF massless
#: (Kosmulator before item 27). BBN uses the same N_eff (CT-39; Observations/BBN/bbn_grid.npz has
#: nodes at 3.044 and 3.046, so both load without AlterBBN).
#: Set per run in Kosmulator.py (neutrinos, nu_mass_eV, nu_N_eff) through set_neutrinos(); the
#: values below are the defaults, or those of the run that started this process (KOSM_* variables).
NEUTRINO_MODES: Tuple[str, ...] = ("massive", "massless")
NEUTRINOS: str = os.environ.get("KOSM_NEUTRINOS", "massive")
NU_MASS_EV: float = float(os.environ.get("KOSM_NU_MASS_EV", "0.06"))
NU_N_EFF: float = float(os.environ.get("KOSM_NU_N_EFF", "3.044"))
#: N_eff of one massive eigenstate while relativistic, for CLASS's T_ncdm = 0.71611
#: ((0.71611 / (4/11)^(1/3))^4); with it N_ur = 3.044 - 1.0132 = 2.0308.
NU_NEFF_PER_MASSIVE: float = 1.0132016
#: omega_nu = m_nu / NU_EV_PER_OMEGA (CLASS's 93.14 eV for T_ncdm = 0.71611)
NU_EV_PER_OMEGA: float = 93.14

N_UR_LATE: float = (NU_N_EFF - NU_NEFF_PER_MASSIVE) if NEUTRINOS == "massive" else NU_N_EFF


def set_neutrinos(mode: str = "massive", mass_ev: float = 0.06, n_eff: float = 3.044) -> None:
    """
    Neutrinos of this run (Kosmulator.py: neutrinos, nu_mass_eV, nu_N_eff). Sets them in
    this process and in KOSM_NEUTRINOS, KOSM_NU_MASS_EV and KOSM_NU_N_EFF, which pool workers
    started afterwards inherit (they import this module afresh; MPI ranks call this
    themselves). Call it before any pool is built.
    """
    global NEUTRINOS, NU_MASS_EV, NU_N_EFF, N_UR_LATE
    m = str(mode).strip().lower()
    if m not in NEUTRINO_MODES:
        raise ValueError(f"neutrinos must be one of {NEUTRINO_MODES}, not {mode!r}")
    mass, neff = float(mass_ev), float(n_eff)
    if m == "massive" and not mass > 0.0:
        raise ValueError(f"nu_mass_eV must be > 0 with neutrinos = 'massive' (got {mass_ev!r})")
    if neff <= (NU_NEFF_PER_MASSIVE if m == "massive" else 0.0):
        raise ValueError(f"nu_N_eff = {n_eff!r} is too small for neutrinos = {m!r}")
    NEUTRINOS, NU_MASS_EV, NU_N_EFF = m, mass, neff
    N_UR_LATE = (NU_N_EFF - NU_NEFF_PER_MASSIVE) if NEUTRINOS == "massive" else NU_N_EFF
    os.environ["KOSM_NEUTRINOS"] = NEUTRINOS
    os.environ["KOSM_NU_MASS_EV"] = repr(NU_MASS_EV)
    os.environ["KOSM_NU_N_EFF"] = repr(NU_N_EFF)


def neutrino_signature() -> Tuple[str, float, float]:
    """The neutrino settings, for cache keys of CLASS results."""
    return (NEUTRINOS, NU_MASS_EV if NEUTRINOS == "massive" else 0.0, NU_N_EFF)


def omega_nu() -> float:
    """Physical density of the massive neutrino (0 with NEUTRINOS = "massless")."""
    return NU_MASS_EV / NU_EV_PER_OMEGA if NEUTRINOS == "massive" else 0.0


def class_neutrino_params() -> Dict[str, float]:
    """CLASS input for the neutrinos set above (the same in every CLASS call)."""
    if NEUTRINOS == "massive":
        return {"N_ur": NU_N_EFF - NU_NEFF_PER_MASSIVE, "N_ncdm": 1, "m_ncdm": NU_MASS_EV}
    return {"N_ur": NU_N_EFF}


#: Non-linear matter power for the CMB lensing potential (items 27, 13), as in Cobaya: one setting
#: for every model, passed to CLASS (Kosmulator.py cmb_nonlinear, hmcode_version; set_cmb_nonlinear).
#: None (default, Renier 11 Oct): linear lensing. "halofit", or "hmcode": HMcode, HMCODE_VERSION
#: "2020" (Mead et al. 2021, arXiv:2009.01858; CAMB's default) or "2015" (CLASS's earlier HMcode,
#: Mead et al. 2015, 2016, which Planck 2018 used in CAMB: arXiv:1807.06209, Sect. 2.1). These fits
#: are calibrated on simulations with GR: for models that change
#: gravity on small scales (not in GR_GRAVITY_MODELS) they are an approximation, and a warning says
#: so. Cost per lensed CLASS run in the cloud (2 cores, one 0.06 eV neutrino): linear 3.6 s, halofit
#: 3.9 s, HMcode 6.1 s; plik_lite -2lnL at a Planck-like point: linear 614.5, halofit 611.9, HMcode 611.9.
CMB_NONLINEAR_MODES: Tuple[str, ...] = ("hmcode", "halofit", "none")
HMCODE_VERSIONS: Tuple[str, ...] = ("2020", "2015")
_nl_env = os.environ.get("KOSM_CMB_NONLINEAR", "none").strip().lower()
CMB_NONLINEAR = None if _nl_env in ("", "none") else _nl_env
HMCODE_VERSION: str = os.environ.get("KOSM_HMCODE_VERSION", "2020")


def set_cmb_nonlinear(mode=None, version="2020") -> None:
    """
    Non-linear correction of this run (Kosmulator.py: cmb_nonlinear, hmcode_version), for every
    model. Sets KOSM_CMB_NONLINEAR and KOSM_HMCODE_VERSION as well, for pool workers started
    afterwards (see set_neutrinos). Call it before any pool is built.
    """
    global CMB_NONLINEAR, HMCODE_VERSION
    m = "none" if mode is None else str(mode).strip().lower()
    if m not in CMB_NONLINEAR_MODES:
        raise ValueError(f"cmb_nonlinear must be one of 'hmcode', 'halofit' or None, not {mode!r}")
    v = str(version).strip()
    if m == "hmcode" and v not in HMCODE_VERSIONS:
        raise ValueError(f"hmcode_version must be one of {HMCODE_VERSIONS}, not {version!r}")
    CMB_NONLINEAR = None if m == "none" else m
    HMCODE_VERSION = v
    os.environ["KOSM_CMB_NONLINEAR"] = m
    os.environ["KOSM_HMCODE_VERSION"] = v


def class_nonlinear_params(model_name: str = "") -> Dict[str, str]:
    """CLASS input for the non-linear correction of the lensing potential (the same for every
    model; model_name is kept for the callers' signature)."""
    if not CMB_NONLINEAR:
        return {}
    if CMB_NONLINEAR == "hmcode":
        return {"non_linear": "hmcode", "hmcode_version": str(HMCODE_VERSION)}
    return {"non_linear": str(CMB_NONLINEAR)}


def nonlinear_description() -> str:
    """Short text for logs and notes."""
    if not CMB_NONLINEAR:
        return "linear (no non-linear correction)"
    if CMB_NONLINEAR == "hmcode":
        return f"HMcode {HMCODE_VERSION}"
    return str(CMB_NONLINEAR)


#: Models whose gravity is GR (LCDM, dark-energy fluids): the growth equation needs no G_eff,
#: and the non-linear lensing fits (calibrated on GR simulations) apply. Add your own GR models.
GR_GRAVITY_MODELS: Tuple[str, ...] = ("LCDM", "LCDM_v", "LCDM_nv", "wowaCDM_v")

#: Linear growth for f and f_sigma_8 (Kosmulator.py growth_model; set_growth_model).
#: "gamma" (default, Kosmulator's approach so far): f = Omega_m(z)^gamma with the growth index
#: gamma sampled (fixed to GAMMA_FS8_SINGLETON for f_sigma_8 alone), a phenomenological test of
#: gravity. "geff": the model's own prediction from the linear growth equation
#:     df/dlna + f^2 + (2 + dlnE/dlna) f = (3/2) Omega_m(a) G_eff/G
#: (Nesseris et al. 2013, arXiv:1308.6142, Eq. 23), with G_eff/G from <model>_Geff(z, p) in
#: User_defined_modules.py; gamma is not sampled. A model without <model>_Geff gets G_eff = G:
#: GR growth on its own expansion history (a warning says so unless it is in GR_GRAVITY_MODELS).
GROWTH_MODES: Tuple[str, ...] = ("gamma", "geff")
GROWTH_MODEL: str = os.environ.get("KOSM_GROWTH_MODEL", "gamma")
#: geff: the growth equation starts at GROWTH_Z_START (matter era, f = Omega_m(z)^0.55) and is
#: integrated with GROWTH_STEPS fourth-order Runge-Kutta steps in ln a
GROWTH_Z_START: float = 100.0
GROWTH_STEPS: int = 300


def set_growth_model(mode="gamma") -> None:
    """Growth model of this run (Kosmulator.py growth_model); also KOSM_GROWTH_MODEL for pool workers."""
    global GROWTH_MODEL
    m = str(mode).strip().lower()
    if m not in GROWTH_MODES:
        raise ValueError(f"growth_model must be one of {GROWTH_MODES}, not {mode!r}")
    GROWTH_MODEL = m
    os.environ["KOSM_GROWTH_MODEL"] = m


# Proposal moves (Kosmulator.py: emcee_moves, zeus_light_mode; Kosmulator_main/sampler_moves.py).
# emcee: "de_snooker" (default since 13 Oct, Renier: differential evolution 0.8 + snooker 0.2,
# ter Braak & Vrugt 2008; t11 matched t9's stretch posterior in 4900 instead of 11 600 steps) or
# "stretch" (emcee's own move). zeus: its DifferentialMove; light mode (default since 18 Oct, CT-21:
# T07 needed 1.1 - 1.9 times more steps but 1.3 - 2.3 times less time, same posteriors) skips the
# expansions after tuning.
# Only the process that runs the sampler uses these (rank 0 under MPI, which also calls
# MCMC_setup.main), so pool workers need no KOSM_* variable.
EMCEE_MOVES: str = "de_snooker"
EMCEE_MOVES_OPTIONS: Tuple[str, ...] = ("stretch", "de_snooker")
DE_SNOOKER_WEIGHTS: Tuple[float, float] = (0.8, 0.2)
ZEUS_LIGHT_MODE: bool = True


def set_sampler_moves(emcee_moves="de_snooker", zeus_light_mode=True) -> None:
    """Moves of this run (Kosmulator.py emcee_moves, zeus_light_mode), validated."""
    global EMCEE_MOVES, ZEUS_LIGHT_MODE
    m = str(emcee_moves).strip().lower()
    if m not in EMCEE_MOVES_OPTIONS:
        raise ValueError(f"emcee_moves must be one of {EMCEE_MOVES_OPTIONS}, not {emcee_moves!r}")
    if not isinstance(zeus_light_mode, bool):
        raise ValueError(f"zeus_light_mode must be True or False, not {zeus_light_mode!r}")
    EMCEE_MOVES, ZEUS_LIGHT_MODE = m, bool(zeus_light_mode)


# r_d of groups with an early-time calibrator (BBN or Planck) for BAO/DESI (Kosmulator.py
# rd_background; Kosmulator_main/rd_class.py). "model" (default): rd_Class, CLASS with the
# model's own E(z) in place of the cosmological constant, for every model not in
# RD_LCDM_MODELS. "lcdm": CLASS's flat-LCDM r_d for every model (Kosmulator before October
# 2026). KOSM_RD_BACKGROUND carries the choice to spawned pool workers.
RD_BACKGROUND: str = os.environ.get("KOSM_RD_BACKGROUND", "model")
RD_BACKGROUND_MODES: Tuple[str, ...] = ("model", "lcdm")
RD_LCDM_MODELS: Tuple[str, ...] = ("LCDM", "LCDM_v", "LCDM_nv")


def set_rd_background(mode="model") -> None:
    """r_d background of this run (Kosmulator.py rd_background); also KOSM_RD_BACKGROUND."""
    global RD_BACKGROUND
    m = str(mode).strip().lower()
    if m not in RD_BACKGROUND_MODES:
        raise ValueError(f"rd_background must be one of {RD_BACKGROUND_MODES}, not {mode!r}")
    RD_BACKGROUND = m
    os.environ["KOSM_RD_BACKGROUND"] = m


def class_signature() -> Tuple:
    """Settings that change CLASS results (neutrinos, non-linear correction, l_max), for cache keys."""
    return neutrino_signature() + (CMB_NONLINEAR, HMCODE_VERSION if CMB_NONLINEAR == "hmcode" else "",
                                   CMB_CLASS_LMAX)

#: h used for Omega_r when a group does not sample H_0 (e.g. f, f_sigma_8 alone)
H0_RADIATION_FALLBACK: float = 67.4

#: Neutron lifetime [s] used consistently in BBN (AlterBBN + grid)
TAU_N_DEFAULT: float = 879.4

#: Legacy fixed sound horizon [Mpc] used in "singleton" BAO/DESI modes
R_D_SINGLETON: float = 147.5

#: GR-like growth index used when fσ8 is a singleton (and as a default)
GAMMA_FS8_SINGLETON: float = 0.55

#: f_sigma_8.dat (Skara & Perivolaropoulos 2020, arXiv:1911.10609, Table VI).
#: WiggleZ rows (z, f sigma_8) and their covariance (Blake et al. 2012, as given in
#: Kazantzidis & Perivolaropoulos 2018, arXiv:1803.01337, Eq. 3.3); the diagonal
#: equals the squared errors in the file. Used when FS8_WIGGLEZ_COV is True.
FS8_WIGGLEZ_COV: bool = False
FS8_WIGGLEZ_ROWS: Tuple[Tuple[float, float], ...] = ((0.44, 0.413), (0.60, 0.390), (0.73, 0.437))
FS8_WIGGLEZ_COV_MATRIX: Tuple[Tuple[float, ...], ...] = (
    (6.400e-3, 2.570e-3, 0.0),
    (2.570e-3, 3.969e-3, 2.540e-3),
    (0.0, 2.540e-3, 5.184e-3),
)
#: Alcock-Paczynski-type correction of each f sigma_8 point to the sampled cosmology:
#: the prediction is divided by q = H(z) d_A(z) / [H_fid(z) d_A_fid(z)], the fiducial
#: being flat LCDM with the Omega_m of the file's 4th column (Kazantzidis &
#: Perivolaropoulos 2018, Eqs. 1.7 and 3.1; Skara & Perivolaropoulos 2020, Eq. 3.1).
FS8_AP_CORRECTION: bool = False

#: Uncalibrated supernovae (Pantheon+ without SH0ES, DES-Y5, Union3, JLA,
#: Pantheon): marginalise the magnitude offset (M_abs, or the H0 normalisation
#: of the distance moduli) analytically, as in the Cobaya SN likelihoods
#: (use_abs_mag: False). M_abs is then not sampled for PantheonP, and a BAO/DESI
#: group whose other data fix neither H0 nor r_d uses the fixed r_d below, so
#: H_0 measures h*r_d as DESI's hrd does. Set False for the old behaviour
#: (M_abs sampled for PantheonP; H0 acting as the offset of distance moduli).
SN_MARGINALISE_OFFSET: bool = True

#: Datasets that share measurements (item 35): in one group they count the same data twice,
#: which makes the constraints too tight. Kosmulator warns at start-up and notes it in the
#: statistics; it does not stop the run.
DATA_OVERLAPS: Tuple[Tuple[frozenset, str], ...] = (
    (frozenset({"OHD", "CC"}), "OHD.dat repeats 31 of the cosmic-chronometer H(z) points of CC.dat "
                               "(its other 20 rows are BAO/clustering H(z)); use CC or OHD, not both"),
    (frozenset({"BAO", "DESI_DR1"}), "BAO is the 10 anisotropic DESI DR1 points, which DESI_DR1 also contains"),
    (frozenset({"f", "f_sigma_8"}), "f.dat and f_sigma_8.dat have measurements from the same galaxy surveys "
                                    "(SDSS, VIPERS, 2dFGRS, GAMA, WiggleZ, VVDS; see the f.dat header); use one"),
    (frozenset({"DESI_DR1", "DESI_DR2"}), "DESI DR2 includes the DR1 data"),
    (frozenset({"CMB_hil", "CMB_hil_TT"}), "the Planck TT,TE,EE likelihood contains the TT spectra of the TT one"),
    # Ground-based CMB and the Planck + ACT (+ SPT) combinations (item 57)
    (frozenset({"CMB_lowl", "CMB_lowl_sroll2"}), "SimAll and SRoll2 are two analyses of the same Planck low-l EE data"),
    (frozenset({"CMB_ACT", "CMB_hil"}), "ACT DR6 and full Planck overlap in sky and multipoles: ACT recommends "
                                        "Planck cut at l = 1000 (TT) and 600 (TE, EE), i.e. CMB_planck_ACTcut or CMB_PACT"),
    (frozenset({"CMB_ACT", "CMB_hil_TT"}), "ACT DR6 and full Planck TT overlap in sky and multipoles: use CMB_planck_ACTcut "
                                           "or CMB_PACT"),
    (frozenset({"CMB_planck_ACTcut", "CMB_hil"}), "both are Planck high-l TT,TE,EE (the cut one is a subset)"),
    (frozenset({"CMB_planck_ACTcut", "CMB_hil_TT"}), "both contain Planck high-l TT"),
    # CMB lensing (item 61)
    (frozenset({"CMB_lensing", "CMB_lensing_ACTPlanck"}), "Planck 2018 (PR3) lensing and the Planck PR4 part of "
                                                          "CMB_lensing_ACTPlanck are two reconstructions from the same "
                                                          "Planck data; use one"),
)
#: The Planck + ACT (+ SPT) combinations contain these tags; with any of them in the same
#: group the data are counted twice (utils.data_overlaps)
CMB_BUNDLE_CONFLICTS: Tuple[str, ...] = ("CMB_hil", "CMB_hil_TT", "CMB_lowl", "CMB_lowl_TT", "CMB_lowl_sroll2",
                                         "CMB_planck_ACTcut", "CMB_ACT", "CMB_SPT", "CMB_PACT", "CMB_SPA")
#: Supernova compilations: any two of them share supernovae
SN_COMPILATIONS: Tuple[str, ...] = ("JLA", "JLA_legacy", "Pantheon", "PantheonP", "PantheonPS", "DESY5", "Union3")
#: BBN data: the D/H systems (two predictions of the same data) and the DESI DR2 BBN prior on
#: Omega_b h^2, which is derived from primordial deuterium (and helium) measurements
BBN_DATASETS: Tuple[str, ...] = ("BBN_DH", "BBN_DH_AlterBBN", "BBN_PryMordial")

#: Official JLA likelihood (Betoule et al. 2014, arXiv:1401.4064), tag "JLA".
#: Light-curve nuisance parameters sampled with JLA: (low, high, start). Priors
#: and starting values follow Cobaya's sn.jla. The absolute magnitudes M_B and
#: M_B + Delta_M (host-mass step) are fitted analytically and not sampled.
JLA_NUISANCE_DEFAULTS: Dict[str, Tuple[float, float, float]] = {
    "alpha_JLA": (0.01, 2.0, 0.14),
    "beta_JLA": (0.9, 4.6, 3.1),
}
#: Host-galaxy split for the magnitude step, in log10(M_stellar / M_sun)
JLA_HOST_MASS_SPLIT: float = 10.0
#: Late-time datasets whose likelihood is expensive (JLA rebuilds and factorises
#: a 740 x 740 covariance for every alpha, beta: about 10 - 20 ms). For groups
#: containing them zeus spreads the walkers over the worker pool instead of
#: evaluating them in one vectorised call on a single core.
POOL_PREFERRED_DATASETS: Set[str] = {"JLA"}

# ----------------------------------------------------------------------
# BBN deuterium (BBN_DH likelihood)
# ----------------------------------------------------------------------
#: Standard-BBN D/H(omega_b) used by the "approx" backend: a power law fitted to
#: the PRyMordial table PRyM_Yp_DH_cosmoMC_2023.dat (Burns et al. 2023,
#: arXiv:2307.07061; NACRE II rates with the LUNA d(p,gamma)3He rate, N_eff =
#: 3.044, Delta N = 0) over omega_b = 0.018 - 0.027, where it matches the table
#: to 0.7% (the table's own Monte Carlo noise). PArthENoPE 3.0 (Pisanti et al.
#: 2021, arXiv:2011.11537) gives the same value to 0.2%; PRIMAT (arXiv:2011.11320)
#: is about 3% lower, which is the present nuclear-rate systematic.
BBN_DH_REF: float = 2.508e-5          # D/H at omega_b = BBN_DH_OMEGA_B_REF
BBN_DH_OMEGA_B_REF: float = 0.0224
BBN_DH_SLOPE: float = -1.640          # d ln(D/H) / d ln(omega_b)
#: Fractional theory (nuclear-rate and neutron-lifetime) uncertainty of the D/H
#: prediction, from the PRyMordial Monte Carlo errors in the same table (3.8 -
#: 4.4% over omega_b = 0.018 - 0.027). It is one common error for all quasar
#: systems (fully correlated) and is added to every BBN_DH backend.
BBN_DH_THEORY_FRAC: float = 0.041

#: Primordial D/H measurements, PDG Review of Particle Physics 2025, "Big-Bang
#: nucleosynthesis" (Fields, Molaro, Sarkar), Table 24.1 (12 quasar absorption
#: systems; values 1e6 D/H) and eq. (24.2): weighted mean 25.08 +/- 0.29 with
#: scale factor S = 1.08. The 2023 revision used before had 11 systems
#: (25.47 +/- 0.29, S = 1.137); PKS 1937-1009 at z = 3.572 changed from
#: 26.24 +/- 0.48 to 26.08 +/- 1.02 and QSO J1332+0052 was added.
BBN_DH_PDG_SYSTEMS = (
    {"name": "SDSS J1419+0829", "DH": 25.06, "sig_up": 0.52, "sig_dn": 0.52},
    {"name": "HS 0105+1619",    "DH": 25.76, "sig_up": 1.54, "sig_dn": 1.54},
    {"name": "QSO B0913+0715",  "DH": 25.29, "sig_up": 1.05, "sig_dn": 1.05},
    {"name": "SDSS J1358+0349", "DH": 26.18, "sig_up": 0.72, "sig_dn": 0.72},
    {"name": "SDSS J1358+6522", "DH": 25.82, "sig_up": 0.71, "sig_dn": 0.71},
    {"name": "SDSS J1558-0031", "DH": 24.04, "sig_up": 1.44, "sig_dn": 1.44},
    {"name": "PKS 1937-1009 A", "DH": 24.49, "sig_up": 2.80, "sig_dn": 2.80},
    {"name": "QSO J1444+2919",  "DH": 19.68, "sig_up": 3.3,  "sig_dn": 2.8},
    {"name": "PKS 1937-1009 B", "DH": 26.08, "sig_up": 1.02, "sig_dn": 1.02},
    {"name": "QSO 1009+2956",   "DH": 24.77, "sig_up": 4.1,  "sig_dn": 3.5},
    {"name": "QSO 1243+307",    "DH": 23.88, "sig_up": 0.82, "sig_dn": 0.82},
    {"name": "QSO J1332+0052",  "DH": 23.88, "sig_up": 0.77, "sig_dn": 0.77},
)
BBN_DH_PDG_MEAN = {"DH": 25.08, "sigma": 0.29}
BBN_DH_PDG_S: float = 1.08

#: BLAS/OpenMP threads in the main process while a group samples without a
#: worker pool (vectorised zeus and emcee): at most this many, and never more than
#: the run's own main-process threads (MAIN_THREADS: 1 for a serial run, so
#: --num_cores 1 stays serial; --num_cores with a pool). None: the run's threads.
#: Renier's 12-core machine (bench_blas_threads.py, 12 Oct, DESI DR2 + DES-Y5 and
#: + Pantheon+): 2 - 4 threads 10 - 20% faster than 1, 8 and 12 slower; 4 was best
#: or within 6% of best in every case (item 24).
MAIN_BLAS_THREADS_NO_POOL = 4

#: The main process's threads for CLASS and BLAS, set by utils.set_main_threads (item 45):
#: 1 for a serial run, --num_cores with a local pool, the ranks on rank 0's machine
#: under MPI, or the user's OMP_NUM_THREADS. None until a run sets it.
MAIN_THREADS = None

#: MPI ranks on rank 0's machine, counted at start-up (MCMC_setup.main); rank 0 runs its
#: own CLASS work (pre-fit, polish, bands) on this many threads while the others wait
MPI_NODE_RANKS: int = 1

#: CLASS threads per pool worker or MPI worker rank (--class_threads, CT-40); 1 by default.
#: With t > 1 a local pool has num_cores // t workers, each running CLASS on t threads (CLASS
#: v3 reads OMP_NUM_THREADS at every call; perturbations, transfer, lensing and harmonic are
#: threaded, background, thermodynamics and HMcode are not); BLAS stays at one thread. It pays
#: only when the cores outnumber nwalkers/2, since emcee and zeus evaluate half the walkers at
#: a time. The spectra do not depend on it (LCDM_v, l_max 8501, HMcode: bitwise identical with
#: 1 and 2 threads; one call 24.5 s on 1 thread, 13.0 s on 2, cloud, 18 Oct 2026).
CLASS_THREADS_PER_WORKER: int = 1

#: Defaults of the run's switches (Kosmulator.py, constants.py data options and the command
#: line), for the "Settings" summary at start-up and the results' header (CT-26); a switch
#: that differs is marked "*".
SETTINGS_DEFAULTS: Dict[str, Any] = {
    "neutrinos": "massive", "nu_mass_eV": 0.06, "nu_N_eff": 3.044, "rd_background": "model",
    "growth_model": "gamma", "cmb_nonlinear": None, "hmcode_version": "2020", "cmb_nuisance": "baseline",
    "lensing_only_priors": True, "sample_H0_uncalibrated_SNe": False, "waic_include_cmb": False,
    "CC_SYS_COV": None, "PANTHEON_SYS_COV": None, "FS8_WIGGLEZ_COV": False, "FS8_AP_CORRECTION": False,
    "engine_mode": "mixed", "emcee_moves": "de_snooker", "zeus_light_mode": True,
    "tau_consecutive": 1, "autocorr_buffer": 0, "class_threads": 1, "corner_show_nuisance": False,
}

#: The run's settings (MCMC_setup.main), for the summary and the results' header (CT-26)
RUN_SETTINGS: Dict[str, Any] = {}

#: Largest --class_threads the start-up suggestion proposes (CT-44). T35c on Renier's 12 CPUs:
#: 2, 3 and 4 threads per worker all gave 58 - 60 s per step against 75 s with 1 (the CPUs were
#: saturated from 2 on), so 2: the fewest threads per process for the gain.
CLASS_THREADS_SUGGEST_MAX: int = 2

#: emcee keeps its chain in memory and appends it to the chain file every this many steps and at
#: the end (CT-19; emcee's HDFBackend opened and wrote the file at every step). A killed run loses
#: at most this many steps; Ctrl-C none.
EMCEE_WRITE_EVERY: int = 100

#: {model: {group label: "zeus" or "emcee"}}, set at start-up (MCMC_setup step 10, CT-31); the
#: model banners use it (CT-17)
GROUP_ENGINES: dict = {}

#: A finite log-posterior at or below this value is a point the likelihood could not
#: compute: the CMB likelihoods return -1e10 when CLASS or clik fails. Counted during
#: sampling and reported per group (item 47).
LOGLIKE_FAILED_BELOW: float = -1e9

# ----------------------------------------------------------------------
# Cosmic chronometers: SPS-model systematic covariance (CC)
# ----------------------------------------------------------------------
#: Moresco et al. 2020 (arXiv:2003.07362), Table 3: percentage bias of H(z)
#: from the IMF, the stellar library and the SPS model, for the D4000 method;
#: columns z, IMF, stellar library, SPS, SPS without the most discordant model
#: ("odd one out"). The quoted CC errors already include the SFH and
#: metallicity terms (their Sect. 3.2), so only these three are added.
CC_MORESCO2020_TABLE3: Tuple[Tuple[float, float, float, float, float], ...] = (
    (0.075, 0.47, 7.40, 15.86, 9.91), (0.125, 0.47, 7.40, 14.23, 6.98),
    (0.175, 0.47, 7.40, 13.34, 5.40), (0.225, 0.47, 7.40, 13.21, 5.40),
    (0.275, 0.47, 7.40, 13.29, 5.40), (0.325, 0.47, 7.40, 12.20, 5.40),
    (0.375, 0.47, 7.40, 12.99, 5.40), (0.425, 0.47, 7.40, 10.29, 6.20),
    (0.475, 0.46, 7.39, 8.91, 5.86), (0.525, 0.23, 7.40, 9.99, 6.51),
    (0.575, 0.28, 6.87, 10.09, 6.12), (0.625, 0.47, 6.65, 11.17, 6.21),
    (0.675, 0.47, 6.57, 11.12, 5.71), (0.725, 0.47, 5.90, 10.81, 5.16),
    (0.775, 0.45, 6.03, 10.75, 5.05), (0.825, 0.47, 6.10, 10.75, 5.05),
    (0.875, 0.47, 5.89, 9.08, 2.79), (0.925, 0.44, 5.80, 8.62, 3.70),
    (0.975, 0.40, 5.94, 7.32, 3.65), (1.025, 0.27, 6.07, 5.84, 3.37),
    (1.075, 0.20, 6.08, 6.02, 3.49), (1.125, 0.20, 6.07, 4.72, 2.33),
    (1.175, 0.19, 6.09, 4.31, 2.33), (1.225, 0.19, 6.09, 3.90, 2.33),
    (1.275, 0.19, 6.09, 3.90, 2.33), (1.325, 0.20, 6.09, 3.91, 2.34),
    (1.375, 0.19, 6.09, 3.90, 2.34), (1.425, 0.19, 6.09, 3.90, 2.33),
    (1.475, 0.20, 6.09, 3.91, 2.34),
)
#: Redshifts of the CC.dat rows measured with the D4000 method (Moresco et al.
#: 2012, Moresco 2015, Moresco et al. 2016), the only rows the table applies to.
CC_D4000_Z: Tuple[float, ...] = (0.179, 0.199, 0.352, 0.3802, 0.4004, 0.4247, 0.4497,
                                 0.4783, 0.593, 0.68, 0.781, 0.875, 1.037, 1.363, 1.965)
#: SPS-model covariance added to CC (Moresco 2020, Eq. 9, summed over IMF, stellar
#: library and SPS: C_ij += sum_X eta_X(z_i) H_i eta_X(z_j) H_j, H the measured
#: values, eta interpolated linearly in z and held at the last row above z = 1.475):
#: None (diagonal errors and CC_corr.txt only), "full" (SPS column) or "ooo"
#: (SPS without the most discordant model).
CC_SYS_COV = None

#: Pantheon (Scolnic et al. 2018, arXiv:1710.00845, CT-14): None (the errors of Pantheon.dat only, the
#: paper's statistical-only analysis: flat LCDM Omega_m = 0.284 +- 0.012, Table 8; Kosmulator T02 0.285
#: +- 0.013) or "full" (adds the release's systematic covariance, Observations/Pantheon_release/
#: sys_full_long.txt, reordered to Pantheon.dat's rows: the published analysis, Omega_m = 0.298 +- 0.022,
#: Sect. 6.1).
PANTHEON_SYS_COV = None

# ----------------------------------------------------------------------
# Convergence rule (zeus and emcee, utils.ConvergenceMonitor)
# ----------------------------------------------------------------------
#: Checked every --autocorr-check-every steps on the chain after burn-in. tau_max
#: is the largest integrated autocorrelation time over the sampled parameters,
#: estimated with zeus's default method (utils.autocorr_time_mk).
#: Chain after burn-in must be at least this many tau_max long. The tau estimate itself is
#: reliable beyond ~50 tau (emcee and zeus documentation); emcee's own stopping example uses
#: 100 tau. 100 since item 59: in a seed study of w0wa with CC + DESI_DR2 + Pantheon+ it
#: refused the one false stop seen (at 4 000 steps, tau underestimated before the chain had
#: reached the low-Omega_m ridge); over the 128 runs of the item 55 benchmark it costs about
#: 5% more steps on average (33 runs longer, at most 2.2 times).
CONV_TAU_FACTOR = 100.0
#: Effective sample size of the slowest parameter, N_post x walkers / tau_max
#: (2000 gives a Monte Carlo error of ~2% of sigma on a posterior mean).
CONV_ESS_MIN = 2000.0
#: Split-Rhat limit (each walker's chain cut in two halves; Gelman et al. 2013;
#: 1.01 is the zeus documentation's SplitRCallback tolerance).
CONV_RHAT_MAX = 1.01
#: Relative change of tau_max between two checks below which tau counts as
#: stable; Kosmulator.py's `convergence` sets it for a run.
CONV_TAU_RTOL = 0.05
#: Number of checks in a row at which all the conditions above must hold before a run
#: stops (--tau-consecutive overrides it; 2 or more for a stricter stop). 1 since 13 Oct
#: (Renier): 2 replaced the fixed minimum run length in 8bfcc8c, before the length rule
#: became 100 tau (item 59). Successive checks share ~97% of the samples, so a second
#: check mostly waits for split-Rhat to cross 1.01 again: t11 (Planck lite, DE + snooker)
#: met the rule at step 4200 and stopped at 4900, with the same posterior (means within
#: 0.05 sd of t9's stretch chain at both steps). Replayed on the 75 item 55 / 59 benchmark
#: chains where both apply: 6% fewer steps on average (at most a third), the same quantile
#: errors (median 0.028 against 0.025 of the 68% half-width, largest 0.09 for both).
CONV_CONSECUTIVE = 1


# ======================================================================
# 2. Data / path defaults
# ======================================================================

#: Base directory where all observational data live
OBSERVATIONS_BASE: str = "./Observations"

#: Folder (inside OBSERVATIONS_BASE) with the official JLA files
JLA_DIR_RELATIVE: str = "JLA"

#: Relative path (inside OBSERVATIONS_BASE) to the default BBN grid
BBN_GRID_RELATIVE: str = "BBN/bbn_grid.npz"

#: Base directory where plots are saved
DEFAULT_PLOTS_BASE: str = "./Plots/Saved_Plots"

#: Default ASCII CMB spectra files (used for quick-plot helpers)
DEFAULT_CMB_FILES: Dict[str, str] = {
    "TT": f"{OBSERVATIONS_BASE}/CMB_TT.dat",
    "TE": f"{OBSERVATIONS_BASE}/CMB_TE.dat",
    "EE": f"{OBSERVATIONS_BASE}/CMB_EE.dat",
    "PP": f"{OBSERVATIONS_BASE}/CMB_PP.dat",  # lensing
}

#: Max ell for CMB TT/TE/EE plotting
CMB_ELL_MAX_PLOT: int = 2500

#: Lensing multipole range for plotting PP
CMB_LENSING_LMIN: int = 8
CMB_LENSING_LMAX: int = 400


# ======================================================================
# 3. Plotting & LaTeX helpers
# ======================================================================

# --- LaTeX symbol helpers ------------------------------------------------

GREEK_SYMBOLS: Dict[str, str] = {
    "Omega": r"\Omega", "omega": r"\omega",
    "alpha": r"\alpha", "beta": r"\beta",
    "gamma": r"\gamma", "delta": r"\delta",
    "epsilon": r"\epsilon", "zeta": r"\zeta",
    "eta": r"\eta", "theta": r"\theta",
    "iota": r"\iota", "kappa": r"\kappa",
    "lambda": r"\lambda", "mu": r"\mu",
    "nu": r"\nu", "xi": r"\xi",
    "pi": r"\pi", "rho": r"\rho",
    "sigma": r"\sigma", "tau": r"\tau",
    "upsilon": r"\upsilon", "phi": r"\phi",
    "chi": r"\chi", "psi": r"\psi",
    "Lambda": r"\Lambda",
    "ell": r"\ell", "ℓ": r"\ell",
}

# --- Observation → pretty label (LaTeX, plain text) ---------------------

OBS_PRETTY_MAP: Dict[str, Tuple[str, str]] = {
    # SNe datasets
    "JLA":       ("JLA",               "JLA"),
    "JLA_legacy": ("JLA (legacy)",     "JLA (legacy)"),
    "DESY5":       ("DESY5",               "DESY5"),
    "Union3":       ("Union3",               "Union3"),
    "Pantheon":       (r"Pantheon",       "Pantheon"),
    "PantheonP":       (r"Pantheon$^{+}$",      "Pantheon+"),
    "PantheonP_SH0ES": (r"Pantheon$^{+}$+SH0ES","Pantheon+SH0ES"),

    # CMB datasets
    "CMB_ACT":         (r"ACT DR6 TTTEEE",              "ACT DR6 TTTEEE"),
    "CMB_SPT":         (r"SPT-3G D1 TTTEEE",            "SPT-3G D1 TTTEEE"),
    "CMB_planck_ACTcut": (r"Planck TTTEEE (ACT cut)",   "Planck TTTEEE (ACT cut)"),
    "CMB_lowl_sroll2": (r"Planck low-$\ell$ EE (SRoll2)", "Planck low-ℓ EE (SRoll2)"),
    "CMB_PACT":        (r"P-ACT",                       "P-ACT"),
    "CMB_SPA":         (r"CMB-SPA",                     "CMB-SPA"),
    "CMB_lowl":        (r"Planck low-$\ell$ EE",       "Planck low-ℓ EE"),
    "CMB_lowl_TT":     (r"Planck low-$\ell$ TT",       "Planck low-ℓ TT"),
    "CMB_hil":         (r"Planck high-$\ell$ TTTEEE",  "Planck high-ℓ TTTEEE"),
    "CMB_hil_TT":      (r"Planck high-$\ell$ TT",      "Planck high-ℓ TT"),
    "CMB_lensing":     (r"Planck lensing",             "Planck lensing"),
    "CMB_lensing_ACTPlanck": (r"ACT DR6 + Planck PR4 lensing", "ACT DR6 + Planck PR4 lensing"),
    "CMB_lensing_SPT": (r"SPT-3G MUSE lensing",         "SPT-3G MUSE lensing"),

    # Growth-rate data
    "f_sigma_8":       (r"$f_{\sigma_8}$",             "fσ₈"),
    "f":               (r"$f(z)$",                     "f(z)"),
    
    # Cosmic Chronometers
    "CC":        ("CC",                "CC"),
    "OHD":       ("OHD",               "OHD"),
    
    #BAO
    "BAO":       ("BAO",               "BAO"),
    "DESI_DR1":      (r"DESI DR1",           "DESI DR1"),
    "DESI_DR2":      (r"DESI DR2",           "DESI DR2"),
    
    #Big Bang Nucleosythesis
    "BBN_PryMordial":      (r"BBN (primordial)",             "BBN (primordial)"),
    "BBN_DH":              (r"BBN D/H",                      "BBN D/H"),
    "BBN_DH_AlterBBN":     (r"BBN D/H (AlterBBN)",           "BBN D/H (AlterBBN)"),
}

# --- Plot grouping / colours --------------------------------------------

#: Which observation types can share an axis column in best-fit panels
COMBINE_GROUPS: List[Set[str]] = [
    {"OHD", "CC"},
    {"PantheonP", "PantheonP_SH0ES", "Pantheon", "JLA", "JLA_legacy", "DESY5", "Union3"},
    {"BAO", "DESI_DR1", "DESI_DR2"},
]

#: Global colour palette for plotting
DEFAULT_PLOT_COLORS: List[str] = [
     "b", "green", "r", "cyan", "purple", "grey", "yellow", "m", "k", "olive",
    "orange", "pink", "crimson", "darkred", "salmon",
]

#: Default colours used by observation data in summary panels
OBS_COLOR_ORDER: List[str] = DEFAULT_PLOT_COLORS

#: Default colour used to draw model curves
MODEL_COLOR: str = "r"

# --- DESI/BAO code → (label, linestyle) ---------------------------------

CODE_STYLE: Dict[int, Tuple[str, object]] = {
    8: (r"$D_M/r_d$", "-"),
    6: (r"$D_H/r_d$", "--"),
    5: (r"$D_A/r_d$", "-."),
    3: (r"$D_V/r_d$", ":"),
    7: (r"$r_d/D_V$", (0, (1, 2))),
}


# ======================================================================
# 4. Corner-table layout heuristics
# ======================================================================

# These are hand-tuned anchors for placing the stats table relative to a
# corner plot. Kept here so Plot_functions / Plots can share them.

TABLE_ANCHORS_OBS = {
    "x":             [1,   2,    5,    8],
    "corner_top":    [0.95, 0.94, 0.90, 0.86],
    "per_row":       [1.05, 0.95, 0.55, 0.18],
    "cell_height_k": [9.0,  8.5,  5.5,  2.8],
}

TABLE_ANCHORS_PARM = {
    "x":             [2,   3,    4],  # when n_obs == 1
    "corner_top":    [0.90, 0.93, 0.94],
    "per_row":       [0.95, 0.95, 0.95],
    "cell_height_k": [8.5,  8.5,  8.5],
}


# ======================================================================
# 5. Engine overrides (runtime switches)
# ======================================================================

force_vectorisation: bool = False   # Force all models to run in vectorised mode (if supported)
disable_vectorisation: bool = False # Explicitly disable vectorisation (scalar evaluation only)

force_zeus: bool = False           # Force Kosmulator to prefer the Zeus engine
force_emcee: bool = False          # Force Kosmulator to use the emcee engine
engine_mode: str = "mixed"         # "mixed", "single", or "fastest"
engine_for_model: Dict[str, str] = {}   # populated in MCMC_setup for single-engine mode


def set_engine_overrides(
    force_vec: bool = False,
    disable_vec: bool = False,
    force_z: bool = False,
    force_e: bool = False,
    mode: str = "mixed",
) -> None:
    """
    Set global engine / execution policy toggles.

    Parameters
    ----------
    force_vec : bool
        Force vectorised model evaluation where supported.
    disable_vec : bool
        Explicitly disable vectorisation even if supported (scalar evaluation).
    force_z : bool
        Force the Zeus sampler wherever possible.
    force_e : bool
        Force the emcee sampler (ignores Zeus even if available).
    mode : {"mixed", "single", "fastest"}
        High-level engine strategy.
    """
    global force_vectorisation, disable_vectorisation
    global force_zeus, force_emcee, engine_mode, engine_for_model

    force_vectorisation   = bool(force_vec)
    disable_vectorisation = bool(disable_vec)

    # Mutually exclusive sanity check
    if force_vectorisation and disable_vectorisation:
        raise ValueError(
            "Cannot use --force_vectorisation and --disable_vectorisation together."
        )

    force_zeus  = bool(force_z)
    force_emcee = bool(force_e)
    engine_mode = mode or "mixed"

    # Per-model map is recomputed in MCMC_setup.main
    engine_for_model = {}



# ======================================================================
# 6a. CMB theory spectra
# One lensed CLASS run per point serves every CMB likelihood of a group (high-l, low-l,
# lensing). l_max_scalars = 3000: at 2509, lensed TT is 0.5% low at l = 2500 against
# l_max 4000; at 3000 it is 0.05% (about 2.05 s against 1.83 s per point). The last
# CMB_SPECTRA_CACHE_SIZE points are kept per process.
# ======================================================================

CMB_CLASS_LMAX: int = int(os.environ.get("KOSM_CMB_CLASS_LMAX", "3000"))
CMB_CLASS_LMAX_PLANCK: int = 3000   # enough for Planck (plik to l = 2508, lensing to 2048)


def set_cmb_class_lmax(lmax: int) -> None:
    """l_max of the run's lensed CLASS spectra (one run per point for every CMB likelihood of a
    group): 3000 for Planck, more when ACT (8501) or SPT (4095) is in the run (item 57);
    KOSM_CMB_CLASS_LMAX carries it to spawned pool workers."""
    global CMB_CLASS_LMAX
    CMB_CLASS_LMAX = int(lmax)
    os.environ["KOSM_CMB_CLASS_LMAX"] = str(CMB_CLASS_LMAX)
CMB_SPECTRA_CACHE_SIZE: int = 64

# ======================================================================
# 6. Planck CMB nuisance parameters (centralised here)
# These give default true values and prior ranges for CMB runs.
# ======================================================================

# Planck 2018 baseline treatment of the plik nuisance parameters (Planck 2018 V,
# arXiv:1907.12875; as in Cobaya's planck_2018_highl_plik likelihoods, checked against
# Cobaya 3.6.2): 21 sampled with the priors below, the other 26 fixed, plus the SZ prior.
# The lite likelihoods, Commander, SimAll and lensing have only A_planck.
#
# Sampled: name -> (reference value, (prior box low, high), Gaussian (mean, sigma) or None).
# A Gaussian prior is applied in log_prior_all; its box is mean +- 5 sigma.
def _g(mu: float, sigma: float, ref: float | None = None):
    return (mu if ref is None else ref, (mu - 5.0 * sigma, mu + 5.0 * sigma), (mu, sigma))

PLANCK_NUISANCE_SAMPLED: Dict[str, Tuple[float, Tuple[float, float], Tuple[float, float] | None]] = {
    # Overall calibration (every Planck likelihood) and temperature calibrations
    "A_planck":   _g(1.0, 0.0025),
    "calib_100T": _g(1.0002, 0.0007),
    "calib_217T": _g(0.99805, 0.00065),
    # Foregrounds with uniform priors (reference values: Cobaya's starting points)
    "A_cib_217":  (67.0, (0.0, 200.0), None),
    "xi_sz_cib":  (0.1,  (0.0, 1.0), None),
    "A_sz":       (7.0,  (0.0, 10.0), None),
    "ksz_norm":   (3.0,  (0.0, 10.0), None),
    "ps_A_100_100": (257.0, (0.0, 400.0), None),
    "ps_A_143_143": (47.0,  (0.0, 400.0), None),
    "ps_A_143_217": (40.0,  (0.0, 400.0), None),
    "ps_A_217_217": (104.0, (0.0, 400.0), None),
    # Galactic dust in TT (545 GHz template) and TE: Gaussian priors
    "gal545_A_100":     _g(8.6, 2.0),
    "gal545_A_143":     _g(10.6, 2.0),
    "gal545_A_143_217": _g(23.5, 8.5),
    "gal545_A_217":     _g(91.9, 20.0),
    "galf_TE_A_100":     _g(0.130, 0.042),
    "galf_TE_A_100_143": _g(0.130, 0.036),
    "galf_TE_A_100_217": _g(0.46, 0.09),
    "galf_TE_A_143":     _g(0.207, 0.072),
    "galf_TE_A_143_217": _g(0.69, 0.09),
    "galf_TE_A_217":     _g(1.938, 0.54),
}

# Fixed in the Planck baseline (values from Planck 2018 V and Cobaya's yaml files)
PLANCK_NUISANCE_FIXED: Dict[str, float] = {
    "cib_index": -1.3,
    "A_pol": 1.0, "calib_100P": 1.021, "calib_143P": 0.966, "calib_217P": 1.040,
    "galf_EE_index": -2.4, "galf_TE_index": -2.4,
    "galf_EE_A_100": 0.055, "galf_EE_A_100_143": 0.040, "galf_EE_A_100_217": 0.094,
    "galf_EE_A_143": 0.086, "galf_EE_A_143_217": 0.21, "galf_EE_A_217": 0.70,
    "A_cnoise_e2e_100_100_EE": 1.0, "A_cnoise_e2e_143_143_EE": 1.0, "A_cnoise_e2e_217_217_EE": 1.0,
    "A_sbpx_100_100_TT": 1.0, "A_sbpx_143_143_TT": 1.0, "A_sbpx_143_217_TT": 1.0, "A_sbpx_217_217_TT": 1.0,
    "A_sbpx_100_100_EE": 1.0, "A_sbpx_100_143_EE": 1.0, "A_sbpx_100_217_EE": 1.0,
    "A_sbpx_143_143_EE": 1.0, "A_sbpx_143_217_EE": 1.0, "A_sbpx_217_217_EE": 1.0,
}

# SZ prior on the sampled amplitudes: ksz_norm + PLANCK_SZ_PRIOR[0] * A_sz ~ N(mean, sigma)
PLANCK_SZ_PRIOR: Tuple[float, float, float] = (1.6, 9.5, 3.0)

# Larger sets of sampled plik nuisances (Kosmulator.py cmb_nuisance; item 58).
# "extended": the baseline 21 plus the 9 fixed parameters whose uncertainty Planck states
# (the comments of Cobaya 3.6.2 planck_2018_highl_plik/params_calib_pol.yaml and params_EE.yaml,
# "EE now recommended to be left to the central values of the stated priors"), sampled with
# those Gaussian priors; dust amplitudes kept >= 0.
def _g0(mu: float, sigma: float):
    ref, (lo, hi), g = _g(mu, sigma)
    return (ref, (max(0.0, lo), hi), g)

PLANCK_NUISANCE_EXTENDED: Dict[str, Tuple[float, Tuple[float, float], Tuple[float, float] | None]] = {
    "calib_100P": _g(1.021, 0.01), "calib_143P": _g(0.966, 0.01), "calib_217P": _g(1.040, 0.01),
    "galf_EE_A_100": _g0(0.055, 0.014), "galf_EE_A_100_143": _g0(0.040, 0.010),
    "galf_EE_A_100_217": _g0(0.094, 0.023), "galf_EE_A_143": _g0(0.086, 0.022),
    "galf_EE_A_143_217": _g0(0.21, 0.051), "galf_EE_A_217": _g0(0.70, 0.18),
}

# "full": every one of the 47 inputs of plik TT,TE,EE (20 of plik TT). The other 17 have no
# published uncertainty, Planck never sampled them, and some are nearly degenerate with others
# (A_pol with the three polarisation efficiencies): these flat boxes are Kosmulator's choice
# around Planck's fixed values (template amplitudes 0 - 2, spectral indices +- 1, the CIB index
# as Kosmulator's old box), not Planck's. Expect slow convergence and prior-dominated nuisances.
PLANCK_NUISANCE_FULL_EXTRA: Dict[str, Tuple[float, Tuple[float, float], Tuple[float, float] | None]] = {
    "A_pol": (1.0, (0.9, 1.1), None),
    "cib_index": (-1.3, (-2.0, -0.5), None),
    "galf_EE_index": (-2.4, (-3.4, -1.4), None),
    "galf_TE_index": (-2.4, (-3.4, -1.4), None),
    **{n: (1.0, (0.0, 2.0), None) for n in (
        "A_cnoise_e2e_100_100_EE", "A_cnoise_e2e_143_143_EE", "A_cnoise_e2e_217_217_EE",
        "A_sbpx_100_100_TT", "A_sbpx_143_143_TT", "A_sbpx_143_217_TT", "A_sbpx_217_217_TT",
        "A_sbpx_100_100_EE", "A_sbpx_100_143_EE", "A_sbpx_100_217_EE",
        "A_sbpx_143_143_EE", "A_sbpx_143_217_EE", "A_sbpx_217_217_EE")},
}
# The parameters of plik TT among them
PLANCK_TT_FULL_EXTRA: Set[str] = {"cib_index", "A_sbpx_100_100_TT", "A_sbpx_143_143_TT",
                                  "A_sbpx_143_217_TT", "A_sbpx_217_217_TT"}

# Every plik nuisance that can be sampled: name -> (reference value, prior box, Gaussian or None)
PLANCK_NUISANCE_ALL: Dict[str, Tuple[float, Tuple[float, float], Tuple[float, float] | None]] = {
    **PLANCK_NUISANCE_SAMPLED, **PLANCK_NUISANCE_EXTENDED, **PLANCK_NUISANCE_FULL_EXTRA}

# Gaussian priors applied in log_prior_all (name -> (mean, sigma)); only to parameters a group samples
PLANCK_GAUSSIAN_PRIORS: Dict[str, Tuple[float, float]] = {
    n: v[2] for n, v in PLANCK_NUISANCE_ALL.items() if v[2] is not None
}

# All plik nuisance names (sampled and fixed); they do not change the theory spectra
PLANCK_NUISANCE_NAMES: Set[str] = set(PLANCK_NUISANCE_SAMPLED) | set(PLANCK_NUISANCE_FIXED)

# Reference value and prior box of every nuisance that can be sampled (the format Config injects)
PLANCK_NUISANCE_DEFAULTS: Dict[str, Tuple[float, Tuple[float, float]]] = {
    n: (v[0], v[1]) for n, v in PLANCK_NUISANCE_ALL.items()
}

# Sampled nuisances of plik TT (15) and plik TTTEEE (21)
PLANCK_TT_ONLY_NUISANCE: Set[str] = {
    "A_planck", "calib_100T", "calib_217T",
    "A_cib_217", "xi_sz_cib", "A_sz", "ksz_norm",
    "ps_A_100_100", "ps_A_143_143", "ps_A_143_217", "ps_A_217_217",
    "gal545_A_100", "gal545_A_143", "gal545_A_143_217", "gal545_A_217",
}

PLANCK_TTTEEE_NUISANCE: Set[str] = set(PLANCK_NUISANCE_SAMPLED)


# High-l nuisance treatment, set from Kosmulator.py (cmb_nuisance):
#   "baseline": plik, with the nuisance parameters above (21 for TTTEEE, 15 for TT);
#   "lite":     plik_lite, Planck's foreground-marginalised likelihood, A_planck only;
#   "extended": plik, baseline + PLANCK_NUISANCE_EXTENDED (30 for TTTEEE, 15 for TT);
#   "full":     plik, all 47 inputs of TTTEEE (20 of TT) sampled (PLANCK_NUISANCE_FULL_EXTRA).
CMB_NUISANCE: str = "baseline"
CMB_NUISANCE_MODES: Tuple[str, ...] = ("baseline", "lite", "extended", "full")

# Planck 2018 likelihood folders (in Observations/) per Kosmulator tag
PLANCK_CLIK_FILES: Dict[str, Dict[str, str]] = {
    "baseline": {"CMB_hil": "plik_rd12_HM_v22b_TTTEEE.clik", "CMB_hil_TT": "plik_rd12_HM_v22_TT.clik"},
    "lite":     {"CMB_hil": "plik_lite_v22_TTTEEE.clik", "CMB_hil_TT": "plik_lite_v22_TT.clik"},
}
PLANCK_CLIK_FILES["extended"] = dict(PLANCK_CLIK_FILES["baseline"])
PLANCK_CLIK_FILES["full"] = dict(PLANCK_CLIK_FILES["baseline"])
PLANCK_LOWL_FILES: Dict[str, str] = {
    "CMB_lowl": "simall_100x143_offlike5_EE_Aplanck_B.clik",      # low-l EE (SimAll, "lowE")
    "CMB_lowl_TT": "commander_dx12_v3_2_29.clik",                 # low-l TT (Commander, "lowl")
}

# Planck tags (primary CMB: everything but lensing)
# Ground-based CMB through candl and the Planck + ACT (+ SPT) combinations (item 57;
# Kosmulator_main/ground_cmb.py):
#   CMB_ACT            ACT DR6 TT/TE/EE, foreground-marginalised (ACT-lite), l = 600 - 8500
#                      (Louis et al. 2025, arXiv:2503.14452), candl_data's ACT_DR6_TTTEEE
#   CMB_SPT            SPT-3G D1 T&E, foreground-marginalised (SPT-lite), l = 400 - 4000
#                      (Camphuis et al. 2025, arXiv:2506.20707), spt_candl_data's SPT3G_D1_TnE_lite
#                      (both data sets in Observations/candl; only the candl code is installed)
#   CMB_planck_ACTcut  Planck 2018 plik_lite TT,TE,EE without l > 1000 (TT) and l > 600 (TE, EE),
#                      ACT DR6's "PlanckActCut"
#   CMB_lowl_sroll2    Planck low-l EE from the SRoll2 maps (Pagano et al. 2020, arXiv:1908.09856)
#   CMB_PACT           P-ACT: CMB_lowl_TT + CMB_lowl_sroll2 + CMB_planck_ACTcut + CMB_ACT
#   CMB_SPA            CMB-SPA: CMB_PACT + CMB_SPT, as the SPT collaboration's SPA-lite set-up
GROUND_CMB_TAGS: Tuple[str, ...] = ("CMB_ACT", "CMB_SPT", "CMB_planck_ACTcut", "CMB_lowl_sroll2",
                                    "CMB_PACT", "CMB_SPA")
CMB_BUNDLES: Dict[str, Tuple[str, ...]] = {
    "CMB_PACT": ("CMB_lowl_TT", "CMB_lowl_sroll2", "CMB_planck_ACTcut", "CMB_ACT"),
    "CMB_SPA": ("CMB_lowl_TT", "CMB_lowl_sroll2", "CMB_planck_ACTcut", "CMB_ACT", "CMB_SPT"),
}
#: l_max each likelihood needs from CLASS (candl: its theory range; Planck: CMB_CLASS_LMAX_PLANCK)
CMB_TAG_LMAX: Dict[str, int] = {"CMB_ACT": 8501, "CMB_SPT": 4095,
                                  "CMB_lensing_SPT": 3100}     # MUSE phi phi windows reach L = 3073
#: Planck high-l cut of ACT DR6's P-ACT (TT, TE, EE): bins above it are dropped
PLANCK_ACT_CUT_LMAX: Tuple[int, int, int] = (1000, 600, 600)
#: Nuisance parameters of the ground-based likelihoods: (reference value, prior box, Gaussian
#: (mean, sigma) or None). A_act: ACT's overall calibration, prior from the dipole
#: calibration (candl's internal prior and P-ACT's cal_dip_prior); tied to A_planck (one
#: parameter) when Planck data are in the group, as in P-ACT and SPA. P_act: ACT's
#: polarisation efficiency, flat 0.9 - 1.1. Tcal, Ecal: SPT-lite's temperature and E-mode
#: calibrations (Gaussian 1 +- 0.0036 and flat 0.8 - 1.2, as in spt_candl_data's SPA-lite).
GROUND_NUISANCE_SAMPLED: Dict[str, Tuple[float, Tuple[float, float], Tuple[float, float] | None]] = {
    "A_act": (1.0, (0.985, 1.015), (1.0, 0.003)),
    "P_act": (1.0, (0.9, 1.1), None),
    "Tcal": (1.0, (0.982, 1.018), (1.0, 0.0036)),
    "Ecal": (1.0, (0.8, 1.2), None),
}
GROUND_NUISANCE_PARAMS: Dict[str, Tuple[str, ...]] = {
    "CMB_ACT": ("A_act", "P_act"), "CMB_SPT": ("Tcal", "Ecal"),
    "CMB_planck_ACTcut": ("A_planck",), "CMB_lowl_sroll2": ("A_planck",),
    "CMB_PACT": ("A_planck", "P_act"), "CMB_SPA": ("A_planck", "P_act", "Tcal", "Ecal"),
}
#: Nuisance parameters of the likelihoods (CT-15, CT-37): not drawn in the corner plots unless
#: corner_show_nuisance (Kosmulator.py) or --corner-show-all-cmb-params; listed after a "nuisance" line
#: in the parameter table (CT-43). The tables and statistics keep every parameter.
#: CMB: every plik nuisance, A_planck, and the ACT and SPT calibrations.
CMB_NUISANCE_PARAM_NAMES: Set[str] = (set(PLANCK_NUISANCE_ALL) | set(PLANCK_NUISANCE_FIXED)
                                      | set(GROUND_NUISANCE_SAMPLED) | {"A_planck"})
#: SN calibration and standardisation
SN_NUISANCE_PARAM_NAMES: Set[str] = {"M_abs", "alpha_JLA", "beta_JLA"}
#: Nuisance parameters drawn in the corner plots all the same (Renier, 18 Oct: A_planck, as before)
CORNER_SHOWN_NUISANCE: Set[str] = {"A_planck"}
#: corner_show_nuisance in Kosmulator.py (MCMC_setup.main sets it): True draws the nuisance parameters
CORNER_SHOW_NUISANCE: bool = False

#: tau prior for a group with ACT or SPT and no low-l EE likelihood: the one candl applies by
#: default (SPT-lite: from the Planck PR4 maps, as in Camphuis et al. 2025; ACT-lite: ACT DR6)
CMB_TAU_PRIORS: Dict[str, Tuple[float, float]] = {"CMB_SPT": (0.051, 0.006), "CMB_ACT": (0.0566, 0.0058)}
#: Likelihoods that constrain tau through low-l EE
CMB_LOWL_EE_TAGS: Tuple[str, ...] = ("CMB_lowl", "CMB_lowl_sroll2", "CMB_PACT", "CMB_SPA")
#: Data points: ACT-lite 135 and SPT-lite 196 band powers, SRoll2 l = 2 - 29
GROUND_N_DATA: Dict[str, int] = {"CMB_ACT": 135, "CMB_SPT": 196, "CMB_lowl_sroll2": 28}
#: SRoll2 low-l EE probability table (Cobaya's planck_native_data release v1, planck_sroll2_lowE.zip)
SROLL2_TABLE_FILE: str = "sroll2_prob_table.txt"

# CMB lensing beyond Planck 2018 (item 61; Kosmulator_main/ground_cmb.py):
#   CMB_lensing_ACTPlanck  ACT DR6 + Planck PR4 (NPIPE) lensing power spectrum as one likelihood with
#                          their joint covariance (act_dr6_lenslike, variant ACT_LENS_VARIANT; ACT
#                          L = 40 - 763 and Planck PR4 L = 8 - 400 in the baseline; Madhavacheril et al.
#                          2024, arXiv:2304.05203; Qu et al. 2024, arXiv:2304.05202; Carron et al. 2022,
#                          arXiv:2206.07773). The lensing of ACT's P-ACT-LB and of the DESI DR2 CMB
#                          combinations, and the most used choice with P-ACT in 2025 - 2026. With primary
#                          CMB in the group the likelihood corrections (normalisation and N1, linear in the
#                          CMB spectra) are applied; alone, the CMB-marginalised covariance is used.
#                          Data: ACT_dr6_likelihood_<ACT_LENS_VERSION>.tgz from LAMBDA, unpacked so that
#                          Observations/<ACT_LENS_DIR>/<ACT_LENS_VERSION>/clkk_bandpowers_act.txt exists
#                          (or act_dr6_lenslike.get_data(), the package's own folder). The corrections are
#                          binned once into a 2.9 MB cache (Observations/<ACT_LENS_DIR>/.cache/), so pool
#                          workers do not each hold the package's 1.4 GB of derivative matrices.
#   CMB_lensing_SPT        SPT-3G 2019 - 2020 lensing power spectrum from MUSE, phi phi only (16 band powers,
#                          L = 20 - 3073; delensed EE and systematics marginalised; Ge et al. 2025,
#                          arXiv:2411.06000), a numpy port of muse3glike 1.0.0 (ground_cmb.MusePhiPhi).
#                          With CMB_lensing_ACTPlanck it is an independent likelihood, as in Camphuis et al.
#                          2025 (arXiv:2506.20707) and most CMB-SPA papers; that neglects the SPT x ACT and
#                          SPT x Planck lensing cross-covariances, which only the joint ACT + SPT likelihood
#                          models (Qu et al. 2025, arXiv:2504.20038). Data: Observations/<SPT_MUSE_DIR>/
#                          <SPT_MUSE_FILE> (from muse_3g_like_march_2025.zip), or an installed muse3glike.
EXT_LENSING_TAGS: Tuple[str, ...] = ("CMB_lensing_ACTPlanck", "CMB_lensing_SPT")
ACT_LENS_VARIANT: str = "actplanck_baseline"
ACT_LENS_VERSION: str = "v1.2"
ACT_LENS_DIR: str = "act_dr6_lenslike"
#: simulations behind the covariances, for the Hartlap factor (as Cobaya's ACTDR6LensLike)
ACT_LENS_NSIMS: Tuple[int, int] = (792, 400)
SPT_MUSE_DIR: str = "spt3g_muse_lensing"
SPT_MUSE_FILE: str = "90_150_220.h5"
#: Every CMB lensing tag (Planck 2018 and those above)
LENSING_TAGS: Tuple[str, ...] = ("CMB_lensing",) + EXT_LENSING_TAGS

# Lensing without primary CMB (item 62; Kosmulator.py lensing_only_priors, default True). The lensing
# power spectrum alone constrains about sigma_8 Omega_m^0.25 and leaves n_s free; analyses of lensing
# without primary CMB put a prior on it, n_s ~ N(0.96, 0.02) (Madhavacheril et al. 2024,
# arXiv:2304.05203, Table 1, "Lensing + BAO"). tau_reio enters none of these likelihoods (no primary
# CMB, so no lensed spectra are used), so it is fixed at its reference value instead of sampled (that
# table has no tau either). Omega_b h^2: add a BBN tag (e.g. BBN_DH), which also calibrates r_d, as
# in that paper. KOSM_LENSING_ONLY_PRIORS carries the switch to spawned pool workers.
LENSING_ONLY_NS_PRIOR: Tuple[float, float] = (0.96, 0.02)
#: tau_reio of such a group when Kosmulator.py's reference_values has none (Planck 2018 TT,TE,EE+lowE)
TAU_REIO_FALLBACK: float = 0.0544
LENSING_ONLY_PRIORS: bool = os.environ.get("KOSM_LENSING_ONLY_PRIORS", "1") != "0"


def set_lensing_only_priors(on: bool = True) -> None:
    """Item 62 switch of this run (Kosmulator.py lensing_only_priors); also KOSM_LENSING_ONLY_PRIORS."""
    global LENSING_ONLY_PRIORS
    LENSING_ONLY_PRIORS = bool(on)
    os.environ["KOSM_LENSING_ONLY_PRIORS"] = "1" if LENSING_ONLY_PRIORS else "0"

CMB_TAGS: Tuple[str, ...] = (("CMB_hil", "CMB_hil_TT", "CMB_lowl", "CMB_lowl_TT", "CMB_lensing") + GROUND_CMB_TAGS
                             + EXT_LENSING_TAGS)
CMB_PRIMARY_TAGS: Tuple[str, ...] = ("CMB_hil", "CMB_hil_TT", "CMB_lowl", "CMB_lowl_TT") + GROUND_CMB_TAGS


def cmb_components(tag: str) -> Tuple[str, ...]:
    """The likelihoods behind a CMB tag: a combination's parts, else the tag itself."""
    return CMB_BUNDLES.get(tag, (tag,))


def cmb_group_components(group) -> Set[str]:
    """Every CMB likelihood a group evaluates (combinations expanded)."""
    out: Set[str] = set()
    for o in group or ():
        out |= set(cmb_components(str(o)))
    return out


def group_has_primary_cmb(group) -> bool:
    """True if a group has a primary-CMB likelihood (temperature or polarisation), not only lensing.
    Low-l data alone count, as in Kosmulator's rule for Planck lensing (raw with any primary CMB,
    CMB-marginalised without; Kosmulator_MCMC.model_likelihood), kept for ACT + Planck lensing
    (Renier, 12 Oct)."""
    return bool(cmb_group_components(group) & set(CMB_PRIMARY_TAGS))


def group_is_lensing_only(group) -> bool:
    """True if a group's CMB data are lensing only (a lensing tag and no primary CMB; item 62)."""
    comps = cmb_group_components(group)
    return bool(comps & set(LENSING_TAGS)) and not (comps & set(CMB_PRIMARY_TAGS))


def cmb_lmax_for_groups(groups) -> int:
    """l_max of the CLASS spectra a run needs (CMB_CLASS_LMAX_PLANCK unless ACT or SPT is in it)."""
    lmax = CMB_CLASS_LMAX_PLANCK
    for g in groups or ():
        for c in cmb_group_components(g):
            lmax = max(lmax, CMB_TAG_LMAX.get(c, 0))
    return lmax


def cmb_tau_prior(group) -> Tuple[float, float] | None:
    """tau prior of a group with ACT or SPT but no low-l EE likelihood (CMB_TAU_PRIORS), else None."""
    comps = cmb_group_components(group)
    if comps & set(CMB_LOWL_EE_TAGS) or any(str(o) in CMB_LOWL_EE_TAGS for o in group or ()):
        return None
    for tag in ("CMB_SPT", "CMB_ACT"):
        if tag in comps:
            return CMB_TAU_PRIORS[tag]
    return None


def cmb_gaussian_priors(group) -> Dict[str, Tuple[float, float]]:
    """
    Gaussian priors on the CMB parameters of a group: Planck's nuisance priors, the ground-based
    nuisance priors, A_planck with ACT's calibration prior (1 +- 0.003) when it stands for ACT's
    A_act too (P-ACT, SPA), tau when ACT or SPT has no low-l EE likelihood with it, and n_s for
    lensing without primary CMB (item 62, LENSING_ONLY_PRIORS).
    """
    out: Dict[str, Tuple[float, float]] = dict(PLANCK_GAUSSIAN_PRIORS)
    out.update({n: v[2] for n, v in GROUND_NUISANCE_SAMPLED.items() if v[2] is not None})
    comps = cmb_group_components(group)
    if "CMB_ACT" in comps and comps & {"CMB_planck_ACTcut", "CMB_lowl_sroll2", "CMB_lowl_TT", "CMB_lowl",
                                       "CMB_hil", "CMB_hil_TT", "CMB_lensing"}:
        out["A_planck"] = GROUND_NUISANCE_SAMPLED["A_act"][2]
    tp = cmb_tau_prior(group)
    if tp is not None:
        out["tau_reio"] = tp
    if LENSING_ONLY_PRIORS and group_is_lensing_only(group):
        out["n_s"] = LENSING_ONLY_NS_PRIOR          # item 62
    return out


def planck_clik_file(tag: str) -> str:
    """Planck likelihood folder name for a Kosmulator tag under the current CMB_NUISANCE."""
    if tag in PLANCK_LOWL_FILES:
        return PLANCK_LOWL_FILES[tag]
    return PLANCK_CLIK_FILES[CMB_NUISANCE][tag]


def planck_sampled_nuisances(tag: str) -> Set[str]:
    """Planck nuisance parameters sampled for one Kosmulator CMB tag under the current
    CMB_NUISANCE (A_planck is shared)."""
    lite = (CMB_NUISANCE == "lite")
    if tag == "CMB_hil":
        if lite:
            return {"A_planck"}
        out = set(PLANCK_TTTEEE_NUISANCE)
        if CMB_NUISANCE in ("extended", "full"):
            out |= set(PLANCK_NUISANCE_EXTENDED)
        if CMB_NUISANCE == "full":
            out |= set(PLANCK_NUISANCE_FULL_EXTRA)
        return out
    if tag == "CMB_hil_TT":
        if lite:
            return {"A_planck"}
        return set(PLANCK_TT_ONLY_NUISANCE) | (set(PLANCK_TT_FULL_EXTRA) if CMB_NUISANCE == "full" else set())
    if tag in ("CMB_lowl", "CMB_lowl_TT", "CMB_lensing"):
        return {"A_planck"}
    if "A_planck" in GROUND_NUISANCE_PARAMS.get(tag, ()):
        return {"A_planck"}              # Planck parts of the ground-based combinations (item 57)
    return set()


# Size of each Planck likelihood's data vector (bins or multipoles fitted), read from the
# clik files: plik's smica covariance (2289 for TT,TE,EE, 765 for TT), plik_lite's bins
# (613, 215), SimAll and Commander l = 2 - 29 (28 each), the lensing bandpowers (9). N in the
# statistics (reduced chi^2, BIC, AICc); clik's Python API does not report it.
PLANCK_N_DATA: Dict[str, int] = {
    "plik_rd12_HM_v22b_TTTEEE.clik": 2289,
    "plik_rd12_HM_v22_TT.clik": 765,
    "plik_lite_v22_TTTEEE.clik": 613,
    "plik_lite_v22_TT.clik": 215,
    "simall_100x143_offlike5_EE_Aplanck_B.clik": 28,
    "commander_dx12_v3_2_29.clik": 28,
    "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing": 9,
    "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_CMBmarged.clik_lensing": 9,
}


def planck_fixed_now() -> Dict[str, float]:
    """plik nuisance values held fixed under the current CMB_NUISANCE (Planck's baseline values;
    "extended" samples 9 of them, "full" all)."""
    if CMB_NUISANCE == "full":
        return {}
    if CMB_NUISANCE == "extended":
        return {n: v for n, v in PLANCK_NUISANCE_FIXED.items() if n not in PLANCK_NUISANCE_EXTENDED}
    return dict(PLANCK_NUISANCE_FIXED)


def planck_n_data(tag: str, lensing_mode: str = "raw") -> int:
    """Number of data points of a Planck tag's likelihood under the current CMB_NUISANCE."""
    if tag == "CMB_lensing":
        f = ("smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing" if lensing_mode == "raw"
             else "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_CMBmarged.clik_lensing")
        return PLANCK_N_DATA[f]
    return PLANCK_N_DATA[planck_clik_file(tag)]


# Best-fit polish of a group with Planck data (Post_processing.find_polished_mle): one
# Nelder-Mead run of at most this many likelihood calls (each runs CLASS and clik), from the
# best of the posterior median and the 5 best distinct samples (one call each beforehand).
# On the step 8 lite chain (7 parameters) chi^2 stopped changing after about 200 calls.
# Environment variable KOSM_POLISH_MAXFEV_CMB overrides it.
POLISH_MAXFEV_CMB: int = 300
# Groups with two or more CMB nuisance parameters (plik baseline: 21; P-ACT, SPA: 2 - 4) are
# polished nested (item 38): Nelder-Mead over the parameters that change the spectra, at most
# this many of its points, each with the nuisances optimised by L-BFGS-B on cached spectra.
# t10 (27 parameters, cloud): chi^2 + priors 2774.87 after 60 points (data chi^2 2773.04) against
# 2791.43 (2786.06) for the joint Nelder-Mead's 300 calls; still falling by about 0.01 per point at 60.
POLISH_NESTED_MAXFEV_CMB: int = 100

#: Calls of the Nelder-Mead pre-fit that centres the initial walker ball, for groups
#: with Planck data (item 34; other groups 2000). KOSM_OPT_MAXFUN overrides.
PREFIT_MAXFEV_CMB: int = 300

# WAIC: the draws used for the pointwise log-likelihood matrix (at most 1000) are picked
# with this seed, so a rerun of the statistics gives the same WAIC
WAIC_SUBSAMPLE_SEED: int = 20260108

#: Posterior draws for WAIC (item 52): late-time data (cheap; Monte Carlo spread of WAIC
#: about 0.34 with 1000 draws, 0.16 with 4000, for CC + Pantheon) and groups with Planck data
#: (one CLASS run per draw, through the worker pool when there is one).
WAIC_DRAWS: int = 4000
WAIC_DRAWS_CMB: int = 1000

#: WAIC of groups with Planck data (item 53). The Planck likelihoods give one total
#: log-likelihood per point, not one per data point, so each enters WAIC as one term,
#: which costs one lensed CLASS run per WAIC draw (1000 draws). False: WAIC is n/a, with
#: a note, for every group with Planck data. Set by waic_include_cmb in Kosmulator.py.
WAIC_INCLUDE_CMB: bool = False
