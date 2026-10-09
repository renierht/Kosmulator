# Kosmulator_main/constants.py
from __future__ import annotations

from typing import Dict, List, Set, Tuple

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

#: Standard-model effective number of relativistic species
N_EFF_DEFAULT: float = 3.046

#: Radiation in the late-time E(z) of the built-in models (LCDM, w0waCDM,
#: f1CDM, NonLinear_IDE_2): Omega_r h^2 = omega_gamma (1 + 0.22711 N_ur), with
#: photons at T_CMB_DEFAULT and N_ur massless neutrino species. Omega_DE is
#: reduced by Omega_r so the model stays flat. N_UR_LATE = N_EFF_DEFAULT treats
#: all neutrinos as massless, as the CLASS r_d call does; DESI/Planck put one
#: 0.06 eV neutrino in Omega_m, which corresponds to N_UR_LATE = 2.0328.
#: A sampled or fixed 'Omega_r' in the parameters overrides this.
LATE_TIME_RADIATION: bool = True
N_UR_LATE: float = N_EFF_DEFAULT
#: h used for Omega_r when a group does not sample H_0 (e.g. f, f_sigma_8 alone)
H0_RADIATION_FALLBACK: float = 67.4

#: Neutron lifetime [s] used consistently in BBN (AlterBBN + grid)
TAU_N_DEFAULT: float = 879.4

#: Legacy fixed sound horizon [Mpc] used in "singleton" BAO/DESI modes
R_D_SINGLETON: float = 147.5

#: GR-like growth index used when fσ8 is a singleton (and as a default)
GAMMA_FS8_SINGLETON: float = 0.55

#: Uncalibrated supernovae (Pantheon+ without SH0ES, DES-Y5, Union3, JLA,
#: Pantheon): marginalise the magnitude offset (M_abs, or the H0 normalisation
#: of the distance moduli) analytically, as in the Cobaya SN likelihoods
#: (use_abs_mag: False). M_abs is then not sampled for PantheonP, and a BAO/DESI
#: group whose other data fix neither H0 nor r_d uses the fixed r_d below, so
#: H_0 measures h*r_d as DESI's hrd does. Set False for the old behaviour
#: (M_abs sampled for PantheonP; H0 acting as the offset of distance moduli).
SN_MARGINALISE_OFFSET: bool = True

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
#: worker pool (vectorised zeus and emcee, or any run with --num_cores 1).
#: None keeps NumPy's default (all cores). The best value depends on the
#: machine: compare the steps/s of a short run with 1, 2 and 4 threads.
#: In a 2-core cloud test 2 threads were 1.3x faster than 1 for DESI DR2 +
#: DES-Y5 and equal for Pantheon+; 1 is kept until measured on the target
#: machine, so a run never takes every core by default.
MAIN_BLAS_THREADS_NO_POOL = 1

# ----------------------------------------------------------------------
# Convergence rule (zeus and emcee, utils.ConvergenceMonitor)
# ----------------------------------------------------------------------
#: Checked every --autocorr-check-every steps on the chain after burn-in. tau_max
#: is the largest integrated autocorrelation time over the sampled parameters,
#: estimated with zeus's default method (utils.autocorr_time_mk).
#: Chain after burn-in must be at least this many tau_max long (emcee and zeus
#: documentation: tau estimates are reliable beyond ~50 tau).
CONV_TAU_FACTOR = 50.0
#: Effective sample size of the slowest parameter, N_post x walkers / tau_max
#: (2000 gives a Monte Carlo error of ~2% of sigma on a posterior mean).
CONV_ESS_MIN = 2000.0
#: Split-Rhat limit (each walker's chain cut in two halves; Gelman et al. 2013;
#: 1.01 is the zeus documentation's SplitRCallback tolerance).
CONV_RHAT_MAX = 1.01
#: Relative change of tau_max between two checks below which tau counts as
#: stable; Kosmulator.py's `convergence` sets it for a run.
CONV_TAU_RTOL = 0.05


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
    "CMB_lowl":        (r"Planck low-$\ell$",          "Planck low-ℓ"),
    "CMB_hil":         (r"Planck high-$\ell$ TTTEEE",  "Planck high-ℓ TTTEEE"),
    "CMB_hil_TT":      (r"Planck high-$\ell$ TT",      "Planck high-ℓ TT"),
    "CMB_lensing":     (r"Planck lensing",             "Planck lensing"),

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
# 6. Planck CMB nuisance parameters (centralised here)
# These give default true values and prior ranges for CMB runs.
# ======================================================================

PLANCK_NUISANCE_DEFAULTS: Dict[str, Tuple[float, Tuple[float, float]]] = {
    # Calibration & overall amplitude
    "A_planck": (1.0, (0.995, 1.005)),
    "calib_100T": (1.0, (0.99, 1.01)),
    "calib_217T": (1.0, (0.99, 1.01)),
    "calib_100P": (1.0, (0.99, 1.01)),
    "calib_143P": (1.0, (0.99, 1.01)),
    "calib_217P": (1.0, (0.99, 1.01)),
    "A_pol":      (0.75, (0.0, 1.5)),

    # Foregrounds: CIB, SZ, tSZ×CIB, kSZ
    "A_cib_217": (30.0, (0.0, 100.0)),
    "cib_index": (-1.3, (-2.0, -0.5)),
    "xi_sz_cib": (0.45, (0.0, 3.0)),
    "A_sz":      (2.5,  (0.0, 15.0)),
    "ksz_norm":  (1.6,  (0.0, 15.0)),

    # Poisson point sources
    "ps_A_100_100": (185.0, (50.0, 300.0)),
    "ps_A_143_143": (72.0,  (20.0, 150.0)),
    "ps_A_143_217": (59.0,  (20.0, 150.0)),
    "ps_A_217_217": (151.0, (50.0, 300.0)),

    # Galactic dust TT @545 template
    "gal545_A_100":      (8.6,  (0.0, 200.0)),
    "gal545_A_143":      (10.6, (0.0, 200.0)),
    "gal545_A_143_217":  (23.5, (0.0, 200.0)),
    "gal545_A_217":      (91.9, (0.0, 200.0)),

    # Galactic EE
    "galf_EE_A_100":      (20.0, (5.0, 50.0)),
    "galf_EE_A_100_143":  (20.0, (5.0, 50.0)),
    "galf_EE_A_100_217":  (20.0, (5.0, 50.0)),
    "galf_EE_A_143":      (20.0, (5.0, 50.0)),
    "galf_EE_A_143_217":  (20.0, (5.0, 50.0)),
    "galf_EE_A_217":      (20.0, (5.0, 50.0)),
    "galf_EE_index":      (0.0,  (-5.0, 5.0)),

    # Galactic TE
    "galf_TE_A_100":      (20.0, (5.0, 50.0)),
    "galf_TE_A_100_143":  (20.0, (5.0, 50.0)),
    "galf_TE_A_100_217":  (20.0, (5.0, 50.0)),
    "galf_TE_A_143":      (20.0, (5.0, 50.0)),
    "galf_TE_A_143_217":  (20.0, (5.0, 50.0)),
    "galf_TE_A_217":      (20.0, (5.0, 50.0)),
    "galf_TE_index":      (0.0,  (-5.0, 5.0)),

    # End-to-end EE noise
    "A_cnoise_e2e_100_100_EE": (5.0, (0.0, 10.0)),
    "A_cnoise_e2e_143_143_EE": (5.0, (0.0, 10.0)),
    "A_cnoise_e2e_217_217_EE": (5.0, (0.0, 10.0)),

    # Spurious beam leakage (TT)
    "A_sbpx_100_100_TT": (1.0, (0.0, 3.0)),
    "A_sbpx_143_143_TT": (1.0, (0.0, 3.0)),
    "A_sbpx_143_217_TT": (1.0, (0.0, 3.0)),
    "A_sbpx_217_217_TT": (1.0, (0.0, 3.0)),

    # Spurious beam leakage (EE)
    "A_sbpx_100_100_EE": (1.0, (0.0, 3.0)),
    "A_sbpx_100_143_EE": (1.0, (0.0, 3.0)),
    "A_sbpx_100_217_EE": (1.0, (0.0, 3.0)),
    "A_sbpx_143_143_EE": (1.0, (0.0, 3.0)),
    "A_sbpx_143_217_EE": (1.0, (0.0, 3.0)),
    "A_sbpx_217_217_EE": (1.0, (0.0, 3.0)),
}

PLANCK_TT_ONLY_NUISANCE: Set[str] = {
    "A_planck",
    "calib_100T", "calib_217T",
    "A_sbpx_100_100_TT","A_sbpx_143_143_TT",
    "A_sbpx_143_217_TT","A_sbpx_217_217_TT",
    "A_cib_217","cib_index","xi_sz_cib","A_sz","ksz_norm",
    "ps_A_100_100","ps_A_143_143","ps_A_143_217","ps_A_217_217",
    "gal545_A_100","gal545_A_143","gal545_A_143_217","gal545_A_217",
}

PLANCK_TTTEEE_NUISANCE: Set[str] = (
    set(PLANCK_NUISANCE_DEFAULTS.keys())
    - {"calib_100P","calib_143P","calib_217P"}
    | {"calib_100P","calib_143P","calib_217P","A_pol"}
)


# Size of each Planck likelihood's data vector (bins or multipoles fitted), read from the
# clik files: plik's smica covariance (2289 for TT,TE,EE, 765 for TT), SimAll l = 2 - 29 (28),
# the lensing bandpowers (9); plik_lite (613, 215) and Commander (28) for later use. N in the
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
# The Planck likelihood folder each tag uses (as in Config.load_all_data and MCMC_setup)
PLANCK_TAG_FILES: Dict[str, str] = {
    "CMB_hil": "plik_rd12_HM_v22b_TTTEEE.clik",
    "CMB_hil_TT": "plik_rd12_HM_v22_TT.clik",
    "CMB_lowl": "simall_100x143_offlike5_EE_Aplanck_B.clik",
}


def planck_n_data(tag: str, lensing_mode: str = "raw") -> int:
    """Number of data points of a Planck tag's likelihood (PLANCK_N_DATA)."""
    if tag == "CMB_lensing":
        f = ("smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing" if lensing_mode == "raw"
             else "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_CMBmarged.clik_lensing")
        return PLANCK_N_DATA[f]
    return PLANCK_N_DATA[PLANCK_TAG_FILES[tag]]

# WAIC: the draws used for the pointwise log-likelihood matrix (at most 1000) are picked
# with this seed, so a rerun of the statistics gives the same WAIC
WAIC_SUBSAMPLE_SEED: int = 20260108
