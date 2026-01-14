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

#: Neutron lifetime [s] used consistently in BBN (AlterBBN + grid)
TAU_N_DEFAULT: float = 879.4

#: Legacy fixed sound horizon [Mpc] used in "singleton" BAO/DESI modes
R_D_SINGLETON: float = 147.5

#: GR-like growth index used when fσ8 is a singleton (and as a default)
GAMMA_FS8_SINGLETON: float = 0.55


# ======================================================================
# 2. Data / path defaults
# ======================================================================

#: Base directory where all observational data live
OBSERVATIONS_BASE: str = "./Observations"

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
    "DESI_DR1":      (r"DESI_{DR1}",         "DESI DR1"),
    "DESI_DR2":      (r"DESI_{DR2}",         "DESI DR2"),
    
    #Big Bang Nucleosythesis
    "BBN_PryMordial":      (r"BBN (primordial)",             "BBN (primordial)"),
    "BBN_DH":              (r"BBN D/H",                      "BBN D/H"),
    "BBN_DH_AlterBBN":     (r"BBN D/H (AlterBBN)",           "BBN D/H (AlterBBN)"),
}

# --- Plot grouping / colours --------------------------------------------

#: Which observation types can share an axis column in best-fit panels
COMBINE_GROUPS: List[Set[str]] = [
    {"OHD", "CC"},
    {"PantheonP", "PantheonP_SH0ES", "Pantheon", "JLA", "DESY5", "Union3"},
    {"BAO", "DESI_DR1", "DESI_DR2"},
]

#: Global colour palette for plotting
DEFAULT_PLOT_COLORS: List[str] = [
     "b", "green", "r", "cyan", "purple", "grey", "yellow", "m", "k", "gray",
    "orange", "pink", "crimson", "darkred", "salmon",
]

#: Default colours used by observation data in summary panels
OBS_COLOR_ORDER: List[str] = DEFAULT_PLOT_COLORS

#: Default colour used to draw model curves
MODEL_COLOR: str = "r"

# --- DESI/BAO code → (label, linestyle) ---------------------------------

CODE_STYLE: Dict[int, Tuple[str, object]] = {
    8: (r"$D_M/r_s$", "-"),
    6: (r"$D_H/r_s$", "--"),
    5: (r"$D_A/r_s$", "-."),
    3: (r"$D_V/r_d$", ":"),
    7: (r"$r_s/D_V$", (0, (1, 2))),
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
    "A_planck": (1.0, (0.9, 1.1)),
    "calib_100T": (1.0, (0.9, 1.1)),
    "calib_217T": (1.0, (0.9, 1.1)),
    "calib_100P": (1.0, (0.9, 1.1)),
    "calib_143P": (1.0, (0.9, 1.1)),
    "calib_217P": (1.0, (0.9, 1.1)),
    "A_pol":      (0.75, (0.0, 1.5)),

    # Foregrounds: CIB, SZ, tSZ×CIB, kSZ
    "A_cib_217": (30.0, (0.0, 100.0)),
    "cib_index": (-1.3, (-3.0, 0.0)),
    "xi_sz_cib": (0.45, (0.0, 3.0)),
    "A_sz":      (2.5,  (0.0, 15.0)),
    "ksz_norm":  (1.6,  (0.0, 15.0)),

    # Poisson point sources
    "ps_A_100_100": (185.0, (0.0, 500.0)),
    "ps_A_143_143": (72.0,  (0.0, 500.0)),
    "ps_A_143_217": (59.0,  (0.0, 500.0)),
    "ps_A_217_217": (151.0, (0.0, 500.0)),

    # Galactic dust TT @545 template
    "gal545_A_100":      (8.6,  (0.0, 200.0)),
    "gal545_A_143":      (10.6, (0.0, 200.0)),
    "gal545_A_143_217":  (23.5, (0.0, 200.0)),
    "gal545_A_217":      (91.9, (0.0, 200.0)),

    # Galactic EE
    "galf_EE_A_100":      (20.0, (0.0, 100.0)),
    "galf_EE_A_100_143":  (20.0, (0.0, 100.0)),
    "galf_EE_A_100_217":  (20.0, (0.0, 100.0)),
    "galf_EE_A_143":      (20.0, (0.0, 100.0)),
    "galf_EE_A_143_217":  (20.0, (0.0, 100.0)),
    "galf_EE_A_217":      (20.0, (0.0, 100.0)),
    "galf_EE_index":      (0.0,  (-5.0, 5.0)),

    # Galactic TE
    "galf_TE_A_100":      (20.0, (0.0, 100.0)),
    "galf_TE_A_100_143":  (20.0, (0.0, 100.0)),
    "galf_TE_A_100_217":  (20.0, (0.0, 100.0)),
    "galf_TE_A_143":      (20.0, (0.0, 100.0)),
    "galf_TE_A_143_217":  (20.0, (0.0, 100.0)),
    "galf_TE_A_217":      (20.0, (0.0, 100.0)),
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
