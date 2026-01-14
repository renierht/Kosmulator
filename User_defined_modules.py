# User_defined_modules.py  — clean, extensible, vectorized
# --------------------------------------------------------
"""
Central place for *user cosmology models* and small wrappers.

This is where you:
  1. Implement E(z) for a new background model (flat or not).
  2. (Optionally) define any prior / sanity restriction for its parameters.
  3. Register the model name + parameter list in the model registry.
  4. (Optional) expose CMB Cl wrappers for that model.

QUICK START for adding a new model
----------------------------------
Example outline:

    def MyMG_MODEL_vectorised(z, p):
        z = _asarray(z)
        # compute E(z) = H(z)/H0 here using params in p
        return Ez

    def restrict_MyMG_param(x: float) -> bool:
        return x < 1.0   # example guard rail

    # Scroll down to the "Model registry / discovery" section and add:
    # _MODEL_REGISTRY["MyMG_v"] = (MyMG_MODEL_vectorised, ["Omega_m", "my_param"])
    #
    # and (optionally) in the "Model restrictions" section:
    # restrictions_map["MyMG_v"] = {"my_param": restrict_MyMG_param}

The rest of the file is mostly helpers that other modules (likelihoods,
plotting, rd_helpers) already use; users typically only touch the
"Core E(z) models", the restrictions, and the model registry.
"""

from __future__ import annotations

from typing import Union, Dict, Callable, List, Tuple

import logging
import numpy as np
from scipy.optimize import fsolve
import classy

from Kosmulator_main.constants import C_KM_S
import logging
logger = logging.getLogger(__name__)

# These come from your MCMC runtime (vectorized implementations)
from Kosmulator_main.utils import (
    Comoving_distance_vectorized as _vect_comove,
    matter_density_z_array as _omega_m_z_arr,
    integral_term_array as _integral_term_arr,
    asarray as _asarray,                    # safe np.asarray wrapper, used everywhere
    ensure_background_params as _ensure_background_params,
)

Number = Union[float, np.ndarray]

__all__ = [
    # background models
    "LCDM_MODEL_vectorised",
    "LCDM_MODEL_non_vectorised",
    "f1CDM_MODEL_vectorised",
    "f1CDM_MODEL_non_vectorised",
    # registry helpers
    "Get_model_function",
    "Get_model_names",
    "Get_model_restrictions",
    "register_model",
    # common wrappers
    "Hubble",
    "Comoving_distance_vectorized",
    "matter_density_z_array",
    "integral_term_array",
]

# ============================================================================
#  Core E(z) models (dimensionless expansion rate E(z) = H(z)/H0)
#  ---------------------------------------------------------------------------
#  This is the main place users add their own background models.
# ============================================================================


def LCDM_MODEL_vectorised(z: Number, p: Dict[str, float]) -> Number:
    """
    Flat ΛCDM:  E^2(z) = Ω_m (1+z)^3 + (1 - Ω_m)

    Parameters in `p`:
      • Omega_m
    """
    z = _asarray(z)
    Om = float(p["Omega_m"])
    E2 = Om * (1 + z) ** 3 + (1 - Om)

    if (not np.isfinite(E2).all()) or (E2.min() <= 0):
        out = np.full_like(z, np.nan)
    else:
        out = np.sqrt(E2)

    return out if out.size > 1 else float(out)


def LCDM_MODEL_non_vectorised(z: Number, p: Dict[str, float]) -> Number:
    """Slow but simple non-vectorised LCDM wrapper (kept for completeness)."""
    z_arr = _asarray(z)
    out = [LCDM_MODEL_vectorised(zi, p) for zi in z_arr]
    out = np.array(out, dtype=float)
    return out if out.size > 1 else float(out)


def f1CDM_MODEL_vectorised(
    z: Number,
    p: Dict[str, float],
    tol: float = 1e-8,
    maxiter: int = 60,
) -> Number:
    r"""
    f1CDM: E^2 = Ω_m (1+z)^3 + (1 - Ω_m) E^{2n}

    Parameters in `p`:
      • Omega_m
      • n          (f(T)-like exponent; constrained by `restrict_f1CDM_v`)
    """
    z = _asarray(z)
    Om = float(p["Omega_m"])
    n = float(p["n"])

    # Seed with ΛCDM
    E = np.sqrt(Om * (1 + z) ** 3 + (1 - Om))

    converged = np.zeros_like(E, dtype=bool)
    for _ in range(maxiter):
        f = E**2 - (Om * (1 + z) ** 3 + (1 - Om) * E ** (2.0 * n))
        df = 2.0 * E - (1.0 - Om) * (2.0 * n) * np.where(
            E > 0, E ** (2.0 * n - 1.0), np.inf
        )
        step = f / np.where(df == 0.0, np.inf, df)
        E -= step
        converged |= np.abs(step) < tol
        if np.all(np.abs(step) < tol):
            break

    # Fallback for any bad points
    bad = (~converged) | (~np.isfinite(E)) | (E <= 0)
    if np.any(bad):
        logging.getLogger(__name__).warning(
            "f1CDM: %d/%d points hit fallback (maxiter=%d).",
            int(bad.sum()),
            int(bad.size),
            maxiter,
        )

        def eq(Eval, zi):
            return Eval**2 - (Om * (1 + zi) ** 3 + (1 - Om) * Eval ** (2.0 * n))

        seeds = np.maximum(1.0, np.sqrt(Om * (1 + z[bad]) ** 3 + (1 - Om)))
        try:
            E[bad] = np.array(
                [fsolve(eq, x0=float(x0), args=(zi,))[0] for x0, zi in zip(seeds, z[bad])]
            )
        except Exception:
            E[bad] = np.nan

    E[~np.isfinite(E)] = np.nan
    E[E <= 0] = np.nan
    return E if E.size > 1 else float(E)


def f1CDM_MODEL_non_vectorised(z: Number, p: Dict[str, float]) -> Number:
    """Non-vectorised f1CDM wrapper (mainly for debugging / testing)."""
    z_arr = _asarray(z)
    out = [f1CDM_MODEL_vectorised(zi, p) for zi in z_arr]
    out = np.array(out, dtype=float)
    return out if out.size > 1 else float(out)


# ============================================================================
#  CMB wrappers (CLASS C_ℓ)
#  ---------------------------------------------------------------------------
#  These are used by the CMB likelihood. They take *full* cosmological
#  parameter dicts (not just Omega_m, n, etc.) and return raw C_ℓ arrays.
# ============================================================================

# Internal cache for CLASS; defined here so the CMB wrappers can share it.
_class_cache = None


# User_defined_modules.py
def LCDM_v_CMB(p: dict, mode: str = "hil"):
    """
    Canonical CMB helper for LCDM_v.

    mode:
      - "lowl" → cheap low-ℓ EE-only setup (no lensing, lmax ~ 30)
      - anything else → full high-ℓ, lensed spectra for Plik + (optionally) lensing

    Returns
    -------
    dict or None
      CLASS C_ell dict (raw_cl or lensed_cl) or None on failure.
    """
    # Make sure all background quantities are present
    # (Ω_m, Ω_b, Ω_bh^2, Ω_dh^2 given H_0)
    p = _ensure_background_params(p)

    m = (mode or "").lower()
    is_lowl = m.startswith("low")

    # Guardrail: bail out early if dark matter becomes negative / nonsensical
    Om = float(p.get("Omega_m", 0.0))
    Ob = float(p.get("Omega_b", 0.0))
    if Om <= 0 or Ob <= 0 or Ob >= Om:
        return None

    global _class_cache
    if _class_cache is None:
        _class_cache = classy.Class()  # <--- Now looks up the CURRENT binary
    cosmo = _class_cache
    
    # IMPORTANT:
    #   For low-ℓ SimAll EE we do NOT need lensing Cls at all.
    #   Requesting lCl in low-ℓ mode is unnecessary and can increase fragility.
    if is_lowl:
        output_str = "tCl,pCl"
        class_params = {
            "l_max_scalars": 31,
            "lensing": "no",
        }
    else:
        output_str = "tCl,pCl,lCl"
        class_params = {
            "l_max_scalars": 2509,  # enough for Planck high-ℓ
            "lensing": "yes",
        }

    base_params = {
        "output": output_str,
        "n_s": float(p["n_s"]),
        "h": float(p["H_0"]) / 100.0,
        "omega_b": float(p["Omega_bh^2"]),
        "omega_cdm": float(p["Omega_dh^2"]),
        "tau_reio": float(p["tau_reio"]),
        "A_s": float(np.exp(p["ln10^10_As"]) * 1e-10),
    }

    # Silence CLASS verbosity
    VERBOSE_OFF_SAFE = {
        "input_verbose": 0,
        "background_verbose": 0,
        "thermodynamics_verbose": 0,
        "perturbations_verbose": 0,
        "transfer_verbose": 0,
        "primordial_verbose": 0,
        "lensing_verbose": 0,
        "output_verbose": 0,
    }

    cosmo_params = {**base_params, **class_params, **VERBOSE_OFF_SAFE}

    try:
        # Hard-reset CLASS internal state between calls.
        # This is CRITICAL when mixing low-ℓ and high-ℓ likelihoods in one run.
        try:
            cosmo.struct_cleanup()
        except Exception:
            pass
        try:
            cosmo.empty()
        except Exception:
            pass

        cosmo.set(cosmo_params)
        cosmo.compute()

        # Low-ℓ: raw_cl is fine and cheaper.
        # High-ℓ: use lensed_cl for Plik / TT-only.
        return cosmo.raw_cl() if is_lowl else cosmo.lensed_cl()

    except classy.CosmoComputationError as e:
        logger.error("CLASS CosmoComputationError: %s | cosmo_params=%s", e, cosmo_params)
        return None
    except Exception as e:
        logger.exception("Unexpected error in LCDM_v_CMB: %s | cosmo_params=%s", e, cosmo_params)
        return None


def f1CDM_v_CMB(p: dict, mode: str = "hil"):
    """
    Simplified CMB helper for f1CDM_v. 
    Fixes the 'unread parameter' error by mapping n -> n_fT directly.
    """
    import classy
    p = _ensure_background_params(p)
    
    # Mode setup
    m = (mode or "").lower()
    is_lowl = m.startswith("low")
    
    global _class_cache
    if _class_cache is None:
        _class_cache = classy.Class()
    cosmo = _class_cache

    # 1. BUILD THE PARAMS MANUALLY
    fT_value = p.get("n", p.get("n_fT", 0.0))

    class_params = {
        "output": "tCl,pCl,lCl" if not is_lowl else "tCl,pCl",
        "l_max_scalars": 2509 if not is_lowl else 31,
        "lensing": "yes" if not is_lowl else "no",
        "n_s": float(p["n_s"]),
        "h": float(p["H_0"]) / 100.0,
        "omega_b": float(p["Omega_bh^2"]),
        "omega_cdm": float(p["Omega_dh^2"]),
        "tau_reio": float(p["tau_reio"]),
        "A_s": float(np.exp(p["ln10^10_As"]) * 1e-10),
        "n_fT": float(fT_value), 
    }

    # 2. ADD VERBOSITY SILENCERS
    class_params.update({
        "input_verbose": 0, "background_verbose": 0, "thermodynamics_verbose": 0,
        "perturbations_verbose": 0, "transfer_verbose": 0, "primordial_verbose": 0,
        "lensing_verbose": 0, "output_verbose": 0,
    })

    try:
        cosmo.struct_cleanup()
        cosmo.empty()
        
        # 3. SET AND COMPUTE
        cosmo.set(class_params)
        cosmo.compute()
        
        return cosmo.raw_cl() if is_lowl else cosmo.lensed_cl()
    except Exception as e:
        # If it still fails, we want to know why in the terminal
        # print(f"DEBUG: CLASS failed with params {class_params}. Error: {e}")
        return None

# ============================================================================
#  Model registry / discovery
#  ---------------------------------------------------------------------------
#  This is the single source of truth that maps a string name → (func, params)
#  and is used everywhere (MCMC, plotting, likelihoods).
#
#  To add a new model:
#    1. Implement `MyMG_MODEL_vectorised(z, p)` above.
#    2. Add an entry here:
#
#       _MODEL_REGISTRY["MyMG_v"] = (MyMG_MODEL_vectorised,
#                                    ["Omega_m", "my_param"])
#
#    3. Optionally add a restriction in the restrictions_map above.
# ============================================================================

_MODEL_REGISTRY: Dict[str, Tuple[Callable, List[str]]] = {
    # Background-only models
    "LCDM_v":   (LCDM_MODEL_vectorised,      ["Omega_m"]),
    "LCDM_nv":  (LCDM_MODEL_non_vectorised,  ["Omega_m"]),
    "f1CDM_v":  (f1CDM_MODEL_vectorised,     ["Omega_m", "n"]),
    "f1CDM_nv": (f1CDM_MODEL_non_vectorised, ["Omega_m", "n"]),

    # CMB-specific models for CLASS Cls (used by CMB likelihoods)
    "LCDM_v_CMB": (
        LCDM_v_CMB,
        ["Omega_m", "Omega_b", "H_0", "n_s", "tau_reio", "ln10^10_As"],
    ),
    "f1CDM_v_CMB": (
        f1CDM_v_CMB,
        ["Omega_m", "Omega_b", "H_0", "n_s", "tau_reio", "ln10^10_As", "n"],
    ),
    # Example for a new MG model:
    # "MyMG_v": (MyMG_MODEL_vectorised, ["Omega_m", "my_param"]),
}


def register_model(name: str, func: Callable, parameters: List[str]) -> None:
    """
    Register a new model at runtime.

    Typical usage (after defining MyMG_MODEL_vectorised):

        register_model("MyMG_v", MyMG_MODEL_vectorised, ["Omega_m", "my_param"])

    This simply updates the internal `_MODEL_REGISTRY`.
    """
    if not callable(func):
        raise TypeError("func must be callable")
    _MODEL_REGISTRY[name] = (func, list(parameters))


def Get_model_function(model_name: str) -> Callable[[Number, Dict[str, float]], Number]:
    """
    Look up the model function by name.

    This is what the MCMC / likelihood layer calls to get E(z).
    """
    try:
        return _MODEL_REGISTRY[model_name][0]
    except KeyError:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )


def Get_model_names(model_name: Union[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Return a mapping model → { 'parameters': [ ... ] } for one or more names.

    This is mainly used for UI / logging / consistency checks.
    """
    names = [model_name] if isinstance(model_name, str) else list(model_name)
    out: Dict[str, Dict[str, List[str]]] = {}
    for nm in names:
        if nm in _MODEL_REGISTRY:
            out[nm] = {"parameters": _MODEL_REGISTRY[nm][1]}
    return out
    
    


# ============================================================================
#  Model restrictions (simple param-level guard rails)
#  ---------------------------------------------------------------------------
#  These are optional "prior" predicates used to avoid crazy / unstable
#  regions in parameter space. They are consumed via Get_model_restrictions.
# ============================================================================


def restrict_LCDM_Omega_m(x: float) -> bool:
    """Keep Ω_m in a physically sensible range; very loose lower bound."""
    return x > 0.0002


def restrict_f1CDM_v(x: float) -> bool:
    """
    Guard-rail prior for f1CDM exponent n.

    Empirically, n < 0.5 keeps E(z) well-behaved and the root-finding stable.
    """
    return x < 0.5


# Global map that Get_model_restrictions reads from.
restrictions_map: Dict[str, Dict[str, Callable[[float], bool]]] = {
    "LCDM":    {"Omega_m": restrict_LCDM_Omega_m},
    "LCDM_v":  {"Omega_m": restrict_LCDM_Omega_m},
    "LCDM_nv": {"Omega_m": restrict_LCDM_Omega_m},
    "f1CDM_v": {"n": restrict_f1CDM_v},
    # Example for a new model:
    # "MyMG_v": {"my_param": restrict_MyMG_param},
}


def Get_model_restrictions(
    model_name: Union[str, List[str]]
) -> Union[Dict[str, Callable[[float], bool]], Dict[str, Dict[str, Callable[[float], bool]]]]:
    """
    Return param->predicate restrictions for given model(s).

    If a list is given, return a mapping model->param->predicate.
    All callables are top-level functions to allow MPI pickling.

    Users can extend this by adding entries to `restrictions_map`.
    """
    if isinstance(model_name, list):
        return {m: restrictions_map.get(m, {}) for m in model_name}
    return restrictions_map.get(model_name, {})


# ============================================================================
#  Common wrappers used across likelihoods / BAO / growth
#  ---------------------------------------------------------------------------
#  These provide a stable, central API for distances and growth-related
#  arrays that both Statistical_packages and Plot_functions re-use.
# ============================================================================


def Hubble(p: Dict[str, float]) -> float:
    """Hubble distance, in Mpc (c / H0)."""
    return C_KM_S / float(p["H_0"])


def Comoving_distance_vectorized(MODEL_func: Callable, redshifts: Number, p: Dict[str, float]) -> Number:
    """
    Safe wrapper around the vectorised comoving distance, with background
    parameter injection for CMB-calibrated chains.

    Signature is shared with Statistical_packages.Comoving_distance_vectorized.
    """
    p = _ensure_background_params(p)
    return _vect_comove(MODEL_func, redshifts, p)


def matter_density_z_array(z: Number, param_dict: Dict[str, float], MODEL_func: Callable) -> Number:
    """
    Vector Ω_m(z); signature matches SP.matter_density_z_array(z, p, model).

    Exposed here so that plotting and likelihood code can share one
    implementation, and users don't have to worry about it.
    """
    return _omega_m_z_arr(z, param_dict, MODEL_func)


def integral_term_array(
    z: Number, param_dict: Dict[str, float], MODEL_func: Callable, gamma: float
) -> Number:
    """
    Vector growth integral ∫ Ω_m(z')^γ / (1+z') dz'.

    This is used both in the fσ8 likelihood and in the plotting code
    (compute_sigma8z), so we expose a single shared wrapper here.
    """
    return _integral_term_arr(z, param_dict, MODEL_func, float(gamma))
