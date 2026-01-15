#!/usr/bin/env python3
"""
Vectorized statistical and likelihood routines for Kosmulator.

This module is the backend "numerics layer" for Kosmulator. It provides:

  * CMB theory helpers (CLASS / user-defined models) and per-process caches.
  * Planck high-ℓ / low-ℓ / lensing likelihood wrappers (TT, TTTEEE, EE, φφ).
  * Generic chi-squared helpers for CC/OHD, Pantheon+, BAO, DESI-like BAO.
  * BBN D/H prediction and χ² (approximate, live AlterBBN, and grid-based).
  * BAO geometry helpers: D_M / r_d, D_H / r_d, D_V / r_d, D_A / r_d.

Only the high-level “front end” code (MCMC_setup, Post_processing, etc.) should
call this module; end users normally interact via the CLI and config files.
"""

from __future__ import annotations

import logging
from numbers import Number
from typing import Callable, Dict, Any, Optional
import inspect

import numpy as np
import scipy.linalg as la
from scipy.interpolate import PchipInterpolator

from Kosmulator_main.constants import (
    C_KM_S,
    T_CMB_DEFAULT,
    N_EFF_DEFAULT,
    TAU_N_DEFAULT,
    R_D_SINGLETON,
    OBSERVATIONS_BASE,
    BBN_GRID_RELATIVE,
    PLANCK_NUISANCE_DEFAULTS,
)

from . import Class_run as CR
from Kosmulator_main import utils as U

try:
    import clik  # noqa: F401  # imported for side effects in some environments
except Exception:
    clik = None

# User-defined cosmology models (background + Cℓ wrappers)
try:
    import User_defined_modules as UDM  # type: ignore
except Exception:
    import os
    import sys

    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    import User_defined_modules as UDM  # type: ignore

from functools import lru_cache

from .rd_helpers import (
    _try_compute_rd,
    _resolve_rd,
    _try_resolve_rd,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Global CMB likelihood handles / caches
# -----------------------------------------------------------------------------

_hil_like = None        # Planck high-ℓ TTTEEE
_hilTT_like = None      # Planck high-ℓ TT-only
_lensing_like = None    # Planck lensing (raw / cmbmarged) – accessed via helper
_lowl_like = None       # Planck low-ℓ EE

_T_CMB_uK = T_CMB_DEFAULT * 1.0e6    # K → µK
TCMB2 = _T_CMB_uK ** 2               # (µK)^2

# Simple per-process cache for theory Cℓs
_CMB_THEORY_CACHE: Dict[str, Any] = {"key": None, "cls": None}
_classy_cache_sp = None             # reusable low-ℓ CLASS instance

# -----------------------------------------------------------------------------
# CLASS / theory helpers
# -----------------------------------------------------------------------------

def _compute_cls_cached(
    pd: Dict[str, float],
    Lmax: Optional[int] = None,
    mode: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Return raw Cℓs for the parameter dictionary `pd`.

    Returns a dict with keys:
      'tt', 'ee', 'te', 'pp' (and optionally 'bb').

    Parameters
    ----------
    pd : dict containing either:
      (H_0, Omega_m, Omega_b, ...)  OR  (H_0, Omega_bh^2, Omega_dh^2, ...)
    Lmax : int, optional
        Maximum ℓ requested. If None, uses:
          * 31  for low-ℓ mode ("lowl")
          * 2509 otherwise (Planck high-ℓ)
        Clamped to [8, 4096].
    mode : {"lowl", ...}, optional
        If 'lowl', use a low-ℓ setup; anything else → high-ℓ.
    model_name : str, optional
        Name of the user model (e.g. "LCDM_v", "f1CDM_v"). If None, tries to
        infer from CR._current_class_model.

    Notes
    -----
    1. We first try model-specific helpers in `User_defined_modules`:
         LCDM_v_CMB, f1CDM_v_CMB, ...
       where you can hard-wire fast Cℓ pipelines for specific models.

    2. If no model-specific helper is available for this regime, we fall back
       to driving CLASS directly with a minimal lensing-friendly setup.
    """
    from Kosmulator_main.utils import ensure_background_params
    pd = ensure_background_params(pd)
    m = (mode or "").lower()
    ell_max_default = 31 if m.startswith("low") else 2509
    ell_max = int(Lmax) if (Lmax is not None) else ell_max_default
    ell_max = max(8, min(int(ell_max), 4096))

    # Try to infer model name from the CLASS runtime if not supplied
    model_name = model_name or getattr(CR, "_current_class_model", "LCDM_v")

    # Build a stable cache key from numeric pieces of pd
    try:
        numeric_items = tuple(
            sorted((k, float(v)) for k, v in pd.items() if isinstance(v, (int, float)))
        )
    except Exception:
        numeric_items = tuple(
            sorted((k, v) for k, v in pd.items() if isinstance(v, (int, float)))
        )

    key = (model_name, ell_max, m, numeric_items)

    if _CMB_THEORY_CACHE.get("key") == key:
        return _CMB_THEORY_CACHE.get("cls")

    cl: Optional[Dict[str, np.ndarray]] = None

    cmb_fn = getattr(UDM, f"{model_name}_CMB", None)
    if cmb_fn is not None:
        try:
            sig = inspect.signature(cmb_fn)
            if "mode" in sig.parameters:
                cl = cmb_fn(pd, mode=mode)
            else:
                cl = cmb_fn(pd)
        except Exception as e:
            logger.exception("Model-specific CMB helper failed (%s_CMB): %s", model_name, e)
            cl = None  # continue to generic CLASS fallback
    else:
        logger.warning("No model-specific CMB helper (%s_CMB). Using generic CLASS fallback.", model_name)

    # 2) Generic fallback: drive CLASS directly
    if cl is None:
        try:
            from classy import Class, CosmoComputationError  # local import

            lensing_on = not m.startswith("low")
            global _classy_cache_sp

            if (not lensing_on) and (_classy_cache_sp is not None):
                # Re-use pre-allocated low-ℓ instance
                cosmo = _classy_cache_sp
                for _fn in ("struct_cleanup", "empty"):
                    try:
                        getattr(cosmo, _fn)()
                    except Exception:
                        pass
            else:
                cosmo = Class()

            # Amplitude mapping
            if "ln10^10_As" in pd:
                A_s = float(np.exp(pd["ln10^10_As"]) * 1e-10)
            else:
                A_s = float(pd.get("A_s", 2.1e-9))

            pars = {
                "output": "tCl,pCl,lCl",
                "modes": "s",
                "lensing": "yes" if lensing_on else "no",
                "l_max_scalars": int(ell_max),
                "k_max_tau0_over_l_max": 5.0,
                "h": float(pd["H_0"]) / 100.0,
                "omega_b": float(pd["Omega_bh^2"]),
                "omega_cdm": float(pd["Omega_dh^2"]),
                "n_s": float(pd["n_s"]),
                "tau_reio": float(pd["tau_reio"]),
                "A_s": A_s,
                "Omega_k": 0.0,
                "N_ur": float(pd.get("N_eff", N_EFF_DEFAULT)),
                "N_ncdm": 0,
                "non linear": "none",
            }

            # Optional model-specific extensions (e.g. f(T) parameters)
            if "n" in pd:
                pars["n_fT"] = float(pd["n"])

            cosmo.set(pars)
            cosmo.compute()

            raw = cosmo.raw_cl()
            cl = {
                "tt": np.asarray(raw.get("tt", []), dtype=float),
                "ee": np.asarray(raw.get("ee", []), dtype=float),
                "te": np.asarray(raw.get("te", []), dtype=float),
                "pp": np.asarray(raw.get("pp", []), dtype=float),
            }
            if "bb" in raw:
                cl["bb"] = np.asarray(raw.get("bb", []), dtype=float)

            if lensing_on:
                try:
                    cosmo.struct_cleanup()
                    cosmo.empty()
                except Exception:
                    pass
            else:
                _classy_cache_sp = cosmo

        except Exception as e:
            logger.error("CLASS failed (lensing=%s, L=%s): %s", not m.startswith("low"), ell_max, e)
            cl = None

    _CMB_THEORY_CACHE["key"] = key
    _CMB_THEORY_CACHE["cls"] = cl
    return cl


# -----------------------------------------------------------------------------
# Planck clik handle helpers
# -----------------------------------------------------------------------------

def _get_Class():
    """Thin wrapper to keep the old import style alive."""
    import Class_run as CR_local
    return CR_local.get_Class()


def _get_hil_like():
    """High-ℓ TTTEEE clik handle (cached)."""
    global _hil_like
    if _hil_like is None:
        with U.quiet_cstdio():
            _hil_like = CR.get_clik_hil()
    return _hil_like


def _get_hilTT_like():
    """High-ℓ TT-only clik handle (cached)."""
    global _hilTT_like
    if _hilTT_like is None:
        _hilTT_like = CR.get_clik_hilTT()
    return _hilTT_like


_LENSING_MODE = "raw"  # "raw" (non-marged) or "cmbmarged"


def set_lensing_mode(mode: str) -> None:
    """
    Select which Planck lensing likelihood handle to use.

    Parameters
    ----------
    mode : str
        "raw"      → Planck lensing non-marginalised (default).
        "cmbmarg"  → CMB-marginalised version.
    """
    global _LENSING_MODE
    _LENSING_MODE = "cmbmarged" if str(mode).lower().startswith("cmb") else "raw"


def _get_lensing_like():
    """Return the Planck lensing clik handle consistent with _LENSING_MODE."""
    if _LENSING_MODE == "cmbmarged":
        return CR.get_clik_lensing_cmbmarg()
    return CR.get_clik_lensing()


def _get_lowl_like():
    """Low-ℓ EE clik handle (cached)."""
    global _lowl_like
    if _lowl_like is None:
        with U.quiet_cstdio():
            _lowl_like = CR.get_clik_lowl()
    return _lowl_like


# -----------------------------------------------------------------------------
# CMB theory access from user-defined models
# -----------------------------------------------------------------------------

from typing import Optional, Dict
import numpy as np

def _get_cls_from_model(
    pd: Dict[str, float],
    model_name: str,
    mode: str = "hil",
) -> Optional[Dict[str, np.ndarray]]:
    """
    Obtain theory Cℓ dict from the user model.

    Returns:
      - dict with keys like 'tt','ee','te','pp' on success
      - None on failure (CLASS failure / invalid params / user model returned None)
    """
    model_func = getattr(UDM, model_name + "_CMB", None)
    if model_func is None:
        logger.error("No CMB helper found for model '%s' (expected %s_CMB).", model_name, model_name)
        return None

    try:
        # Try new-style signature (p, mode)
        return model_func(pd, mode=mode)
    except TypeError:
        # Old-style signature (p)
        try:
            return model_func(pd)
        except Exception as e:
            logger.exception("Model CMB helper failed (old-style) for %s: %s", model_name, e)
            return None
    except Exception as e:
        logger.exception("Model CMB helper failed for %s: %s", model_name, e)
        return None



# -----------------------------------------------------------------------------
# Planck high-ℓ TTTEEE likelihood
# -----------------------------------------------------------------------------

def cmb_hil_loglike(pd: Dict[str, float], model_name: str, floor: float = -1e10) -> float:
    """
    Planck high-ℓ TTTEEE Plik likelihood.

    Packs a vector [TT, EE, TE, nuisances] in µK^2, with each spectrum
    from ℓ = 0..ℓ_max in the clik ordering TT, EE, BB, TE, TB, EB.
    """
    try:
        CR.ensure_class_ready(model_name)
    except Exception as e:
        logger.warning("ensure_class_ready(%s) failed (%s); proceeding anyway", model_name, e)

    like = _get_hil_like()

    # 1) lmax in TT, EE, BB, TE, TB, EB order (per clik docs)
    raw_lmax = like.get_lmax() if hasattr(like, "get_lmax") else (2508, 2508, -1, 2508, -1, -1)
    logger.debug("[cmb_hil] get_lmax() = %r", raw_lmax)

    if isinstance(raw_lmax, dict):
        # be robust to lower/upper case
        lTT = int(raw_lmax.get("TT") or raw_lmax.get("tt") or -1)
        lEE = int(raw_lmax.get("EE") or raw_lmax.get("ee") or -1)
        lTE = int(raw_lmax.get("TE") or raw_lmax.get("te") or -1)
    else:
        seq = list(raw_lmax)
        while len(seq) < 6:
            seq.append(-1)
        # clik ordering: TT, EE, BB, TE, TB, EB
        lTT = int(seq[0])
        lEE = int(seq[1])
        lTE = int(seq[3])

    def _n_ell(L: int) -> int:
        return 0 if (L is None or L < 0) else (L + 1)

    nTT = _n_ell(lTT)
    nEE = _n_ell(lEE)
    nTE = _n_ell(lTE)

    # 2) Theory Cℓ from the user model, as for TT-only
    try:
        cl = _get_cls_from_model(pd, model_name, mode="hil")
    except Exception as e:
        logger.error("cmb_hil_loglike: _get_cls_from_model failed: %s", e)
        return float(floor)

    def _block(name: str, n: int) -> Optional[np.ndarray]:
        """Return C_ℓ(name) for ℓ=0..n-1 in µK^2, contiguous float64, or None if n==0."""
        if n <= 0:
            return None
        arr = np.asarray(cl[name], float)
        if arr.size < n:
            arr = np.pad(arr, (0, n - arr.size), mode="constant")
        else:
            arr = arr[:n]
        return np.ascontiguousarray(arr * TCMB2, dtype=np.float64)

    tt = _block("tt", nTT)
    ee = _block("ee", nEE)
    te = _block("te", nTE)

    blocks = [b for b in (tt, ee, te) if b is not None]

    # 3) Nuisance parameters in clik order (same convention as TT-only, extended for pol)
    try:
        raw_names = like.get_extra_parameter_names()
        nuis_names = [
            n.decode() if isinstance(n, (bytes, bytearray)) else n
            for n in raw_names
        ]
    except Exception:
        nuis_names = []

    calib_ones = {
        "A_planck",
        "calib_100T", "calib_143T", "calib_217T",
        "calib_100P", "calib_143P", "calib_217P",
    }

    def _nuis_default(name: str) -> float:
        return 1.0 if name in calib_ones else 0.0

    nuis = (
        np.array([pd.get(n, _nuis_default(n)) for n in nuis_names], dtype=np.float64)
        if nuis_names
        else np.empty(0, np.float64)
    )

    # 4) Pack vector and sanity-check length
    vec = np.ascontiguousarray(np.concatenate(blocks + [nuis]), np.float64)

    expected = None
    if hasattr(like, "get_lkl_length"):
        try:
            expected = like.get_lkl_length()
        except Exception:
            expected = None

    logger.debug(
        "[cmb_hil] nTT=%d nEE=%d nTE=%d len(nuis)=%d total=%d (expected=%s)",
        0 if tt is None else tt.size,
        0 if ee is None else ee.size,
        0 if te is None else te.size,
        nuis.size,
        vec.size,
        str(expected),
    )

    if expected is not None and vec.size != expected:
        logger.error(
            "cmb_hil_loglike: vector length %d != expected %d (lTT=%d, lEE=%d, lTE=%d, nuis=%d)",
            vec.size, expected, lTT, lEE, lTE, nuis.size
        )
        return float(floor)

    # 5) Evaluate likelihood
    try:
        with U.quiet_cstdio():
            out = like(vec)

        # clik sometimes returns array([-198.0]) instead of scalar
        if isinstance(out, np.ndarray):
            if out.ndim == 0:
                val = float(out.item())
            else:
                val = float(out.ravel()[0])
        else:
            val = float(out)

        logger.debug("[cmb_hil] like=%.3e", val)
        return val
    except Exception as e:
        logger.error("cmb_hil_loglike: clik evaluation failed: %s", e)
        return float(floor)


# -----------------------------------------------------------------------------
# Planck high-ℓ TT-only likelihood
# -----------------------------------------------------------------------------

def cmb_hilTT_loglike(pd: Dict[str, float], model_name: str) -> float:
    like = _get_hilTT_like()

    # 1) lmax_TT from clik
    raw_lmax = like.get_lmax() if hasattr(like, "get_lmax") else 2508
    if isinstance(raw_lmax, dict):
        lmax = raw_lmax.get("tt") or raw_lmax.get("TT") or next(iter(raw_lmax.values()))
    elif isinstance(raw_lmax, (tuple, list, np.ndarray)):
        lmax = raw_lmax[0]
    else:
        lmax = raw_lmax
    lmax = int(lmax)
    need = lmax + 1  # ℓ = 0..lmax

    # 2) Theory TT in µK^2
    try:
        cl = _get_cls_from_model(pd, model_name, mode="hil")
    except Exception as e:
        logger.warning("cmb_hilTT_loglike: theory failed (%s). Returning sentinel.", e)
        return -1e10

    if cl is None or "tt" not in cl:
        return -1e10

    tt = np.asarray(cl["tt"], float)
    if tt.size < need:
        tt = np.pad(tt, (0, need - tt.size), mode="constant")
    else:
        tt = tt[:need]
    tt = np.ascontiguousarray(tt * TCMB2, dtype=np.float64)

    # 3) Nuisances in clik order
    nuis_raw = like.get_extra_parameter_names()
    nuis_names = [
        n.decode() if isinstance(n, (bytes, bytearray)) else n
        for n in nuis_raw
    ]

    def _nuis_default(name: str) -> float:
        return 1.0 if name in ("A_planck", "calib_100T", "calib_143T", "calib_217T") else 0.0

    nuis = np.array([pd.get(n, _nuis_default(n)) for n in nuis_names], dtype=np.float64)

    vec = np.ascontiguousarray(np.concatenate([tt, nuis]), np.float64)

    # 4) Optional sanity check vs clik’s expected length
    if hasattr(like, "get_lkl_length"):
        try:
            expected = like.get_lkl_length()
            if vec.size != expected:
                logger.error(
                    "cmb_hilTT_loglike: vec length %d != expected %d (lmax=%d, nuis=%d)",
                    vec.size, expected, lmax, nuis.size
                )
                return -1e10
        except Exception:
            pass

    # 5) Evaluate
    try:
        with U.quiet_cstdio():
            out = like(vec)

        # clik sometimes returns array([val]) instead of scalar
        if isinstance(out, np.ndarray):
            if out.ndim == 0:
                val = float(out.item())
            else:
                val = float(out.ravel()[0])
        else:
            val = float(out)

        logger.debug("[cmb_hilTT] lmax=%d len(vec)=%d like=%.3e", lmax, vec.size, val)
        return val
    except Exception as e:
        logger.error("cmb_hilTT_loglike: clik evaluation failed: %s", e)
        return -1e10


# -----------------------------------------------------------------------------
# Planck lensing likelihood (RAW / CMB-marged)
# -----------------------------------------------------------------------------

def cmb_lensing_loglike(pd: Dict[str, float], model_name: str) -> float:
    """
    Planck 2018 lensing likelihood.

    RAW (non-marginalized) expects:
      [pp, (tt), (ee), (bb), (te), (tb), (eb), nuisances]

    CMB-marged expects:
      [pp, nuisances]

    where φφ = C_L^{φφ}, and all temperature spectra are in µK^2.
    """
    # 0. Make sure CLASS is ready for this model
    try:
        CR.ensure_class_ready(model_name)
    except Exception as e:
        logger.warning("ensure_class_ready(%s) failed (%s); proceeding anyway", model_name, e)

    # 1. Choose which clik handle we use, based on global _LENSING_MODE
    like = _get_lensing_like()
    mode_now = "raw" if (_LENSING_MODE == "raw") else "cmbmarged"

    # Channel ordering used by clik for lmax, etc.
    canon_order = ["pp", "tt", "ee", "bb", "te", "tb", "eb"]

    # ------------------------------------------------------------------
    # Helper: read lmax info from clik
    # ------------------------------------------------------------------
    def _read_lmax_map(default_pp: int) -> Dict[str, int]:
        try:
            raw = like.get_lmax()
        except Exception:
            # Fall back to something sensible if clik has no lmax
            return {"pp": default_pp}

        try:
            arr = list(raw)
        except TypeError:
            arr = [raw]

        out: Dict[str, int] = {}
        for i, ch in enumerate(canon_order):
            if i < len(arr):
                try:
                    out[ch] = int(arr[i])
                except Exception:
                    pass

        if "pp" not in out:
            out["pp"] = default_pp
        return out

    # ------------------------------------------------------------------
    # Helper: decide a reasonable target L for φφ from clik metadata
    # ------------------------------------------------------------------
    def _target_L_for_lensing(like_handle, mode: str) -> int:
        """
        Decide a reasonable target L for φφ based on clik metadata.
        """
        try:
            # Preferred: explicit lensing bins / nbins if provided
            if hasattr(like_handle, "get_lensing_bins"):
                bins = like_handle.get_lensing_bins()
                Lmax_bin = max(
                    int(b[1]) if isinstance(b, (list, tuple)) and len(b) >= 2 else int(b)
                    for b in bins
                )
                return max(8, min(Lmax_bin, 4096))
            if hasattr(like_handle, "get_lensing_nbins"):
                # Planck often uses ~400 bins for lensing
                return 400
        except Exception:
            pass

        # Fallback: use lmax for the first spectrum, with a guardrail
        try:
            lm = like_handle.get_lmax()
            L0 = int(lm[0]) if isinstance(lm, (list, tuple, np.ndarray)) else int(lm)
        except Exception:
            L0 = 400

        # For CMB-marged, be conservative at high L
        if mode == "cmbmarged" and L0 > 800:
            return 400
        return max(8, min(L0, 4096))

    # ------------------------------------------------------------------
    # 2. Decide target L and lmax per channel
    # ------------------------------------------------------------------
    Ltarget = _target_L_for_lensing(like, mode_now)
    lmax = _read_lmax_map(Ltarget)
    Lpp_req = max(8, int(lmax.get("pp", Ltarget)))

    # ------------------------------------------------------------------
    # 3. Compute theory Cℓ up to at least φφ(Lpp_req)
    # ------------------------------------------------------------------
    cl = _compute_cls_cached(pd, Lmax=Lpp_req, mode="lensing")
    if (cl is None) or ("pp" not in cl):
        logger.warning("cmb_lensing_loglike: theory Cl computation failed; returning sentinel")
        return -1e10

    def _avail_L(name: str) -> int:
        arr = cl.get(name, None)
        return (len(arr) - 1) if isinstance(arr, (list, tuple, np.ndarray)) and len(arr) else -1

    Lpp_av = _avail_L("pp")
    if Lpp_av < 2:
        logger.warning("cmb_lensing_loglike: missing/empty 'pp'; returning sentinel")
        return -1e10

    # Clip to what CLASS actually provided
    if Lpp_req > Lpp_av:
        if not getattr(cmb_lensing_loglike, "_warned_pp_clip", False):
            logger.warning(
                "cmb_lensing_loglike: requested Lpp=%d but only have %d; clipping",
                Lpp_req,
                Lpp_av,
            )
            setattr(cmb_lensing_loglike, "_warned_pp_clip", True)
        Lpp_req = Lpp_av
        lmax["pp"] = Lpp_req

    # ------------------------------------------------------------------
    # 4. Build the Cl vector in the order expected by clik
    # ------------------------------------------------------------------
    μK2 = TCMB2  # Convert from K^2 to μK^2 when needed

    def _take(name: str, L_req: int, scale_μK2: bool):
        if L_req < 0:
            return None
        arr = cl.get(name, None)
        if arr is None:
            return None
        L_av = len(arr) - 1
        if L_av < 0:
            return None
        L_eff = min(L_req, L_av)
        out = np.asarray(arr, float)[: L_eff + 1]
        if scale_μK2:
            out = out * μK2
        return np.ascontiguousarray(out, np.float64)

    # φφ part
    cl_pp = _take("pp", Lpp_req, scale_μK2=False)
    if cl_pp is None or cl_pp.size == 0:
        return -1e10

    if mode_now == "raw":
        # RAW lensing: φφ + TT/EE/BB/TE/TB/EB as available
        blocks = [cl_pp]
        for key in ["tt", "ee", "bb", "te", "tb", "eb"]:
            arr = _take(key, int(lmax.get(key, -1)), scale_μK2=True)
            if arr is not None:
                blocks.append(arr)
        base = blocks
    else:
        # CMB-marged: only φφ enters
        base = [cl_pp]

    # ------------------------------------------------------------------
    # 5. Nuisance parameters from clik
    # ------------------------------------------------------------------
    try:
        raw_names = like.get_extra_parameter_names()
        nuis_names = [
            n.decode() if isinstance(n, (bytes, bytearray)) else n
            for n in raw_names
        ]
    except Exception:
        nuis_names = []

    nuis: list[float] = []
    for name in nuis_names:
        if name in pd:
            # If the user/engine provided this nuisance explicitly, use it
            val = float(pd[name])
        else:
            # Fall back to the PLANCK_NUISANCE_DEFAULTS mean (first element of tuple),
            # or 0.0 if the nuisance is not in that dictionary.
            default = PLANCK_NUISANCE_DEFAULTS.get(name, (0.0, (None, None)))[0]
            val = float(default)
        nuis.append(val)

    nuis = np.asarray(nuis, np.float64)

    if nuis.size:
        vec = np.concatenate([np.concatenate(base), nuis])
    else:
        vec = np.concatenate(base)
    vec = np.ascontiguousarray(vec, np.float64).ravel()

    # ------------------------------------------------------------------
    # 6. Evaluate clik safely (and keep C stdout/stderr quiet)
    # ------------------------------------------------------------------
    try:
        with U.quiet_cstdio():
            out = like(vec)

        # clik sometimes returns array([val]) instead of scalar
        if isinstance(out, np.ndarray):
            if out.ndim == 0:
                val = float(out.item())
            else:
                val = float(out.ravel()[0])
        else:
            val = float(out)

        return val
    except Exception as e:
        logger.warning(
            "[cmb_lensing_loglike FAIL] mode=%s Lpp=%d vec_len=%d nuis=%s "
            "err=%s → returning -1e10",
            mode_now,
            int(Lpp_req),
            int(vec.size),
            ",".join(nuis_names) if nuis_names else "-",
            e,
        )
        return -1e10


# -----------------------------------------------------------------------------
# Planck low-ℓ EE likelihood
# -----------------------------------------------------------------------------

def cmb_lowl_loglike(pd: Dict[str, float], model_name: str) -> float:
    """
    Planck low-ℓ EE likelihood (SimAll).
    Returns log-likelihood (float). Bad points are penalised with -1e10.
    """
    try:
        CR.ensure_class_ready(model_name)
    except Exception as e:
        logger.warning(
            "ensure_class_ready(%s) failed (%s); proceeding anyway",
            model_name, e,
        )

    # Ask for a low-ℓ CMB setup
    try:
        cl = _get_cls_from_model(pd, model_name, mode="lowl")
    except Exception:
        return float(-1e10)

    # ---- CRITICAL GUARD ----
    # If CLASS failed (or the model-specific helper returned None),
    # do NOT crash; just penalise.
    if cl is None or (not isinstance(cl, dict)) or ("ee" not in cl):
        return float(-1e10)

    like = _get_lowl_like()

    # Decode nuisance names (clik often returns bytes)
    try:
        raw_names = like.get_extra_parameter_names()
        nuis_names = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in raw_names]
    except Exception:
        nuis_names = []

    # Default nuisances: use PLANCK_NUISANCE_DEFAULTS mean if available, else 0.0
    # (low-ℓ SimAll usually has none, but keep this generic and robust)
    nuis = np.array(
        [pd.get(n, PLANCK_NUISANCE_DEFAULTS.get(n, (0.0, (None, None)))[0]) for n in nuis_names],
        dtype=np.float64
    )

    # EE block: need length 30 (ℓ=0..29) for this likelihood
    ee = np.asarray(cl.get("ee", []), dtype=float)
    need = 30
    if ee.size < need:
        ee = np.pad(ee, (0, need - ee.size), mode="constant")
    else:
        ee = ee[:need]

    cl_ee = np.ascontiguousarray(ee * TCMB2, dtype=np.float64)
    v = np.ascontiguousarray(np.concatenate([cl_ee, nuis]), dtype=np.float64)
    #logger.warning("cl_ee: type=%s dtype=%s shape=%s", type(cl_ee), getattr(cl_ee, "dtype", None), getattr(cl_ee, "shape", None))
    #logger.warning("nuis:  type=%s dtype=%s shape=%s", type(nuis), getattr(nuis, "dtype", None), getattr(nuis, "shape", None))
    #logger.warning("v:     type=%s dtype=%s shape=%s", type(v), getattr(v, "dtype", None), getattr(v, "shape", None))

    try:
        out = like(v)  # call clik first
        #logger.warning("clik out: type=%s repr=%r", type(out), out)
    except Exception as e:
        logger.error("cmb_lowl_loglike: like(v) threw: %s", e)
        return -np.inf

    # Now normalize the output to a python float safely
    try:
        out_arr = np.asarray(out)
        if out_arr.ndim == 0:
            return float(out_arr.item())
        # common case: array([x]) -> take first element
        return float(out_arr.reshape(-1)[0])
    except Exception as e:
        logger.error("cmb_lowl_loglike: could not convert clik output to float. out=%r type=%s err=%s",
                     out, type(out), e)
        return -np.inf




# -----------------------------------------------------------------------------
# Generic background / χ² helpers (CC, Pantheon+, etc.)
# -----------------------------------------------------------------------------

def _ensure_background_params(p: Any) -> Dict[str, float]:
    """
    Ensure we have a parameter dict for background-only functions *and*
    inject/derive any missing background parameters (Omega_m, Omega_b, etc.)
    from CMB-style inputs (Omega_bh^2, Omega_dh^2, ...).

    Always returns a dict compatible with MODEL_func(z, p) which expects Omega_m.
    """
    if not isinstance(p, dict):
        raise TypeError(
            f"_ensure_background_params expected a parameter dict, got {type(p)}. "
            f"Update this call site to pass `param_dict`."
        )

    # Important: derive/inject background params if we're on the CMB track.
    # Use a shallow copy to avoid surprising mutations of the caller's dict.
    return U.ensure_background_params(dict(p))


def Covariance_matrix(
    model: np.ndarray,
    type_data: np.ndarray,
    type_data_error: np.ndarray,
) -> float:
    """
    Efficient χ² for diagonal covariance matrices (e.g. CC data).

    Parameters
    ----------
    model : array
        Model prediction at each data point.
    type_data : array
        Observed data.
    type_data_error : array
        1σ errors for each point.

    Returns
    -------
    float
        χ² value using a diagonal covariance Σ = diag(σ_i^2).
    """
    delta = type_data - model
    inv_diag = 1.0 / (type_data_error ** 2)
    return float(np.sum(delta * delta * inv_diag))


def Calc_chi(
    Type: str,
    type_data: np.ndarray,
    type_data_error: np.ndarray,
    model: np.ndarray,
) -> float:
    """
    General χ² evaluator.

    * For Type == "CC": use diagonal covariance (Covariance_matrix).
    * For everything else: sum of squared residuals with uncorrelated errors.
    """
    if Type == "CC":
        return Covariance_matrix(model, type_data, type_data_error)

    residual = type_data - model
    return float(np.sum((residual ** 2) / (type_data_error ** 2)))


def Calc_PantP_chi(
    mb: np.ndarray,
    trig: np.ndarray,
    cepheid: np.ndarray,
    L_cov: np.ndarray,
    model: np.ndarray,
    param_dict: Dict[str, float],
) -> float:
    """
    Pantheon+ χ² with full covariance, using a Cholesky solve.

    Parameters
    ----------
    mb : array
        Observed apparent magnitudes.
    trig : array
        Binary flag indicating which SNe are calibrators.
    cepheid : array
        Cepheid distance moduli for calibrator SNe.
    L_cov : array
        Cholesky factor (lower-triangular) of the relevant covariance block.
    model : array
        Cosmological distance moduli (mag) for each SN.
    param_dict : dict
        Parameter dictionary containing at least 'M_abs'.

    Returns
    -------
    float
        χ² value for the Pantheon+ subset.
    """
    M = param_dict.get("M_abs", -19.20)

    moduli = np.where(trig == 1, cepheid, model)
    delta = mb - M - moduli

    residuals = la.solve_triangular(L_cov, delta, lower=True, check_finite=False)
    return float(np.dot(residuals, residuals))


# -----------------------------------------------------------------------------
# BAO χ² (legacy 12-element vector)
# -----------------------------------------------------------------------------

def Calc_BAO_chi(data, Model_func, param_dict, Type) -> float:
    """
    χ² for the legacy 12-element BAO vector.

    Data requirements
    -----------------
    data["covd1"] : (12, 12) covariance matrix.
    Optionally data["obs_vec"] : (12,) observed vector. If absent, falls back
    to hard-coded values.

    Theory uses:
      * dmrd(z)  → D_M / r_d
      * dhrd(z)  → D_H / r_d
      * dvrd(z)  → D_V / r_d

    The ordering of the 12 slots matches your original BAO implementation.
    """
    cov = np.asarray(data["covd1"], dtype=float)
    if cov.shape != (12, 12):
        raise ValueError(f"BAO vector expects a 12×12 covariance; got {cov.shape}")
    cov = 0.5 * (cov + cov.T)

    try:
        L = la.cholesky(cov)
    except la.LinAlgError:
        eig_min = np.min(la.eigvalsh(cov))
        jitter = 1e-12 if eig_min > -1e-12 else 10.0 * abs(eig_min) + 1e-12
        L = la.cholesky(cov + jitter * np.eye(cov.shape[0]))

    zs_full = np.array(
        [0.295, 0.510, 0.510, 0.706, 0.706, 0.930, 0.930, 1.317, 1.317, 1.491, 2.330, 2.330],
        dtype=float,
    )
    zs_dv = np.array([0.295, 1.491], dtype=float)

    dm = dmrd(zs_full, Model_func, param_dict, Type)  # D_M / r_d
    dh = dhrd(zs_full, Model_func, param_dict, Type)  # D_H / r_d
    dv = dvrd(zs_dv, Model_func, param_dict, Type)    # D_V / r_d

    if (not np.isfinite(dm).all()) or (not np.isfinite(dh).all()) or (not np.isfinite(dv).all()):
        raise FloatingPointError("BAO theory contains non-finite values (check model/params).")

    theo = np.array(
        [
            dv[0],        # DV/rd @ 0.295
            dm[1], dh[1], # DM/rd, DH/rd @ 0.510
            dm[3], dh[3], # @ 0.706
            dm[5], dh[5], # @ 0.930
            dm[7], dh[7], # @ 1.317
            dv[1],        # DV/rd @ 1.491
            dm[10], dh[10],  # @ 2.330
        ],
        dtype=float,
    )

    if "obs_vec" in data and data["obs_vec"] is not None:
        obs = np.asarray(data["obs_vec"], dtype=float)
    else:
        obs = np.array(
            [
                7.925129270,
                13.62003080,
                20.98334647,
                16.84645313,
                20.07872919,
                21.70841761,
                17.87612922,
                27.78720817,
                13.82372285,
                26.07217182,
                39.70838281,
                8.522565830,
            ],
            dtype=float,
        )

    if obs.shape != (12,):
        raise ValueError(f"BAO observed vector must have shape (12,), got {obs.shape}")

    resid = theo - obs
    y = la.solve_triangular(L, resid, lower=True, check_finite=False)
    return float(y @ y)


# -----------------------------------------------------------------------------
# DESI BAO-like χ²
# -----------------------------------------------------------------------------

def Calc_DESI_chi(data, Model_func, param_dict, Type) -> float:
    """
    χ² for DESI VI-style BAO vectors with mixed observables.

    Supported type codes (after normalisation):
      3: D_V / r_d
      5: D_A / r_d
      6: D_H / r_d
      7: r_d / D_V
      8: D_M / r_d

    Any type==4 entries (raw D_V in Mpc) are auto-normalised to D_V / r_d
    and relabelled to 3, using a heuristic on the observed values.
    """
    def _must_calibrate(Type: str) -> bool:
        T = (Type or "").upper()
        return ("BBN" in T) or ("THETA" in T) or ("CMB" in T)

    z = np.asarray(data["redshift"], dtype=float)
    meas = np.asarray(data["measurement"], dtype=float).copy()
    types = np.asarray(data["type"], dtype=int).copy()
    cov = data.get("cov", None)
    inv_c = data.get("inv_cov", None)
    
    param_dict = _ensure_background_params(param_dict)

    DM = UDM.Comoving_distance_vectorized(Model_func, z, param_dict)  # Mpc
    Ez = Model_func(z, param_dict)
    DA = DM / (1.0 + z)
    DH = UDM.Hubble(param_dict) / Ez
    DV = (DA * DA * (1.0 + z) * (1.0 + z) * (z * DH)) ** (1.0 / 3.0)

    if _must_calibrate(Type):
        rs = _try_compute_rd(param_dict)
        if rs is None:
            return 1e300
    else:
        rs = _try_resolve_rd(param_dict, Type)
        if rs is None:
            rs = _try_compute_rd(param_dict)
        if rs is None:
            rs = float(R_D_SINGLETON)
    rs = float(rs)
    if not np.isfinite(rs) or rs <= 0.0:
        return 1e300

    # Normalise raw-DV entries if present
    if (types == 4).any():
        vals = meas[types == 4]
        med = float(np.nanmedian(np.abs(vals))) if vals.size else np.nan
        if np.isfinite(med) and med < 100.0:
            types[types == 4] = 3
        else:
            meas[types == 4] = vals / rs
            types[types == 4] = 3

    theo = np.full_like(meas, np.nan, dtype=float)
    m3 = types == 3
    m5 = types == 5
    m6 = types == 6
    m7 = types == 7
    m8 = types == 8

    theo[m3] = DV[m3] / rs
    theo[m5] = DA[m5] / rs
    theo[m6] = DH[m6] / rs
    theo[m7] = rs / DV[m7]
    theo[m8] = DM[m8] / rs

    if np.isnan(theo).any():
        unknown = sorted(set(types[np.isnan(theo)]))
        raise ValueError(f"DESI: unhandled type code(s): {unknown}. Supported: 3,5,6,7,8")

    diff = theo - meas

    if inv_c is not None:
        inv_c = np.asarray(inv_c, dtype=float)
        return float(diff @ (inv_c @ diff))

    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        try:
            x = np.linalg.solve(cov, diff)
        except np.linalg.LinAlgError:
            x = np.linalg.pinv(cov, rcond=1e-12) @ diff
        return float(diff @ x)

    raise ValueError("DESI needs 'inv_cov' or 'cov' in data.")


# -----------------------------------------------------------------------------
# BBN D/H prediction + χ²
# -----------------------------------------------------------------------------

def _bbn_predict_DH(param_dict: Dict[str, float]) -> float:
    """
    Approximate primordial D/H (number ratio) from Omega_b h^2.

    Uses a calibrated power law:
      D/H = K * (6 / eta10)^alpha,   alpha ≈ 1.6

    with K chosen such that
      Omega_b h^2 = 0.02205  →  D/H = 25.47 × 10^{-6}.
    """
    obh2 = float(param_dict["Omega_bh^2"])
    alpha = 1.6
    eta10 = 273.9 * obh2

    eta10_ref = 273.9 * 0.02205
    DH_ref = 25.47e-6

    K = DH_ref * (eta10_ref / 6.0) ** alpha
    return K * (6.0 / eta10) ** alpha


def bbn_predict_approx(p: Dict[str, float], data: Optional[dict] = None) -> float:
    """Thin wrapper around the approximate D/H predictor."""
    return _bbn_predict_DH(p)


@lru_cache(maxsize=4096)
def _bbn_call_cached(obh2_r: float, neff_r: float, tau_n_r: float) -> float:
    """
    Cached AlterBBN call.

    The AlterBBN interface is several orders of magnitude slower than the
    algebraic approximation; we cache on (Omega_b h^2, N_eff, tau_n).
    """
    from alterbbn_ctypes import run_bbn
    return float(run_bbn(obh2_r, neff_r, tau_n_r)["D_H"])


def bbn_predict_alterbbn(p: Dict[str, float], data: Optional[dict] = None) -> float:
    """
    Live AlterBBN D/H prediction.

    If `data` is provided, we use any observational Neff/tau_n overrides;
    otherwise we fall back to defaults.
    """
    obh2 = round(float(p["Omega_bh^2"]), 6)
    neff = round(float(p.get("N_eff", data.get("Neff", N_EFF_DEFAULT))) if data else N_EFF_DEFAULT, 6)
    tau = round(float(data.get("tau_n", TAU_N_DEFAULT)) if data else TAU_N_DEFAULT, 3)
    return _bbn_call_cached(obh2, neff, tau)


def Calc_BBN_DH_chi(
    data: dict,
    Model_func: Callable,
    param_dict: Dict[str, float],
    Type: str,
) -> float:
    """
    BBN primordial D/H χ².

    Accepts either:
      * a pre-sliced `obs_data` dict, or
      * a full data dict with keys 'BBN_DH' or 'BBN_DH_AlterBBN'.

    Observational structure
    -----------------------
    obs["units"]   : "scaled1e6" (PDG-style) or "absolute" (number ratio)
    obs["S"]       : PDG scale factor (default 1)
    obs["systems"] : list of systems, each with fields:
                       "DH", "sigma"  (symmetric), or
                       "DH", "sig_up", "sig_dn" (asymmetric)
    obs["weighted_mean"] : fallback dict {"DH","sigma"} if systems missing

    Backend selection
    -----------------
    via any of:
      * "bbn_model_effective"
      * "bbn_backend"
      * "bbn_model" (legacy)
    """
    # --- Accept obs_data or full data dict
    if isinstance(data, dict) and (
        ("systems" in data) or ("weighted_mean" in data) or ("units" in data)
    ):
        obs = data
    elif isinstance(data, dict) and ("BBN_DH_AlterBBN" in data or "BBN_DH" in data):
        key = "BBN_DH_AlterBBN" if "BBN_DH_AlterBBN" in data else "BBN_DH"
        obs = data[key]
    else:
        raise KeyError("BBN_DH data not found: pass obs_data or dict with 'BBN_DH'/'BBN_DH_AlterBBN'.")

    # --- Choose backend
    backend = (
        obs.get("bbn_model_effective")
        or obs.get("bbn_backend")
        or obs.get("bbn_model", "approx")
    )

    strict = bool(obs.get("strict_bbn", False) or obs.get("require_alterbbn", False))

    try:
        if backend == "alterbbn":
            DH_th = bbn_predict_alterbbn(param_dict, obs)
        elif backend == "alterbbn_grid":
            DH_th = bbn_predict_grid(param_dict, obs)
        else:
            DH_th = bbn_predict_approx(param_dict)

    except Exception as e:
        # If the user explicitly requested AlterBBN, fail loudly in strict mode
        if strict and backend in ("alterbbn", "alterbbn_grid"):
            raise RuntimeError(
                "BBN_DH requested AlterBBN backend, but it failed.\n"
                f"backend={backend}\n"
                "Fix: ensure KOSMO_BBN_LIB points to a valid libkosmo_bbn.so, and that "
                "Kosmulator_main/alterbbn_ctypes.py can import & load it.\n"
                f"Original error: {e!r}"
            ) from e

        # Otherwise keep the robust fallback
        DH_th = bbn_predict_approx(param_dict)
    units = obs.get("units", "absolute")
    scale = 1e-6 if units == "scaled1e6" else 1.0
    S = float(obs.get("S", 1.0))

    systems = obs.get("systems", []) or []
    if len(systems) > 0:
        chi2 = 0.0
        for s in systems:
            y = float(s["DH"]) * scale
            if "sigma" in s:
                sig = float(s["sigma"]) * scale
            else:
                su = float(s.get("sig_up", 0.0)) * scale
                sd = float(s.get("sig_dn", 0.0)) * scale
                if su <= 0.0 and sd <= 0.0:
                    # zero-weight system; skip
                    continue
                resid = y - DH_th
                sig = su if resid >= 0.0 else sd

            sig_eff = max(S * sig, 1e-18)
            chi2 += ((y - DH_th) / sig_eff) ** 2
        return float(chi2)

    # Fallback: weighted mean only
    if "weighted_mean" in obs:
        wm = obs["weighted_mean"]
        y = float(wm["DH"]) * scale
        sig = float(wm["sigma"]) * scale
        sig_eff = max(S * sig, 1e-18)
        return float(((y - DH_th) / sig_eff) ** 2)

    raise ValueError("BBN_DH obs has neither 'systems' nor 'weighted_mean'.")


# -----------------------------------------------------------------------------
# BBN backend initialisation (approx / alterbbn / alterbbn_grid)
# -----------------------------------------------------------------------------
def _import_run_bbn():
    """
    Import run_bbn from alterbbn_ctypes.
    Search order:
    1) Normal python import (module on PYTHONPATH)
    2) Repo-bundled helper: <Kosmulator>/AlterBBN_files/alterbbn_ctypes.py
    """
    # 1) normal import
    try:
        from alterbbn_ctypes import run_bbn  # type: ignore
        return run_bbn
    except Exception as e0:
        # 2) load from AlterBBN_files (no need to copy into Kosmulator_main)
        here = Path(__file__).resolve()
        kosmulator_root = here.parent.parent  # .../Kosmulator
        candidate = kosmulator_root / "AlterBBN_files" / "alterbbn_ctypes.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("alterbbn_ctypes", str(candidate))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["alterbbn_ctypes"] = mod  # allow downstream imports
                spec.loader.exec_module(mod)
                if hasattr(mod, "run_bbn"):
                    return mod.run_bbn
        # if still failing, re-raise original error (more informative)
        raise e0

def ensure_bbn_backend(
    bbn_data: dict,
    logger: Optional[logging.Logger] = None,
    priors: dict | None = None,
) -> dict:
    """
    Choose and prepare the BBN backend (approx / alterbbn / alterbbn_grid).

    Parameters
    ----------
    bbn_data : dict
        Observational/config dictionary for BBN D/H.
    logger : logging.Logger, optional
        Logger for status messages. If None, uses module-level logger.
    priors : dict, optional
        Prior ranges for:
          'Omega_bh^2', 'N_eff', 'tau_n' → (low, high).

    Observational overrides
    -----------------------
    bbn_data may also include:
      - 'obh2_grid', 'Neff_grid', 'tau_grid' : explicit grid nodes.
      - 'obh2_step', 'Neff_step', 'tau_step' : defaults 1e-5, 0.02, 1.0 s.
      - 'bbn_grid_path' : path to .npz grid file (default OBSERVATIONS_BASE/BBN_GRID_RELATIVE).
      - 'obh2_buffer', 'Neff_buffer', 'tau_buffer' : small buffer around priors.
    """
    import os

    log = logger or globals().get("logger", None)
    out = dict(bbn_data)
    model = str(out.get("bbn_model", "approx")).lower()

    def set_backend(name: str) -> None:
        out["bbn_backend"] = name
        out["bbn_model_effective"] = name

    # --- approx: nothing to prepare
    if model == "approx":
        set_backend("approx")
        if log:
            log.info("BBN backend: approx")
        return out

    # --- derive ranges from priors
    def _span(key: str, default_pair) -> tuple[float, float]:
        if priors and key in priors:
            lo, hi = priors[key]
        else:
            lo, hi = default_pair
        lo, hi = float(lo), float(hi)
        return (min(lo, hi), max(lo, hi))

    obh2_lo, obh2_hi = _span("Omega_bh^2", (0.0200, 0.0240))
    neff_lo, neff_hi = _span("N_eff", (N_EFF_DEFAULT, N_EFF_DEFAULT))
    tau_lo, tau_hi = _span("tau_n", (out.get("tau_n", TAU_N_DEFAULT), out.get("tau_n", TAU_N_DEFAULT)))

    neff_fixed = np.isclose(neff_hi, neff_lo, atol=1e-12)
    tau_fixed = np.isclose(tau_hi, tau_lo, atol=1e-12)

    obh2_eps = float(out.get("obh2_buffer", 2e-5))
    neff_eps = float(out.get("Neff_buffer", 0.02))
    tau_eps = float(out.get("tau_buffer", 0.0))

    obh2_box = (obh2_lo - obh2_eps, obh2_hi + obh2_eps)
    neff_box = (neff_lo, neff_hi) if neff_fixed else (neff_lo - neff_eps, neff_hi + neff_eps)
    tau_box = (tau_lo, tau_hi) if tau_fixed else (tau_lo - tau_eps, tau_hi + tau_eps)

    # --- choose nodes
    def _mk_nodes(lo: float, hi: float, step: float) -> np.ndarray:
        if np.isclose(lo, hi):
            return np.array([lo], float)
        n = int(np.floor((hi - lo) / step + 0.5)) + 1
        return np.linspace(lo, hi, n, dtype=float)

    obh2_step = float(out.get("obh2_step", 1e-5))
    neff_step = float(out.get("Neff_step", 0.02))
    tau_step = float(out.get("tau_step", 1.0))

    if "obh2_grid" in out:
        obh2_nodes = np.asarray(out["obh2_grid"], float)
    else:
        obh2_nodes = _mk_nodes(*obh2_box, obh2_step)

    if "Neff_grid" in out:
        neff_nodes = np.asarray(out["Neff_grid"], float)
    else:
        neff_nodes = np.array([neff_lo], float) if neff_fixed else _mk_nodes(*neff_box, neff_step)

    if "tau_grid" in out:
        tau_nodes = np.asarray(out["tau_grid"], float)
    else:
        tau_nodes = np.array([tau_lo], float) if tau_fixed else _mk_nodes(*tau_box, tau_step)

    path = out.get("bbn_grid_path", None)
    want_grid = (model == "alterbbn_grid") or (model == "alterbbn" and path)

    if want_grid and not path:
        path = os.path.join(OBSERVATIONS_BASE, BBN_GRID_RELATIVE)
        out["bbn_grid_path"] = path

    def _normalize_grid_shape(G: np.ndarray, n_obh2: int, n_neff: int, n_tau: int) -> np.ndarray:
        G = np.asarray(G, float)
        if G.ndim == 3:
            if G.shape != (n_obh2, n_neff, n_tau):
                raise ValueError(f"Grid shape mismatch: {G.shape} vs ({n_obh2},{n_neff},{n_tau})")
            return G
        if G.ndim == 2:
            if n_tau == 1 and G.shape == (n_obh2, n_neff):
                return G.reshape(n_obh2, n_neff, 1)
            if n_neff == 1 and G.shape == (n_obh2, n_tau):
                return G.reshape(n_obh2, 1, n_tau)
            raise ValueError(
                f"2D grid shape {G.shape} not compatible with (obh2,neff,tau)=({n_obh2},{n_neff},{n_tau})"
            )
        if G.ndim == 1:
            if n_neff == 1 and n_tau == 1 and G.shape == (n_obh2,):
                return G.reshape(n_obh2, 1, 1)
            raise ValueError(f"1D grid shape {G.shape} not compatible with requested axes")
        raise ValueError(f"Unsupported grid ndim={G.ndim}")

    # --- Try loading a pre-built grid
    force = bool(out.get("bbn_force_rebuild", False))
    if want_grid and path and not force:
        try:
            npz = np.load(path, allow_pickle=False)
            g_obh2 = np.asarray(npz["obh2"], float)
            g_neff_full = (
                np.asarray(npz["neff"], float)
                if "neff" in npz.files
                else np.array([N_EFF_DEFAULT], float)
            )
            g_tau_full = (
                np.asarray(npz["tau"], float)
                if "tau" in npz.files
                else np.array([float(npz.get("tau_n", out.get("tau_n", TAU_N_DEFAULT)))], float)
            )

            if "logDH" in npz.files:
                G = np.asarray(npz["logDH"], float)
                logspace = True
            elif "DH" in npz.files:
                G = np.asarray(npz["DH"], float)
                logspace = False
            else:
                raise KeyError("Grid file missing 'logDH'/'DH' key")

            G = _normalize_grid_shape(G, g_obh2.size, g_neff_full.size, g_tau_full.size)

            covers = (
                g_obh2.min() <= obh2_box[0] + 1e-12
                and g_obh2.max() >= obh2_box[1] - 1e-12
                and g_neff_full.min() <= neff_box[0] + 1e-12
                and g_neff_full.max() >= neff_box[1] - 1e-12
                and g_tau_full.min() <= tau_box[0] + 1e-12
                and g_tau_full.max() >= tau_box[1] - 1e-12
            )

            if not covers:
                if log:
                    log.warning(
                        "Grid %s does not cover priors "
                        "[obh2 %.6f..%.6f vs %.6f..%.6f, "
                        "Neff %.3f..%.3f vs %.3f..%.3f, "
                        "tau %.2f..%.2f vs %.2f..%.2f]; rebuilding.",
                        path,
                        g_obh2.min(),
                        g_obh2.max(),
                        *obh2_box,
                        g_neff_full.min(),
                        g_neff_full.max(),
                        *neff_box,
                        g_tau_full.min(),
                        g_tau_full.max(),
                        *tau_box,
                    )
            else:
                g_neff, g_tau = g_neff_full, g_tau_full

                if neff_fixed and g_neff.size > 1:
                    j = int(np.abs(g_neff - neff_lo).argmin())
                    if log:
                        log.info("BBN grid: collapsing Neff to nearest node %.3f", g_neff[j])
                    G = G[:, j, :]
                    g_neff = np.array([g_neff[j]], float)

                if tau_fixed and g_tau.size > 1:
                    k = int(np.abs(g_tau - tau_lo).argmin())
                    if log:
                        log.info("BBN grid: collapsing tau to nearest node %.1f s", g_tau[k])
                    G = G[:, :, k]
                    g_tau = np.array([g_tau[k]], float)

                G = _normalize_grid_shape(G, g_obh2.size, g_neff.size, g_tau.size)

                if log:
                    dims = (g_obh2.size, g_neff.size, g_tau.size)
                    log.info("Loaded BBN grid %s: shape=%s, effective dims=%s, logspace=%s", path, G.shape, dims, logspace)

                grid = {"obh2": g_obh2, "DH": G, "logspace": bool(logspace)}
                if g_neff.size > 1:
                    grid["neff"] = g_neff
                if g_tau.size > 1:
                    grid["tau"] = g_tau

                out["bbn_grid"] = grid
                set_backend("alterbbn_grid")
                return out

        except Exception as e:
            if log:
                log.warning("Failed to load BBN grid at %s; will attempt build. Reason: %s", path, e)
    else:
        if force and log:
            log.info("BBN grid: force rebuild requested (bbn_force_rebuild=True).")

    # --- Build grid if needed
    if want_grid:
        try:
            from alterbbn_ctypes import run_bbn
        except Exception as e:
            strict = bool(out.get("bbn_strict", False))
            run_bbn = _import_run_bbn()
            if strict:
                raise RuntimeError(
                    "AlterBBN requested but alterbbn_ctypes could not be imported.\n"
                    "Fix: (1) build AlterBBN + libkosmo_bbn.so, (2) set env var KOSMO_BBN_LIB, "
                    "and (3) keep alterbbn_ctypes.py either on PYTHONPATH or in Kosmulator/AlterBBN_files/.\n"
                    f"Original import error: {e}"
                )
            if model == "alterbbn_grid":
                if log:
                    log.warning("alterbbn_ctypes unavailable; reverting to approx (grid requested). %s", e)
                set_backend("approx")
                return out
            if log:
                log.warning("alterbbn_ctypes unavailable; falling back to live alterbbn. %s", e)
            set_backend("alterbbn")
            return out

        nO, nN, nT = obh2_nodes.size, neff_nodes.size, tau_nodes.size
        arr = np.empty((nO, nN, nT), float)

        for i, x in enumerate(obh2_nodes):
            xf = float(x)
            for j, y in enumerate(neff_nodes):
                yf = float(y)
                for k, t in enumerate(tau_nodes):
                    tf = float(t)
                    arr[i, j, k] = float(run_bbn(xf, yf, tf)["D_H"])

        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError("ensure_bbn_backend: built grid contains non-positive/non-finite D/H values.")

        logDH = np.log(arr)

        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            np.savez(
                path,
                obh2=obh2_nodes,
                neff=neff_nodes,
                tau=tau_nodes,
                logDH=logDH,
                units="absolute",
                built_with="AlterBBN ctypes",
                prior_box=np.array([obh2_box, neff_box, tau_box], float),
            )
            if log:
                log.info("Wrote BBN grid to %s: shape=%s", path, logDH.shape)

        grid = {"obh2": obh2_nodes, "DH": logDH, "logspace": True}
        if neff_nodes.size > 1:
            grid["neff"] = neff_nodes
        if tau_nodes.size > 1:
            grid["tau"] = tau_nodes
        out["bbn_grid"] = grid
        set_backend("alterbbn_grid")
        return out

    # --- live AlterBBN, no grid
    if model == "alterbbn":
        set_backend("alterbbn")
        if log:
            log.info("BBN backend: alterbbn (live)")
        return out

    # safety net
    set_backend("approx")
    if log:
        log.info("BBN backend: approx (fallback)")
    return out


def bbn_predict_grid(param_dict: Dict[str, float], obs: dict) -> float:
    """
    Predict D/H from a prebuilt grid (obs['bbn_grid']).

    Interpolation strategy
    ----------------------
    * Always interpolate in log(D/H).
    * PCHIP along Ω_b h^2.
    * Quadratic interpolation across N_eff and tau_n (with linear fallback at edges).
    * Supports 1D (Ω only), 2D (Ω + one fixed axis), and 3D grids.
    """
    grid = obs["bbn_grid"]
    obh2_nodes = np.asarray(grid["obh2"], float)
    vals_raw = np.asarray(grid["DH"], float)  # log(D/H) if logspace True
    logspace = bool(grid.get("logspace", True))

    neff_nodes = np.asarray(grid["neff"], float) if "neff" in grid else np.array([3.046], float)
    tau_nodes = (
        np.asarray(grid["tau"], float)
        if "tau" in grid
        else np.array([float(obs.get("tau_n", TAU_N_DEFAULT))], float)
    )

    obh2 = float(param_dict["Omega_bh^2"])
    Neff = float(param_dict.get("N_eff", neff_nodes[0]))
    tau = float(param_dict.get("tau_n", tau_nodes[0]))

    if not (obh2_nodes[0] <= obh2 <= obh2_nodes[-1]):
        raise ValueError("bbn_predict_grid: Omega_bh^2 outside grid.")
    if not (neff_nodes[0] <= Neff <= neff_nodes[-1]):
        raise ValueError("bbn_predict_grid: N_eff outside grid.")
    if not (tau_nodes[0] <= tau <= tau_nodes[-1]):
        raise ValueError("bbn_predict_grid: tau_n outside grid.")

    V = vals_raw if logspace else np.log(vals_raw)

    NΩ, Nn, Nt = obh2_nodes.size, neff_nodes.size, tau_nodes.size
    if V.ndim == 3 and V.shape == (NΩ, Nn, Nt):
        pass
    elif V.ndim == 2:
        if Nt == 1 and V.shape == (NΩ, Nn):
            V = V.reshape(NΩ, Nn, 1)
        elif Nn == 1 and V.shape == (NΩ, Nt):
            V = V.reshape(NΩ, 1, Nt)
        else:
            raise ValueError(f"Unsupported 2D grid shape {V.shape} for axes {(NΩ, Nn, Nt)}")
    elif V.ndim == 1 and Nn == 1 and Nt == 1 and V.shape == (NΩ,):
        V = V.reshape(NΩ, 1, 1)
    else:
        raise ValueError(f"Unsupported grid ndim/shape: {V.ndim}, {V.shape} for axes {(NΩ, Nn, Nt)}")

    _pchip_cache: dict[tuple[int, int], PchipInterpolator] = {}

    def _f_Omega(j: int, k: int) -> float:
        key = (j, k)
        if key not in _pchip_cache:
            _pchip_cache[key] = PchipInterpolator(obh2_nodes, V[:, j, k], extrapolate=False)
        return float(_pchip_cache[key](obh2))

    def _quad_interp_safe(nodes: np.ndarray, x: float, get_at_idx: Callable[[int], float]) -> float:
        N = nodes.size
        if N == 1:
            return float(get_at_idx(0))

        j = int(np.searchsorted(nodes, x) - 1)
        j = max(0, min(j, N - 2))
        x0, x1 = nodes[j], nodes[j + 1]
        v0, v1 = get_at_idx(j), get_at_idx(j + 1)

        if 1 <= j <= N - 3:
            xm, vm = nodes[j - 1], get_at_idx(j - 1)

            def L(xq, xa, xb, xc):
                return ((xq - xb) * (xq - xc)) / ((xa - xb) * (xa - xc))

            return float(vm * L(x, xm, x0, x1) + v0 * L(x, x0, xm, x1) + v1 * L(x, x1, xm, x0))

        # Edge: linear
        t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        return float((1 - t) * v0 + t * v1)

    if Nn == 1 and Nt == 1:
        vlog = float(PchipInterpolator(obh2_nodes, V[:, 0, 0], extrapolate=False)(obh2))
    elif Nn > 1 and Nt == 1:
        vlog = _quad_interp_safe(neff_nodes, Neff, lambda jj: _f_Omega(jj, 0))
    elif Nn == 1 and Nt > 1:
        vlog = _quad_interp_safe(tau_nodes, tau, lambda kk: _f_Omega(0, kk))
    else:
        def _g_tau(kk: int) -> float:
            return _quad_interp_safe(neff_nodes, Neff, lambda jj: _f_Omega(jj, kk))

        vlog = _quad_interp_safe(tau_nodes, tau, _g_tau)

    return float(np.exp(vlog))


# -----------------------------------------------------------------------------
# BAO geometry helpers: D_M / r_d, D_H / r_d, D_V / r_d, D_A / r_d
# -----------------------------------------------------------------------------

def _asarray(x: Any) -> np.ndarray:
    """
    Robust 1D array wrapper:
      * scalars → shape (1,)
      * arrays stay arrays
    """
    a = np.asarray(x, dtype=float)
    return a if a.ndim > 0 else a.reshape(1)


def _E_of_z(z, MODEL_func: Callable, p: Dict[str, float]) -> np.ndarray:
    """
    Wrapper for E(z) = H(z)/H0 using the model's vectorised background.
    """
    z_arr = _asarray(z)
    p = _ensure_background_params(p)
    Ez = MODEL_func(z_arr, p)
    return np.asarray(Ez, dtype=float)


def dmrd(z, MODEL_func: Callable, p: Dict[str, float], kind: Optional[str]):
    """
    D_M / r_d at one or many redshifts.

    Uses:
      * UDM.Comoving_distance_vectorized for geometry (flat).
      * _resolve_rd(...) for r_d, so BAO obeys rd_policy / early-time calibrators.
    """
    z_arr = _asarray(z)
    p = _ensure_background_params(p)
    DM = UDM.Comoving_distance_vectorized(MODEL_func, z_arr, p)
    rd = _resolve_rd(p, kind or "")
    out = DM / rd
    return out if out.size > 1 else float(out)


def dhrd(redshift, MODEL_func: Callable, p: Dict[str, float], Type: Optional[str]):
    """
    D_H / r_d = (c / H0) / (E(z) r_d).

    Uses the model's E(z) and rd_helpers._resolve_rd for r_d.
    """
    z_arr = _asarray(redshift)
    p = _ensure_background_params(p)
    Ez = _E_of_z(z_arr, MODEL_func, p)

    rd = _resolve_rd(p, Type or "")
    DH = C_KM_S / float(p["H_0"]) / Ez
    out = DH / rd
    return out if out.size > 1 else float(out)


def dvrd(
    redshifts: Number,
    MODEL_func: Callable,
    p: Dict[str, float],
    Type: Optional[str] = None,
):
    """
    D_V / r_d at one or many redshifts.

    D_V(z) = [ D_M(z)^2 * (c z / H(z)) ]^(1/3)
           = [ D_M(z)^2 * ((c/H0) z / E(z)) ]^(1/3).
    """
    z = _asarray(redshifts)
    if z.size == 0:
        return z.astype(float)

    p = _ensure_background_params(p)
    DM = UDM.Comoving_distance_vectorized(MODEL_func, z, p)
    Ez = _E_of_z(z, MODEL_func, p)

    cz_over_Hz = UDM.Hubble(p) / Ez
    DV = (DM * DM * cz_over_Hz * z) ** (1.0 / 3.0)

    rd = _resolve_rd(p, Type or "")
    out = DV / rd
    return out if out.size > 1 else float(out)


def dArd(
    redshifts: Number,
    MODEL_func: Callable,
    p: Dict[str, float],
    Type: Optional[str] = None,
):
    """
    D_A / r_d at one or many redshifts.

      D_A(z) = D_M(z) / (1+z)
    """
    z = _asarray(redshifts)
    p = _ensure_background_params(p)
    DM = UDM.Comoving_distance_vectorized(MODEL_func, z, p)
    rd = _resolve_rd(p, Type or "")
    out = (DM / (1.0 + z)) / rd
    return out if out.size > 1 else float(out)
