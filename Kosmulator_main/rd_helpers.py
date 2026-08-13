"""
rd_helpers.py — canonical r_d (sound horizon) helpers for Kosmulator.

We centralise Eisenstein & Hu (1998) style r_d computation here so that
both the likelihood code and plotting code use the exact same implementation.

Public / semi-public pieces:
  - compute_rd(p)                     : EH98 r_d, no CLASS.
  - compute_rd_class(p)               : r_d from CLASS.rs_drag() (if available).
  - _try_compute_rd(p)                : CLASS → EH98 fallback, returns float or None.
  - _resolve_rd(p, Type)              : full r_d policy (fixed / calibrated / free).
  - _try_resolve_rd(p, Type)          : safe wrapper for callers that want None on failure.
  - _compute_r_d_from_bbn_and_background(p) : centralised r_d for BBN-calibrated runs.
  - _maybe_calibrate_rd(theta_map, CONFIG, obs_index) : convenience hook for MCMC setup.
"""

from __future__ import annotations

import math
from typing import Mapping
from functools import lru_cache
import logging

logger = logging.getLogger("Kosmulator.rd_helpers")

# ------------------------------------------------------------------
# Optional CLASS backend
# ------------------------------------------------------------------
try:
    from classy import Class
    _HAVE_CLASS = True
except Exception:
    _HAVE_CLASS = False

_RD_BACKEND_LOGGED = False

# ------------------------------------------------------------------
# Shared constants
# ------------------------------------------------------------------
try:
    from Kosmulator_main.constants import (
        T_CMB_DEFAULT as _T_CMB,
        N_EFF_DEFAULT as _N_EFF,
        R_D_SINGLETON as _R_D_SINGLETON,
    )
    R_D_SINGLETON = float(_R_D_SINGLETON)
except Exception:
    # Very safe literals if constants module is unavailable (tests / docs)
    _T_CMB = 2.7255
    _N_EFF = 3.046
    R_D_SINGLETON = 147.5


# ------------------------------------------------------------------
# EH98 r_d
# ------------------------------------------------------------------
def compute_rd(p: Mapping[str, float]) -> float:
    """
    Eisenstein & Hu (1998) r_d (sound horizon at drag epoch) in Mpc.

    Parameters in `p` (mapping/dict):
      - H_0            : Hubble constant [km s^-1 Mpc^-1]
      - Omega_m        : Total matter density parameter (or infer from
                         Omega_bh^2+Omega_dh^2/Omega_ch^2 with h)
      - Omega_bh^2     : Physical baryon density
      - N_eff          : (optional) Effective number of relativistic species
                         (default ~ 3.046)
      - T_CMB          : (optional) CMB temperature [K]; default ≈ 2.7255

    Notes:
      - Implements EH98 eqs. for z_eq, k_eq, z_d and r_s(z_d).
      - This implementation is N_eff- and T_CMB-aware via the radiation
        density, matching the likelihood backend behaviour.
    """
    H0   = float(p["H_0"])
    h    = H0 / 100.0
    obh2 = float(p["Omega_bh^2"])

    # Omega_m: accept explicit Omega_m or reconstruct from physical densities
    if "Omega_m" in p:
        Om = float(p["Omega_m"])
    else:
        if "Omega_dh^2" in p:
            Om = (obh2 + float(p["Omega_dh^2"])) / (h * h)
        elif "Omega_ch^2" in p:
            Om = (obh2 + float(p["Omega_ch^2"])) / (h * h)
        else:
            raise KeyError(
                "Need Omega_m (or Omega_bh^2 + Omega_dh^2/Omega_ch^2) "
                "to compute r_d."
            )

    Neff = float(p.get("N_eff", _N_EFF))
    Tcmb = float(p.get("T_CMB", _T_CMB))

    # ---- Radiation density and equality
    # Photons: Omega_gamma h^2 = 2.469e-5 * (T/2.7255)^4
    ogam_h2 = 2.469e-5 * (Tcmb / 2.7255) ** 4
    # Neutrinos: factor 0.227107317... = (7/8)*(4/11)^(4/3)
    one_plus_0p2271Neff = 1.0 + 0.227107317660239 * Neff
    orad_h2 = ogam_h2 * one_plus_0p2271Neff

    omh2 = Om * h * h
    one_plus_zeq = omh2 / orad_h2
    zeq = max(one_plus_zeq - 1.0, 1.0)  # guard against nonsense
    aeq = 1.0 / (1.0 + zeq)             # currently unused, kept for clarity

    # ---- Drag redshift z_d  (EH98 fitting formula)
    b1 = 0.313 * omh2 ** (-0.419) * (1.0 + 0.607 * omh2 ** 0.674)
    b2 = 0.238 * omh2 ** 0.223
    zd = (
        1291.0
        * omh2 ** 0.251
        / (1.0 + 0.659 * omh2 ** 0.828)
        * (1.0 + b1 * obh2 ** b2)
    )

    # ---- Baryon-to-photon ratio R(z)
    theta2p7_4 = (Tcmb / 2.7) ** 4
    Rd  = 31.5 * obh2 / theta2p7_4 * (1000.0 / zd)
    Req = 31.5 * obh2 / theta2p7_4 * (1000.0 / zeq)

    # ---- Equality scale and sound horizon (EH98 Eq. 30)
    theta = Tcmb / 2.7
    keq   = 7.46e-2 * (Om * (H0 / 100.0) ** 2) * theta ** (-2.0)  # Mpc^-1
    num   = math.sqrt(1.0 + Rd) + math.sqrt(Rd + Req)
    den   = 1.0 + math.sqrt(Req)
    rs    = (2.0 / (3.0 * keq)) * math.sqrt(6.0 / Req) * math.log(num / den)

    return float(rs)


# ------------------------------------------------------------------
# CLASS backend
# ------------------------------------------------------------------
@lru_cache(maxsize=8192)
def _rd_class_core(
    h: float,
    omega_b: float,
    omega_cdm: float,
    N_ur: float,
    T_cmb: float,
    Omega_k: float,
    YHe_mode,
    tau_n,
) -> float:
    """
    Cached CLASS.rs_drag() evaluator.

    Args are pre-quantised for cache friendliness; see compute_rd_class().
    """
    pars = {
        "h":         float(h),
        "omega_b":   float(omega_b),
        "omega_cdm": float(omega_cdm),
        "N_ur":      float(N_ur),
        "T_cmb":     float(T_cmb),
        "Omega_k":   float(Omega_k),
        "output":    "",  # background-only is enough for rs_drag()
    }

    if str(YHe_mode).upper() == "BBN":
        pars["YHe"] = "BBN"
        if tau_n is not None:
            pars["tau_neutron"] = float(tau_n)
    else:
        pars["YHe"] = float(YHe_mode)

    c = Class()
    c.set(pars)
    c.compute()
    rd = float(c.rs_drag())
    try:
        c.struct_cleanup()
        c.empty()
    except Exception:
        pass
    return rd


def _q(x, step):
    """Quantise x to bins of width `step` for caching."""
    if x is None:
        return None
    return step * round(float(x) / step)


def compute_rd_class(p: dict):
    """
    Compute r_d via CLASS.rs_drag(), with conservative parameter quantisation.
    Returns None if CLASS is unavailable.
    """
    if not _HAVE_CLASS:
        return None

    H0   = float(p["H_0"])
    h    = H0 / 100.0
    Obh2 = float(p["Omega_bh^2"])

    # pick ω_cdm from map or derive it
    if "Omega_dh^2" in p:
        ocdmh2 = float(p["Omega_dh^2"])
    else:
        Om  = float(p["Omega_m"])
        Ob  = Obh2 / (h * h)
        Onu = float(p.get("Omega_nu", 0.0))
        Oc  = max(Om - Ob - Onu, 0.0)
        ocdmh2 = Oc * h * h

    N_ur = float(p.get("N_ur", p.get("N_eff", _N_EFF)))
    Tcmb = float(p.get("T_CMB", _T_CMB))
    Ok   = float(p.get("Omega_k", 0.0))
    tau_n = float(p["tau_n"]) if "tau_n" in p else None

    # --- Quantise (induces << 0.01 Mpc change in r_d)
    hq    = _q(h,      5e-4)   # 0.05% bins in h
    obq   = _q(Obh2,   2e-7)   # ~1e-5 relative bins in ω_b
    ocq   = _q(ocdmh2, 2e-7)   # ~1e-5 relative bins in ω_cdm
    nurq  = _q(N_ur,   1e-3)
    tcmbq = _q(Tcmb,   1e-3)
    okq   = _q(Ok,     5e-4)
    taunq = _q(tau_n,  0.1) if tau_n is not None else None

    return _rd_class_core(hq, obq, ocq, nurq, tcmbq, okq, "BBN", taunq)


# ------------------------------------------------------------------
# Policy helpers
# ------------------------------------------------------------------
def _try_compute_rd(param_dict: dict):
    """
    Central helper: try CLASS rs_drag() first; fall back to EH98 compute_rd().
    Returns r_d [Mpc] or None if both fail.
    """
    global _RD_BACKEND_LOGGED

    # 1) CLASS first
    try:
        rd = compute_rd_class(param_dict)
        if rd is not None:
            if not _RD_BACKEND_LOGGED:
                logger.debug(
                    "r_d backend = CLASS; r_d = %.3f Mpc @ p=%s",
                    rd,
                    {
                        k: param_dict[k]
                        for k in ("H_0", "Omega_m", "Omega_bh^2")
                        if k in param_dict
                    },
                )
                _RD_BACKEND_LOGGED = True
            return float(rd)
    except Exception as e:
        if not _RD_BACKEND_LOGGED:
            logger.warning(
                "CLASS rs_drag() failed (%s); falling back to EH98.", e
            )
            _RD_BACKEND_LOGGED = True

    # 2) EH98 fallback
    try:
        rd = float(compute_rd(param_dict))
        if not _RD_BACKEND_LOGGED:
            logger.debug(
                "r_d backend = EH98; r_d = %.3f Mpc @ p=%s",
                rd,
                {
                    k: param_dict[k]
                    for k in ("H_0", "Omega_m", "Omega_bh^2")
                    if k in param_dict
                },
            )
            _RD_BACKEND_LOGGED = True
        return rd
    except Exception:
        return None


def _resolve_rd(p: dict, Type: str) -> float:
    """
    r_d policy for a given parameter map and dataset type.

    Rules:
      • If 'r_d' present in p:
          → Use that (free parameter case, e.g. BAO/DESI combos without BBN/CMB).
      • Else if we have full background keys (H_0, Omega_m, Omega_bh^2):
          → Compute r_d via CLASS→EH98 (calibrated by early-time datasets).
      • Else if Type is a singleton BAO/DESI-like probe:
          → Use fixed R_D_SINGLETON [Mpc].
      • Otherwise:
          → Error: this configuration must carry a free 'r_d' parameter.
    """
    # 1) Explicit r_d always wins
    if "r_d" in p:
        return float(p["r_d"])

    # 2) If we have a background, try to compute r_d (CLASS first, then EH98)
    have_bg = all(k in p for k in ("H_0", "Omega_m", "Omega_bh^2"))
    if have_bg:
        rd = _try_compute_rd(p)
        if rd is not None:
            return float(rd)

    # 3) Pure BAO/DESI-only combinations: fall back to a fixed fiducial sound horizon
    if Type in ("BAO", "DESI", "DESI_DR1", "DESI_DR2"):
        return float(R_D_SINGLETON)

    # 4) Otherwise this combo must carry a free r_d parameter
    raise ValueError(
        "r_d is required here. Provide 'r_d' (free) or include BBN/CMB so it "
        "can be calibrated from the background."
    )


def _try_resolve_rd(param_dict, Type):
    """
    Thin wrapper around _resolve_rd that never raises.

    Used by Statistical_packages: returns r_d [Mpc] or None if resolution fails.
    """
    try:
        return _resolve_rd(param_dict, Type)
    except Exception:
        return None


def _compute_r_d_from_bbn_and_background(p: dict) -> float:
    """
    Centralised r_d [Mpc] for calibrated runs.

    Try CLASS rs_drag via _try_compute_rd; fall back to EH98 compute_rd().
    """
    try:
        rd = _try_compute_rd(p)
        if rd is not None:
            return float(rd)
    except Exception:
        pass
    # Fallback: EH98
    return float(compute_rd(p))


def _maybe_calibrate_rd(theta_map: dict, CONFIG: dict, obs_index: int) -> None:
    """
    If this observation set includes BBN and r_d is absent, compute it from
    (Omega_bh^2, Omega_m, H_0[, N_eff, T_CMB]) and inject into theta_map.

    Used by Kosmulator_MCMC at the θ→param stage so BAO/DESI runs that
    include BBN can have r_d pre-calibrated.
    """
    try:
        if "r_d" in theta_map:
            return
        obs_set = CONFIG.get("observations", [])[obs_index]
        has_bbn = any(str(o).startswith(("BBN",)) for o in obs_set)
        if not has_bbn:
            return
        needed = ("Omega_bh^2", "Omega_m", "H_0")
        if not all(k in theta_map for k in needed):
            # Silent no-op; downstream will raise if BAO/DESI needs r_d
            return
        theta_map["r_d"] = _compute_r_d_from_bbn_and_background(theta_map)
    except Exception:
        # Fail-safe: do nothing; BAO/DESI branch will error clearly if r_d missing
        pass
