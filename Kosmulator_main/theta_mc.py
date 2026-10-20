"""
100 theta_MC: CosmoMC's approximation to the angular size of the sound horizon at last
scattering, as CAMB's CosmomcTheta (Lewis & Bridle 2002; Hu & Sugiyama 1996, ApJ 471, 542, for
the redshift of last scattering), the parameter Planck and DESI DR2 sample in place of H_0
(DESI DR2, arXiv:2503.14738, Sect. V: a flat prior on 100 theta_MC). Kosmulator.py
cmb_sample_theta = True: groups with primary CMB data sample "100theta_MC" instead of H_0, and
H_0 is solved from it at every point (h_from_theta); the chain's theta column is turned into H_0
before the statistics (theta_chain_to_H0), so the tables and figures show H_0 as usual.

    z_*     = 1048 [1 + 0.00124 omega_b^-0.738] [1 + g1 omega_m^g2]                 (Hu & Sugiyama)
    r_s     = int_0^a_* da c_s / (a^2 H),  c_s = 1 / sqrt(3 (1 + 3e4 a omega_b))
    theta   = r_s / D_M(z_*)

The background is the model's own: its late-time part X(z) = E^2 - Omega_m (1+z)^3 - Omega_r (1+z)^4
from the model's E(z) (as rd_Class), with photons, massless neutrinos (N_ur), the massive
neutrino with its Fermi-Dirac density (CLASS's T_ncdm = 0.71611), baryons and CDM. Agrees with
CAMB's cosmomc_theta to about 1e-5 (kit section THETA); a flat prior on it is DESI's prior
whatever the small difference, since theta is only a coordinate.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, Optional

import numpy as np

from Kosmulator_main import constants as K

THETA_PARAM = K.THETA_MC_PARAM          # "100theta_MC"
_C_KM_S = 299792.458
_T_NCDM = 0.71611                     # CLASS's default T_ncdm (T_nu / T_cmb for the massive state)
_K_B_EV = 8.617333262e-5              # eV / K
_OMEGA_GAMMA_2P7255 = 2.47304e-5      # omega_gamma at T_cmb = 2.7255 K (CAMB 2.0: 2.472975e-5; no effect)
#: The model's late-time part X(z) (its dark energy) is E^2 - Omega_m (1+z)^3 - Omega_r (1+z)^4 from
#: the model's E(z) up to this redshift. Beyond it the difference loses its precision (cancellation),
#: so X continues as the power law in (1+z) through its values at z = _Z_X_FIT and _Z_X_MAX (exact for
#: CPL, whose density goes as (1+z)^(3 (1 + w0 + wa)) at high z, and for a cosmological constant),
#: or is 0 when it is below _X_REL_MIN of E^2 at _Z_X_MAX (dark energy negligible there)
_Z_X_MAX = 3000.0
_Z_X_FIT = 1000.0
_X_REL_MIN = 1e-12


@lru_cache(maxsize=1)
def _fd_table():
    """F(y) / F(0) for F(y) = int q^2 sqrt(q^2 + y^2) / (e^q + 1) dq: rho_nu / rho_nu(relativistic)."""
    q = np.linspace(1e-6, 60.0, 6001)
    f = q * q / (np.exp(q) + 1.0)
    ys = np.concatenate(([0.0], np.logspace(-4, 4, 1601)))
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")       # NumPy 2 renamed trapz
    F = np.array([trap(f * np.sqrt(q * q + y * y), q) for y in ys])
    F0 = 7.0 * np.pi ** 4 / 120.0
    return ys, F / F0


def _nu_ratio(y: np.ndarray) -> np.ndarray:
    """F(y)/F(0); for y beyond the table the non-relativistic limit y * 3 zeta(3) / 2 / F(0)."""
    ys, R = _fd_table()
    y = np.asarray(y, float)
    out = np.interp(y, ys, R)
    big = y > ys[-1]
    if np.any(big):
        out[big] = y[big] * (1.5 * 1.2020569031595942) / (7.0 * np.pi ** 4 / 120.0)
    return out


def _species(p: Dict[str, float]):
    """omega_gamma, omega_ur (relativistic, today), the massive neutrinos (m in eV, number) and T_cmb."""
    t_cmb = float(p.get("T_CMB", K.T_CMB_DEFAULT))
    og = _OMEGA_GAMMA_2P7255 * (t_cmb / 2.7255) ** 4
    nu = K.class_neutrino_params()
    n_ur = float(p.get("N_ur", nu.get("N_ur", K.NU_N_EFF)))
    if "N_eff" in p and "N_ur" not in p:
        n_ur = float(p["N_eff"]) - (K.NU_NEFF_PER_MASSIVE if "m_ncdm" in nu else 0.0)
    our = og * 0.22711 * n_ur
    m = float(nu.get("m_ncdm", 0.0)) if "m_ncdm" in nu else 0.0
    return og, our, m, t_cmb


def _hubble(z: np.ndarray, h: float, p: Dict[str, float], MODEL_func: Callable) -> np.ndarray:
    """H(z) / (100 km/s/Mpc) = h E(z) with the accurate early-universe components."""
    from Kosmulator_main import utils as U
    z = np.asarray(z, float)
    q = dict(p)
    q["H_0"] = 100.0 * h
    q.pop("Omega_m", None)
    q.pop("Omega_b", None)
    q = U.ensure_background_params(q)
    og, our, m, t_cmb = _species(q)
    ocb = float(q["Omega_bh^2"]) + float(q["Omega_dh^2"])
    # the model's late-time part: its E^2 minus its own matter and radiation
    from User_defined_modules import radiation_density
    Om, Or = float(q["Omega_m"]), float(radiation_density(q))
    X = np.zeros_like(z)
    low = z <= _Z_X_MAX
    if np.any(low):
        zl = z[low]
        E_model = np.asarray(MODEL_func(zl, q), float)
        X[low] = E_model ** 2 - Om * (1 + zl) ** 3 - Or * (1 + zl) ** 4
    if np.any(~low):
        X[~low] = _x_high(z[~low], q, Om, Or, MODEL_func)
    a = 1.0 / (1.0 + z)
    E2h2 = X * h * h + ocb * (1 + z) ** 3 + (og + our) * (1 + z) ** 4
    if m > 0:
        y = m * a / (_T_NCDM * t_cmb * _K_B_EV)
        o_nu_rel = og * 0.875 * _T_NCDM ** 4          # one massive species, relativistic limit
        E2h2 = E2h2 + o_nu_rel * (1 + z) ** 4 * _nu_ratio(y)
    return np.sqrt(E2h2)


def _x_high(z: np.ndarray, q: Dict[str, float], Om: float, Or: float, MODEL_func: Callable) -> np.ndarray:
    """X(z) above _Z_X_MAX: the power law through X(_Z_X_FIT) and X(_Z_X_MAX), or 0 (see _Z_X_MAX)."""
    za = np.array([_Z_X_FIT, _Z_X_MAX])
    try:
        E2 = np.asarray(MODEL_func(za, q), float) ** 2
    except Exception:
        return np.zeros_like(z)
    Xa = E2 - Om * (1 + za) ** 3 - Or * (1 + za) ** 4
    if not (np.all(np.isfinite(Xa)) and np.all(Xa > _X_REL_MIN * E2)):
        return np.zeros_like(z)
    n = np.log(Xa[1] / Xa[0]) / np.log((1 + za[1]) / (1 + za[0]))
    return Xa[1] * ((1 + z) / (1 + za[1])) ** n


def theta_mc(h: float, p: Dict[str, float], MODEL_func: Callable) -> float:
    """100 theta_MC at h for the parameters p (Omega_bh^2, Omega_dh^2 and the model's own)."""
    ob = float(p["Omega_bh^2"])
    om = float(p["Omega_bh^2"]) + float(p["Omega_dh^2"]) + K.omega_nu()   # CAMB: omega_m with the neutrinos
    g1 = 0.0783 * ob ** -0.238 / (1 + 39.5 * ob ** 0.763)
    g2 = 0.560 / (1 + 21.1 * ob ** 1.81)
    zstar = 1048.0 * (1 + 0.00124 * ob ** -0.738) * (1 + g1 * om ** g2)
    # D_M(z_*) in Mpc: Simpson in ln(1 + z)
    x = np.linspace(0.0, np.log1p(zstar), 2001)
    z = np.expm1(x)
    Hz = _hubble(z, h, p, MODEL_func)
    if not np.all(np.isfinite(Hz)) or np.any(Hz <= 0):
        return float("nan")
    DM = _C_KM_S / 100.0 * _simpson((1 + z) / Hz, x)
    # r_s: int da c_s / (a^2 H) = int dln a c_s / (a H), from a = 1e-8 to a_*
    astar = 1.0 / (1.0 + zstar)
    la = np.linspace(np.log(1e-8), np.log(astar), 2001)
    a = np.exp(la)
    Ha = _hubble(1.0 / a - 1.0, h, p, MODEL_func)
    cs = 1.0 / np.sqrt(3.0 * (1.0 + 3.0e4 * a * ob))
    rs = _C_KM_S / 100.0 * _simpson(cs / (a * Ha), la)
    return 100.0 * rs / DM


def _simpson(y: np.ndarray, x: np.ndarray) -> float:
    n = y.size - 1
    hstep = (x[-1] - x[0]) / n
    return float(hstep / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()))


def h_from_theta(p: Dict[str, float], MODEL_func: Callable, h_range=(0.2, 1.2), tol=1e-9) -> Optional[float]:
    """h with theta_mc(h) = p["100theta_MC"], or None when no h in h_range gives it."""
    from scipy.optimize import brentq
    target = float(p[THETA_PARAM])
    f = lambda h: theta_mc(h, p, MODEL_func) - target
    try:
        lo, hi = float(h_range[0]), float(h_range[1])
        flo, fhi = f(lo), f(hi)
        if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
            return None
        return float(brentq(f, lo, hi, xtol=tol, rtol=1e-12))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sampling with theta (constants.CMB_SAMPLE_THETA)
# ---------------------------------------------------------------------------
_H_CACHE: Dict[tuple, Optional[float]] = {}


def h_for_point(model_name: str, pd: Dict[str, float], h_range) -> Optional[float]:
    """h of one point (cached: the prior and each likelihood term of a point ask for it)."""
    try:
        key = (str(model_name), tuple(h_range),
               tuple(sorted((k, float(v)) for k, v in pd.items() if isinstance(v, (int, float, np.floating)))))
    except Exception:
        key = None
    if key is not None and key in _H_CACHE:
        return _H_CACHE[key]
    import User_defined_modules as UDM
    h = h_from_theta(pd, UDM.Get_model_function(str(model_name)), h_range=h_range)
    if key is not None:
        if len(_H_CACHE) > 20000:
            _H_CACHE.clear()
        _H_CACHE[key] = h
    return h


def h_range_of(cfg: Dict) -> tuple:
    """The H_0 prior box of a model's configuration, as h (the solver's range)."""
    lo, hi = (cfg.get("prior_limits_global", {}) or {}).get("H_0", (20.0, 100.0))
    return (float(lo) / 100.0, float(hi) / 100.0)


class _H0Rows:
    """Picklable: H_0 of a block of chain rows (for the worker pool kept open after sampling)."""

    def __init__(self, model_name: str, names, fixed: Dict[str, float], h_range):
        self.model_name, self.names, self.fixed, self.h_range = model_name, list(names), dict(fixed), tuple(h_range)

    def __call__(self, rows):
        out = np.full(len(rows), np.nan)
        for i, row in enumerate(np.asarray(rows, float)):
            pd = {**self.fixed, **{n: float(v) for n, v in zip(self.names, row)}}
            h = h_for_point(self.model_name, pd, self.h_range)
            if h is not None:
                out[i] = 100.0 * h
        return out


def _thin_rows(n: int, nw: int, max_rows: int) -> np.ndarray:
    """Rows to keep of n step-major rows (nw walkers per step): every k-th step, all its walkers."""
    if n <= max_rows:
        return np.arange(n)
    if nw > 1 and n % nw == 0:
        k = -(-(n // nw) // max(1, max_rows // nw))
        return (np.arange(0, n // nw, k)[:, None] * nw + np.arange(nw)[None, :]).ravel()
    return np.arange(0, n, -(-n // max_rows))


def convert_theta_runs(All_Samples: Dict, CONFIG: Dict, max_samples: int = 100000) -> None:
    """
    Before the statistics: in every group that sampled 100 theta_MC, the theta column of the samples
    becomes H_0 (solved per sample), the parameter list says H_0, and the group gets the H_0 prior box.
    More than max_samples samples are thinned first by whole steps (every k-th step, all walkers), so
    the rows stay step-major (the corner plot's chains per walker) and the saved log-likelihoods
    (entry["loglike"]) are thinned with them and stay aligned. CONFIG[model]["theta_converted"]
    records the groups, so the convergence report still reads the chain file's theta column.
    """
    from Kosmulator_main import utils as U
    for model_name, Samples in (All_Samples or {}).items():
        cfg = CONFIG.get(model_name, {})
        for gi, names in enumerate(list(cfg.get("parameters", []))):
            if THETA_PARAM not in names:
                continue
            key = U.generate_label(cfg["observations"][gi], config_model=cfg, obs_index=gi)
            if key not in Samples:
                continue
            entry = Samples[key]
            arr = np.asarray(entry["samples"] if isinstance(entry, dict) else entry, float)
            if arr.ndim != 2 or arr.shape[1] != len(names):
                continue
            n0 = arr.shape[0]
            ll = entry.get("loglike") if isinstance(entry, dict) else None
            ll = np.asarray(ll, float).reshape(-1) if ll is not None else None
            if ll is not None and ll.shape[0] != n0:
                ll = None                              # not aligned with the samples: dropped, as on loading
            sel = _thin_rows(n0, int(cfg.get("nwalker", 0) or 0), int(max_samples))
            arr = arr[sel]
            ll = ll[sel] if ll is not None else None
            fixed = dict((cfg.get("fixed_params_by_group", {}) or {}).get(gi, {}))
            task = _H0Rows(model_name, names, fixed, h_range_of(cfg))
            if getattr(U, "_POST_POOL", None) is not None and arr.shape[0] > 200:
                chunks = np.array_split(arr, max(1, min(arr.shape[0] // 200, 64)))
                H0 = np.concatenate(U.post_map(task, chunks))
            else:
                H0 = task(arr)
            ok = np.isfinite(H0)                       # every accepted point had a solution (log_prior_all)
            j = names.index(THETA_PARAM)
            arr = arr[ok].copy()
            arr[:, j] = H0[ok]
            if isinstance(entry, dict):
                entry = dict(entry)
                entry["samples"] = arr
                entry["loglike"] = ll[ok] if ll is not None else None
                Samples[key] = entry
            else:
                Samples[key] = arr
            new_names = list(names)
            new_names[j] = "H_0"
            cfg["parameters"][gi] = new_names
            box = (cfg.get("prior_limits_global", {}) or {}).get("H_0", (20.0, 100.0))
            pl = dict(cfg["prior_limits"][gi])
            pl.pop(THETA_PARAM, None)
            pl["H_0"] = tuple(box)
            cfg["prior_limits"][gi] = pl
            ref = np.asarray(cfg["reference_values"][gi], float).copy()
            ref[j] = float(np.median(arr[:, j])) if arr.size else 67.0
            cfg["reference_values"][gi] = ref
            cfg.setdefault("theta_converted", {})[gi] = True
            thin = f", thinned by whole steps from {n0}" if len(sel) < n0 else ""
            print(f"[theta] {model_name} | {key}: 100theta_MC -> H_0 for {arr.shape[0]} samples{thin} "
                  f"(H_0 = {np.median(arr[:, j]):.2f}, {int((~ok).sum())} outside the H_0 box dropped)", flush=True)
