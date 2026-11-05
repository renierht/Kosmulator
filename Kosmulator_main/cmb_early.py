"""
ThetaStar_CMB (constants.CMB_EARLY_TAG): the early-universe compression of the CMB as a Gaussian
prior on (theta_*, omega_b, omega_bc), DESI DR2's "(theta_*, omega_b, omega_bc)_CMB priors"
(arXiv:2503.14738, Sect. IV B and Appendix A), made from the Planck PR4 CamSpec chains of Lemos &
Lewis (2023, arXiv:2302.12911) that are marginalised over the late-time ISW effect and CMB lensing
(github.com/cmbant/PlanckEarlyLCDM, spline_planck_PR4_TTTEEE_lowE_lensing_ISW).

    mean = (0.010410274, 0.022232079, 0.14207901)
    C    = 1e-9 [[0.006621, 0.12444, -1.1929], [0.12444, 21.344, -94.001], [-1.1929, -94.001, 1488.4]]

The covariance is DESI's eq. (A2), which equals the chains' (the README's Gaussian in theta_*, omega_b,
omega_c, transformed). The mean is the chains' (their .margestats: thetastar/100, ombh2, omegamh2),
which DESI's eq. (A1) prints rounded as (0.01041, 0.02223, 0.14208); the rounding of theta_* alone
is 0.1 sigma. The chains' third mean, 0.14208, is their Omega_m h^2 with the 0.06 eV neutrino (CAMB's
omegamh2); their omega_b + omega_c is 0.14143, half a sigma lower. DESI's notation says omega_bc =
omega_b + omega_c, and applied that way the prior gives DESI's Table VI line (Delta chi^2_MAP -8.0
with DESI DR2 BAO in w0waCDM against LCDM; -6.6 with Omega_m h^2), so the default follows DESI:
constants.CMB_EARLY_OMEGA = "bc" compares the third mean with omega_b + omega_c (CLASS's omega_b +
omega_cdm). "m" compares it with Omega_m h^2 including the neutrino, which is what the chains measure
(the prior then sits half a sigma lower in omega_b + omega_c than DESI's).

theta_* = r_s(z_*) / D_M(z_*) is CAMB's (the chains are CAMB's): z_* where the optical depth is one,
and r_s(z_*). CLASS 3.3's z_star is another definition (1085.1 against CAMB's 1089.9 at Planck's best
fit, which moves theta_* by 0.3%, twelve times the prior's width), so z_* and r_s(z_*) come from CAMB
on a grid in (omega_b, omega_c) for Kosmulator's neutrinos (one 0.06 eV state, N_eff 3.044, as DESI),
shipped as Observations/CMB_early/zstar_rstar_camb.npz (made with make_grid(); CAMB is needed only to
remake it) and splined. r_s(z_*) is scaled by the model's sound horizon over that of LCDM (the same
integral with each background); D_M(z_*) is integrated with the model's E(z) and the accurate early
components (theta_mc._hubble). Against CAMB's thetastar (PPF dark energy): 0.006 sigma of the prior
(1.5e-8 in theta_*) for LCDM and for w0wa with w0 + wa <= -0.5, 0.1 sigma with w0 + wa = -0.3. Where
dark energy is a larger part of H^2 at z_* (about 50 sigma times that fraction: 2 sigma at 4%, w0 + wa
near -0.2), z_* itself moves (CAMB: 1090 to 1094) and the grid's LCDM z_* is no longer right; such points
get a warning once per process (they lie far outside DESI's w0waCDM posterior). Kit section DESIDR2.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Dict, Optional

import numpy as np

from Kosmulator_main import constants as K

logger = logging.getLogger("Kosmulator.cmb_early")

TAG = K.CMB_EARLY_TAG
#: (theta_*, omega_b, third): the chains' means; DESI DR2 eq. (A1) prints them rounded (see above)
MEAN = np.array([0.010410274, 0.022232079, 0.14207901])
MEAN_DESI_PRINTED = np.array([0.01041, 0.02223, 0.14208])
COV = 1e-9 * np.array([[0.006621, 0.12444, -1.1929],
                        [0.12444, 21.344, -94.001],
                        [-1.1929, -94.001, 1488.4]])
_C_KM_S = 299792.458
#: The grid in (omega_b, omega_c): wide enough for BBN- and CMB-like omega_b and any omega_c a
#: late-time fit with this prior reaches (the prior fixes omega_m to 0.9%)
_OB = np.linspace(0.0170, 0.0280, 23)
_OC = np.linspace(0.0900, 0.1600, 29)

_SPLINES = None
#: Fraction of H^2 from dark energy at z_* above which theta_* is approximate (z_* from the LCDM grid):
#: about 0.1 sigma of the prior there
EARLY_DE_WARN = 2e-3
_WARNED_EARLY_DE = False
GRID_FILE = os.path.join("CMB_early", "zstar_rstar_camb.npz")


def make_grid(path: Optional[str] = None) -> str:
    """Compute z_* and r_s(z_*) with CAMB over the grid (Kosmulator's neutrinos) and write the file."""
    import camb
    path = path or os.path.join(K.OBSERVATIONS_BASE, GRID_FILE)
    zs = np.empty((_OB.size, _OC.size))
    rs = np.empty_like(zs)
    massive = K.NEUTRINOS == "massive"
    for i, ob in enumerate(_OB):
        for j, oc in enumerate(_OC):
            pars = camb.set_params(H0=67.0, ombh2=float(ob), omch2=float(oc), mnu=K.NU_MASS_EV if massive else 0.0,
                                   num_massive_neutrinos=1 if massive else 0, nnu=K.NU_N_EFF, TCMB=K.T_CMB_DEFAULT)
            d = camb.get_background(pars).get_derived_params()
            zs[i, j], rs[i, j] = d["zstar"], d["rstar"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, ob=_OB, oc=_OC, z_star=zs, rs_star=rs, camb_version=np.array(camb.__version__),
             neutrinos=np.array(f"{K.NEUTRINOS} {K.NU_MASS_EV} {K.NU_N_EFF}"))
    return path


def _splines():
    """Splines of CAMB's z_* and r_s(z_*) over (omega_b, omega_c), from the shipped grid file."""
    global _SPLINES
    if _SPLINES is not None:
        return _SPLINES
    from scipy.interpolate import RectBivariateSpline
    path = os.path.join(K.OBSERVATIONS_BASE, GRID_FILE)
    if not os.path.exists(path):
        try:
            make_grid(path)
            logger.info("CMB_early: CAMB grid of z_* and r_s(z_*) written to %s", path)
        except ImportError as e:
            raise FileNotFoundError(f"CMB_early needs {path} (in the repository), or CAMB to make it ({e})") from e
    with np.load(path) as f:
        g = {k: f[k] for k in f.files}
    want = f"{K.NEUTRINOS} {K.NU_MASS_EV} {K.NU_N_EFF}"
    if str(g.get("neutrinos", want)) != want:
        logger.warning("CMB_early: the grid was made for neutrinos '%s', this run uses '%s'; the prior itself "
                       "(DESI DR2) assumes one 0.06 eV state and N_eff = 3.044", g.get("neutrinos"), want)
    _SPLINES = (RectBivariateSpline(g["ob"], g["oc"], g["z_star"], kx=3, ky=3),
                RectBivariateSpline(g["ob"], g["oc"], g["rs_star"], kx=3, ky=3),
                (float(g["ob"][0]), float(g["ob"][-1]), float(g["oc"][0]), float(g["oc"][-1])))
    return _SPLINES


def _sound_horizon(zstar: float, h: float, q: Dict[str, float], MODEL_func: Callable) -> float:
    """r_s(z_*) in Mpc with the background of MODEL_func (theta_mc._hubble), c_s with R = 3 rho_b / 4 rho_gamma."""
    from Kosmulator_main import theta_mc as TM
    ob = float(q["Omega_bh^2"])
    astar = 1.0 / (1.0 + zstar)
    la = np.linspace(np.log(1e-8), np.log(astar), 2001)
    a = np.exp(la)
    Ha = TM._hubble(1.0 / a - 1.0, h, q, MODEL_func)
    R = 0.75 * ob / TM._OMEGA_GAMMA_2P7255 * a
    cs = 1.0 / np.sqrt(3.0 * (1.0 + R))
    return _C_KM_S / 100.0 * TM._simpson(cs / (a * Ha), la)


def theta_star(p: Dict[str, float], MODEL_func: Callable) -> float:
    """theta_* (not 100 theta_*) at the point p (H_0, Omega_bh^2 and Omega_m or Omega_dh^2, the model's own)."""
    from Kosmulator_main import theta_mc as TM
    import User_defined_modules as UDM
    q = _densities(p)
    h = float(q["H_0"]) / 100.0
    ob = float(q["Omega_bh^2"])
    oc = float(q["Omega_dh^2"])
    sz, sr, (b0, b1, c0, c1) = _splines()
    if not (b0 <= ob <= b1 and c0 <= oc <= c1):
        return float("nan")
    zstar = float(sz(ob, oc)[0, 0])
    rstar = float(sr(ob, oc)[0, 0])
    if MODEL_func is not UDM.LCDM_MODEL_vectorised:
        # the model's dark energy before recombination: its sound horizon over LCDM's
        r_model = _sound_horizon(zstar, h, q, MODEL_func)
        r_lcdm = _sound_horizon(zstar, h, q, UDM.LCDM_MODEL_vectorised)
        if not (np.isfinite(r_model) and np.isfinite(r_lcdm) and r_lcdm > 0):
            return float("nan")
        rstar *= r_model / r_lcdm
    x = np.linspace(0.0, np.log1p(zstar), 2001)
    z = np.expm1(x)
    Hz = TM._hubble(z, h, q, MODEL_func)
    if not np.all(np.isfinite(Hz)) or np.any(Hz <= 0):
        return float("nan")
    if MODEL_func is not UDM.LCDM_MODEL_vectorised:
        _note_early_de(1.0 - (float(TM._hubble(np.array([zstar]), h, q, UDM.LCDM_MODEL_vectorised)[0]) / Hz[-1]) ** 2)
    DM = _C_KM_S / 100.0 * TM._simpson((1 + z) / Hz, x)
    return rstar / DM


def _note_early_de(frac: float) -> None:
    """Warn once per process when dark energy is more than EARLY_DE_WARN of H^2 at z_*."""
    global _WARNED_EARLY_DE
    if frac > EARLY_DE_WARN and not _WARNED_EARLY_DE:
        _WARNED_EARLY_DE = True
        logger.warning("%s: a point with %.1f%% of H^2 from dark energy at z_*: theta_* is approximate there "
                       "(z_* from the LCDM grid; about %.1f sigma of the prior). Shown once.",
                       K.CMB_EARLY_TAG, 100 * frac, 50 * frac)


def _densities(p: Dict[str, float]) -> Dict[str, float]:
    """The point with Omega_bh^2, Omega_dh^2 (CLASS's omega_cdm) and Omega_m filled in."""
    from Kosmulator_main import utils as U
    q = U.ensure_background_params(dict(p))
    if "Omega_dh^2" not in q and all(k in q for k in ("Omega_m", "H_0", "Omega_bh^2")):
        h = float(q["H_0"]) / 100.0
        q["Omega_dh^2"] = float(q["Omega_m"]) * h * h - float(q["Omega_bh^2"]) - K.omega_nu()
    return q


def omegas(p: Dict[str, float]):
    """(omega_b, the prior's third parameter) of a point: omega_b + omega_c (constants.CMB_EARLY_OMEGA "bc",
    DESI's) or Omega_m h^2 = omega_b + omega_c + omega_nu ("m", the chains')."""
    q = _densities(p)
    obc = float(q["Omega_bh^2"]) + float(q["Omega_dh^2"])
    return float(q["Omega_bh^2"]), obc + (K.omega_nu() if K.CMB_EARLY_OMEGA == "m" else 0.0)


def chi2(p: Dict[str, float], MODEL_func: Callable) -> float:
    """chi^2 of the (theta_*, omega_b, omega_bc) prior at p (inf where theta_* cannot be computed)."""
    th = theta_star(p, MODEL_func)
    if not np.isfinite(th):
        return float("inf")
    ob, obc = omegas(p)
    d = np.array([th, ob, obc]) - MEAN
    return float(d @ np.linalg.solve(COV, d))


def check_available() -> None:
    """Read (or make, with CAMB) the grid at start-up, before any pool."""
    _splines()
