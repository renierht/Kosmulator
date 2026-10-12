"""
Ground-based CMB likelihoods (ACT DR6, SPT-3G D1) and the Planck + ACT (+ SPT) combinations
(item 57). Tags and settings: constants.GROUND_CMB_TAGS and the block above it.

Every likelihood of a group takes its spectra from the point's single lensed CLASS run
(Statistical_packages.get_cmb_spectra, l_max = constants.CMB_CLASS_LMAX, raised to 8501 for
ACT and 4095 for SPT by MCMC_setup), as the Planck likelihoods do.

- ACT DR6 and SPT-3G D1: candl (Balkenhol et al. 2024, arXiv:2401.13433) with the
  foreground-marginalised ("lite") data sets, candl_data.ACT_DR6_TTTEEE and
  spt_candl_data.SPT3G_D1_TnE_lite. candl's internal priors are removed; Kosmulator applies
  the same priors itself (constants.cmb_gaussian_priors), so that a combination does not
  apply a tau prior twice or one next to low-l EE data.
- CMB_planck_ACTcut: Planck 2018 plik_lite (the 613 band powers in Observations/
  plik_lite_v22_TTTEEE.clik) without the bins above l = 1000 in TT and l = 600 in TE and
  EE, as ACT DR6's PlanckActCut (act_dr6_cmbonly, a cut of Cobaya's Python plik_lite,
  E. Calabrese and A. Lewis).
- CMB_lowl_sroll2: the SRoll2 low-l EE likelihood as a probability table in C_l^EE,
  l = 2 - 29 (Pagano et al. 2020, arXiv:1908.09856; Cobaya's planck_2018_lowl.EE_sroll2).
- CMB_PACT and CMB_SPA evaluate their parts in turn on the same spectra.
- CMB_lensing_ACTPlanck (item 61): ACT DR6 + Planck PR4 lensing power spectrum through
  act_dr6_lenslike (constants.ACT_LENS_VARIANT), on the same CLASS run: C_L^kk from C_L^phiphi,
  and the lensed TT, TE, EE, BB for the likelihood corrections when the group has primary CMB
  (binned once into a small cached form, _act_lens_compact).
"""
from __future__ import annotations

import contextlib
import importlib
import io
import logging
import os
from typing import Dict, Optional

import numpy as np

from Kosmulator_main import constants as K

logger = logging.getLogger("Kosmulator.ground_cmb")

_CANDL_DATA = {
    "CMB_ACT": ("candl_data", "ACT_DR6_TTTEEE", "pip install candl-like candl_data"),
    "CMB_SPT": ("spt_candl_data", "SPT3G_D1_TnE_lite",
                "git clone https://github.com/SouthPoleTelescope/spt_candl_data; cd spt_candl_data; pip install ."),
}
_LIKES: Dict[str, object] = {}          # per process


def candl_like(tag: str):
    """The candl likelihood of a tag, loaded once per process, without its internal priors."""
    if tag not in _LIKES:
        mod_name, attr, how = _CANDL_DATA[tag]
        try:
            import candl
            mod = importlib.import_module(mod_name)
        except ImportError as e:
            raise ImportError(f"{tag} needs candl and {mod_name} ({e}); install with: {how}") from e
        with contextlib.redirect_stdout(io.StringIO()):
            like = candl.Like(getattr(mod, attr))
        like.priors = []
        _LIKES[tag] = like
    return _LIKES[tag]


class PlikLiteCut:
    """Planck 2018 plik_lite TT,TE,EE in Python, without the bins outside l_min - l_max."""

    NBINS = (215, 199, 199)     # TT, TE, EE (plik_lite_v22.dataset)
    LMIN_OFFSET = 30

    def __init__(self, folder: str, lmax_cuts=(2508, 2508, 2508), lmin_cuts=(0, 0, 0)):
        from scipy.io import FortranFile
        ext = os.path.join(folder, "clik", "lkl_0", "_external")
        data = np.loadtxt(os.path.join(ext, "cl_cmb_plik_v22.dat"))
        self.blmin = np.loadtxt(os.path.join(ext, "blmin.dat")).astype(int) + self.LMIN_OFFSET
        self.blmax = np.loadtxt(os.path.join(ext, "blmax.dat")).astype(int) + self.LMIN_OFFSET
        w = np.loadtxt(os.path.join(ext, "bweight.dat"))
        ls = np.arange(len(w)) + self.LMIN_OFFSET
        self.weights = np.hstack((np.zeros(self.LMIN_OFFSET), w * 2 * np.pi / ls / (ls + 1)))  # acts on D_l
        n = sum(self.NBINS)
        f = FortranFile(os.path.join(ext, "c_matrix_plik_v22.dat"), "r")
        cov = f.read_reals(dtype=float).reshape((n, n))
        cov = np.tril(cov) + np.tril(cov, -1).T
        if data.shape[0] != n:
            raise ValueError(f"plik_lite data has {data.shape[0]} rows, expected {n}")
        self.used = []                    # per spectrum: (bin indices, offset)
        idx, off = [], 0
        for nb, lo, hi in zip(self.NBINS, lmin_cuts, lmax_cuts):
            b = np.arange(nb)
            keep = b[(self.blmin[b] >= lo) & (self.blmax[b] <= hi)]
            self.used.append(keep)
            idx.append(keep + off)
            off += nb
        self.idx = np.hstack(idx)
        self.n_data = int(self.idx.size)
        self.x = data[self.idx, 1]
        self.invcov = np.linalg.inv(cov[np.ix_(self.idx, self.idx)])
        self.lmax = int(max(self.blmax[u].max() for u in self.used if u.size))

    def loglike(self, dl: Dict[str, np.ndarray], A_planck: float = 1.0) -> float:
        """dl: D_l [muK^2] from l = 0 for 'TT', 'TE', 'EE'."""
        th = []
        for spec, keep in zip(("TT", "TE", "EE"), self.used):
            d = dl[spec]
            for i in keep:
                th.append(np.dot(d[self.blmin[i]:self.blmax[i] + 1], self.weights[self.blmin[i]:self.blmax[i] + 1]))
        diff = self.x - np.asarray(th) / A_planck ** 2
        return -0.5 * float(diff @ self.invcov @ diff)


class TableEE:
    """Low-l EE likelihood tabulated in D_l^EE (SimAll format: 3000 steps of 1e-4 muK^2, l = 2 - 29)."""

    LMIN, LMAX, STEP = 2, 29, 1e-4

    def __init__(self, path: str):
        self.prob = np.loadtxt(path)
        if self.prob.shape[1] != self.LMAX - self.LMIN + 1:
            raise ValueError(f"{path}: {self.prob.shape[1]} columns, expected {self.LMAX - self.LMIN + 1}")

    def loglike(self, dl_ee: np.ndarray, A_planck: float = 1.0) -> float:
        i = (np.asarray(dl_ee[self.LMIN:self.LMAX + 1]) / (A_planck ** 2 * self.STEP)).astype(int)
        if np.any(i < 0) or np.any(i >= self.prob.shape[0]):
            return -np.inf                 # D_l^EE outside the table: rejected, as in Cobaya
        return float(np.take_along_axis(self.prob, i[np.newaxis, :], 0).sum())


_NATIVE: Dict[str, object] = {}


def planck_act_cut() -> PlikLiteCut:
    if "pcut" not in _NATIVE:
        folder = os.path.join(K.OBSERVATIONS_BASE, "plik_lite_v22_TTTEEE.clik")
        _NATIVE["pcut"] = PlikLiteCut(folder, lmax_cuts=K.PLANCK_ACT_CUT_LMAX)
    return _NATIVE["pcut"]


def sroll2() -> TableEE:
    if "sroll2" not in _NATIVE:
        path = os.path.join(K.OBSERVATIONS_BASE, K.SROLL2_TABLE_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is missing (SRoll2 low-l EE table, Cobaya's planck_native_data "
                                    "release v1, planck_sroll2_lowE.zip)")
        _NATIVE["sroll2"] = TableEE(path)
    return _NATIVE["sroll2"]


# ---------------------------------------------------------------------------
# CMB lensing beyond Planck 2018 (item 61)
# ---------------------------------------------------------------------------
_ACT_LENS: Dict[bool, dict] = {}
ACT_LENS_TRIM_LMAX = 2998          # act_dr6_lenslike uses theory spectra to this l
_ACT_LENS_COMPACT_FORMAT = 1       # raise when _act_lens_compact changes (new cache file)
_CMB_ORDER = ("tt", "ee", "bb", "te")   # order of dAL_dC's first axis in act_dr6_lenslike


def act_lens_dir() -> str:
    """Folder with the ACT DR6 lensing data: Observations/<ACT_LENS_DIR>/<version>/, else the
    act_dr6_lenslike package's own data folder (filled by act_dr6_lenslike.get_data())."""
    try:
        import act_dr6_lenslike as alike
    except ImportError as e:
        raise ImportError(f"CMB_lensing_ACTPlanck needs act_dr6_lenslike ({e}); install with: "
                          "pip install act_dr6_lenslike") from e
    cands = [os.path.join(K.OBSERVATIONS_BASE, K.ACT_LENS_DIR, K.ACT_LENS_VERSION),
             os.path.join(os.path.dirname(os.path.abspath(alike.__file__)), "data", K.ACT_LENS_VERSION)]
    for d in cands:
        if os.path.isfile(os.path.join(d, "clkk_bandpowers_act.txt")):
            return d
    raise FileNotFoundError(
        "ACT DR6 lensing data not found. Download https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/"
        f"likelihood/data/ACT_dr6_likelihood_{K.ACT_LENS_VERSION}.tgz and unpack it so that "
        f"{cands[0]}/clkk_bandpowers_act.txt exists, or run: python -c \"import act_dr6_lenslike as a; a.get_data()\"")


def act_lens_package_data(lens_only: bool) -> dict:
    """act_dr6_lenslike.load_data for constants.ACT_LENS_VARIANT, as the package's own Cobaya
    likelihood calls it: lens_only (no primary CMB in the group) gives the CMB-marginalised
    covariance without corrections; otherwise the plain covariance with the normalisation and N1
    corrections, whose derivative matrices take 1.4 GB in memory (12 s to read)."""
    import warnings
    import act_dr6_lenslike as alike
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return alike.load_data(
            K.ACT_LENS_VARIANT, ddir=act_lens_dir(), lens_only=bool(lens_only), like_corrections=not lens_only,
            nsims_act=K.ACT_LENS_NSIMS[0], nsims_planck=K.ACT_LENS_NSIMS[1],
            trim_lmax=ACT_LENS_TRIM_LMAX, version=K.ACT_LENS_VERSION)


def _act_lens_compact(d: dict) -> Dict[str, np.ndarray]:
    """The binned theory of act_dr6_lenslike.generic_lnlike (default corrections) as a few
    (bins x L) matrices. get_corrected_clkk is linear in the spectra, so the band powers are
        B C^kk + B dN1_kk (C^kk - fid_kk) + sum_s [B dN1_s - 2 B diag(w) dAL_dC_s] (C^s - fid_s),
    with w = fid_kk / fAL for L >= 2 (fid_kk below), s = TT, EE, BB, TE and B the ACT (and Planck)
    binning matrix with that block's dN1, dAL_dC and fAL. 2.9 MB instead of 1.4 GB per process;
    the kit compares it with generic_lnlike."""
    blocks = [("", np.asarray(d["binmat_act"], float))]
    if d["include_planck"]:
        blocks.append(("_planck", np.asarray(d["binmat_planck"], float)))
    out = {"B": np.vstack([b for _, b in blocks]),
           "data": np.asarray(d["data_binned_clkk"], float),
           "cinv": np.asarray(d["cinv"], float),
           "corr": np.array(bool(d["likelihood_corrections"]))}
    if d["likelihood_corrections"]:
        fid_kk = np.asarray(d["fiducial_cl_kk"], float)
        ls = np.arange(fid_kk.size)
        out["fid_kk"] = fid_kk
        out["N1_kk"] = np.vstack([b @ d[f"dN1_kk{s}"] for s, b in blocks])
        for i, sp in enumerate(_CMB_ORDER):
            rows = []
            for s, b in blocks:
                w = fid_kk.copy()
                w[ls >= 2] = fid_kk[ls >= 2] / np.asarray(d[f"fAL{s}"], float)[:fid_kk.size][ls >= 2]
                rows.append(b @ d[f"dN1_{sp}{s}"] - 2.0 * (b * w[np.newaxis, :]) @ d[f"dAL_dC{s}"][i])
            out[f"M_{sp}"] = np.vstack(rows)
            out[f"fid_{sp}"] = np.asarray(d[f"fiducial_cl_{sp}"], float)
    return out


def _act_lens_cache_files() -> list:
    """Cache of the compact corrected likelihood: Observations/<ACT_LENS_DIR>/.cache/, else (when
    that cannot be written) the system's temporary folder; named by the variant, the settings,
    the package version and the size and time of every data file."""
    import tempfile
    import hashlib
    import act_dr6_lenslike as alike
    ddir = act_lens_dir()
    h = hashlib.sha1(repr((_ACT_LENS_COMPACT_FORMAT, getattr(alike, "__version__", "?"), K.ACT_LENS_VARIANT,
                           K.ACT_LENS_VERSION, tuple(K.ACT_LENS_NSIMS), ACT_LENS_TRIM_LMAX)).encode())
    for root, dirs, files in os.walk(ddir):
        dirs.sort()
        for f in sorted(files):
            st = os.stat(os.path.join(root, f))
            h.update(f"{os.path.relpath(os.path.join(root, f), ddir)}:{st.st_size}:{st.st_mtime_ns};".encode())
    name = f"{K.ACT_LENS_VARIANT}_corr_{h.hexdigest()[:16]}.npz"
    return [os.path.join(K.OBSERVATIONS_BASE, K.ACT_LENS_DIR, ".cache", name),
            os.path.join(tempfile.gettempdir(), "kosmulator_act_dr6_lenslike", name)]


def _write_npz(path: str, arrays: Dict[str, np.ndarray]) -> None:
    """Write arrays to path through a temporary file in the same folder (complete or absent)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def act_lens_data(lens_only: bool) -> Dict[str, np.ndarray]:
    """The compact likelihood of constants.ACT_LENS_VARIANT, once per process and mode. With the
    corrections (primary CMB in the group) it is read from the cache file, which the first
    process writes (start-up, MCMC_setup), so spawned workers do not each read the 2.8 GB."""
    lens_only = bool(lens_only)
    if lens_only not in _ACT_LENS:
        if lens_only:
            _ACT_LENS[True] = _act_lens_compact(act_lens_package_data(True))
        else:
            paths = _act_lens_cache_files()
            c = None
            for path in paths:
                if c is None and os.path.isfile(path):
                    try:
                        with np.load(path) as z:
                            c = {k: z[k] for k in z.files}
                    except Exception as e:
                        logger.warning("CMB_lensing_ACTPlanck: cache %s unreadable (%s)", path, e)
            if c is None:
                logger.info("CMB_lensing_ACTPlanck: reading the ACT DR6 + Planck PR4 correction matrices "
                            "(2.8 GB, once) for the cache %s", paths[0])
                full = act_lens_package_data(False)
                c = _act_lens_compact(full)
                del full
                import gc
                gc.collect()
                errors = []
                for path in paths:
                    try:
                        _write_npz(path, c)
                        break
                    except OSError as e:
                        errors.append(f"{path}: {e}")
                else:
                    logger.warning("CMB_lensing_ACTPlanck: could not write the cache (%s); every pool worker "
                                   "will read the 2.8 GB itself and hold 1.4 GB", "; ".join(errors))
            _ACT_LENS[False] = c
    return _ACT_LENS[lens_only]


def act_lens_loglike(cl: Dict[str, np.ndarray], lens_only: bool) -> float:
    """ln L of ACT DR6 + Planck PR4 lensing for CLASS's dimensionless C_l (pp, and tt, te, ee, bb):
    act_dr6_lenslike.generic_lnlike in the compact form of _act_lens_compact."""
    import act_dr6_lenslike as alike
    pp = np.asarray(cl["pp"], float)
    if pp.size < ACT_LENS_TRIM_LMAX + 2:                # standardize keeps l = 0 - trim_lmax + 1
        raise ValueError(f"CMB_lensing_ACTPlanck needs CLASS spectra to l = {ACT_LENS_TRIM_LMAX + 1}, "
                         f"got {pp.size - 1}")
    ell = np.arange(pp.size, dtype=float)
    c = act_lens_data(lens_only)
    kk = alike.standardize(ell, alike.pp_to_kk(pp, ell), ACT_LENS_TRIM_LMAX)   # [L (L + 1)]^2 C_L^phiphi / 4
    th = c["B"] @ kk
    if bool(c["corr"]):
        t2 = (K.T_CMB_DEFAULT * 1e6) ** 2                                       # dimensionless -> muK^2
        th = th + c["N1_kk"] @ (kk - c["fid_kk"])
        for sp in _CMB_ORDER:
            cs = alike.standardize(ell, np.asarray(cl[sp], float)[:pp.size] * t2, ACT_LENS_TRIM_LMAX)
            th = th + c[f"M_{sp}"] @ (cs - c[f"fid_{sp}"])
    delta = c["data"] - th
    return -0.5 * float(delta @ c["cinv"] @ delta)


def lensing_loglike(tag: str, pd: Dict[str, float], model_name: str, lens_only: bool,
                    floor: float = -1e10) -> float:
    """ln L of a lensing tag of item 61 on the point's single CLASS run."""
    from Kosmulator_main import Class_run as CR, Statistical_packages as SP
    if tag not in K.EXT_LENSING_TAGS:
        raise KeyError(tag)
    try:
        CR.ensure_class_ready(model_name)
    except Exception as e:
        logger.warning("ensure_class_ready(%s) failed (%s); proceeding anyway", model_name, e)
    cl = SP.get_cmb_spectra(pd, model_name)
    if cl is None or not isinstance(cl, dict):
        return float(floor)
    try:
        v = act_lens_loglike(cl, lens_only)
    except (ImportError, FileNotFoundError, ValueError):
        raise
    except Exception as e:
        logger.error("%s: likelihood failed: %s", tag, e)
        return float(floor)
    return float(v) if np.isfinite(v) else float(floor)


def n_data(tag: str) -> int:
    """Data points of a ground-based tag (a combination: the sum of its parts) or a lensing tag."""
    if tag == "CMB_lensing_ACTPlanck":
        return int(act_lens_data(True)["data"].size)        # the same band powers in both modes
    if tag in K.CMB_BUNDLES:
        return sum(n_data(c) for c in K.cmb_components(tag))
    if tag == "CMB_planck_ACTcut":
        return planck_act_cut().n_data
    if tag in K.GROUND_N_DATA:
        return int(K.GROUND_N_DATA[tag])
    return int(K.planck_n_data(tag))


def _dl(cl: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """D_l [muK^2] from l = 0 from CLASS's dimensionless lensed C_l."""
    tcmb2 = (K.T_CMB_DEFAULT * 1e6) ** 2
    out = {}
    for s in ("tt", "te", "ee"):
        c = np.asarray(cl[s], float)
        ell = np.arange(c.size)
        out[s.upper()] = ell * (ell + 1) / (2 * np.pi) * c * tcmb2
    return out


def _candl_loglike(tag: str, dl: Dict[str, np.ndarray], pd: Dict[str, float]) -> float:
    like = candl_like(tag)
    lo, hi = int(like.ell_min), int(like.ell_max)
    if dl["TT"].size <= hi:
        raise ValueError(f"{tag} needs CLASS spectra to l = {hi}, got {dl['TT'].size - 1} "
                         "(constants.CMB_CLASS_LMAX; MCMC_setup sets it from the run's groups)")
    pars = {"Dl": {s: dl[s][lo:hi + 1] for s in ("TT", "TE", "EE")}}
    if tag == "CMB_ACT":
        # one calibration for Planck and ACT when both are in the group (P-ACT, SPA)
        pars["A_act"] = float(pd.get("A_act", pd.get("A_planck", 1.0)))
        pars["P_act"] = float(pd.get("P_act", 1.0))
    else:
        pars["Tcal"] = float(pd.get("Tcal", 1.0))
        pars["Ecal"] = float(pd.get("Ecal", 1.0))
    pars["tau"] = float(pd.get("tau_reio", 0.054))     # candl keys; no prior on it here
    return float(like.log_like(pars))


def component_loglike(tag: str, cl: Dict[str, np.ndarray], pd: Dict[str, float], model_name: str) -> float:
    """ln L of one ground-based or native Planck likelihood on the point's spectra."""
    if tag in ("CMB_ACT", "CMB_SPT"):
        return _candl_loglike(tag, _dl(cl), pd)
    if tag == "CMB_planck_ACTcut":
        return planck_act_cut().loglike(_dl(cl), float(pd.get("A_planck", 1.0)))
    if tag == "CMB_lowl_sroll2":
        return sroll2().loglike(_dl(cl)["EE"], float(pd.get("A_planck", 1.0)))
    if tag == "CMB_lowl_TT":
        from Kosmulator_main import Statistical_packages as SP
        return float(SP.cmb_lowlTT_loglike(pd, model_name))
    raise KeyError(tag)


def loglike(tag: str, pd: Dict[str, float], model_name: str, floor: float = -1e10) -> float:
    """ln L of a ground-based tag (a combination: the sum over its parts), one CLASS run per point."""
    from Kosmulator_main import Class_run as CR, Statistical_packages as SP
    try:
        CR.ensure_class_ready(model_name)
    except Exception as e:
        logger.warning("ensure_class_ready(%s) failed (%s); proceeding anyway", model_name, e)
    cl = SP.get_cmb_spectra(pd, model_name)
    if cl is None or not isinstance(cl, dict):
        return float(floor)
    total = 0.0
    for c in K.cmb_components(tag):
        try:
            v = component_loglike(c, cl, pd, model_name)
        except (ImportError, FileNotFoundError, ValueError):
            raise
        except Exception as e:
            logger.error("%s (%s): likelihood failed: %s", tag, c, e)
            return float(floor)
        if not np.isfinite(v):
            return float(floor)
        total += v
    return total


def check_available(tags) -> None:
    """Load what a run's ground-based tags need (start-up: a missing package fails here, not in a worker)."""
    comps = set()
    for t in tags:
        comps |= set(K.cmb_components(t))
    for c in sorted(comps):
        if c in ("CMB_ACT", "CMB_SPT"):
            candl_like(c)
        elif c == "CMB_planck_ACTcut":
            planck_act_cut()
        elif c == "CMB_lowl_sroll2":
            sroll2()


def check_lensing(tag: str, lens_only: bool) -> None:
    """Load a lensing tag's data at start-up (a missing package or file fails here, not in a worker)."""
    if tag == "CMB_lensing_ACTPlanck":
        act_lens_data(lens_only)
    else:
        raise KeyError(tag)
