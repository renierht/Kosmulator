"""
Ground-based CMB likelihoods (ACT DR6, SPT-3G D1) and the Planck + ACT (+ SPT) combinations
(item 57). Tags and settings: constants.GROUND_CMB_TAGS and the block above it.

Every likelihood of a group takes its spectra from the point's single lensed CLASS run
(Statistical_packages.get_cmb_spectra, l_max = constants.CMB_CLASS_LMAX, raised to 8501 for
ACT and 4095 for SPT by MCMC_setup), as the Planck likelihoods do.

- ACT DR6 and SPT-3G D1: candl (Balkenhol et al. 2024, arXiv:2401.13433) with the
  foreground-marginalised ("lite") data sets of candl_data (ACT_DR6_TTTEEE, in the repository)
  and spt_candl_data (SPT3G_D1_TnE_lite, downloaded once with Kosmulator_main/get_spt_data.py),
  unchanged copies in Observations/candl, so that only candl itself is installed (the data
  packages are used when the copies are missing). candl's internal
  priors are removed; Kosmulator applies
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
- CMB_lensing_SPT (item 61): SPT-3G MUSE phi phi band powers (MusePhiPhi, a numpy port of
  muse3glike) on the same CLASS run's C_L^phiphi.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import logging
import os
from typing import Dict, List, Optional

import numpy as np

from Kosmulator_main import constants as K

logger = logging.getLogger("Kosmulator.ground_cmb")

#: The two candl data sets in Observations/candl (unchanged copies, see its README), so only the
#: candl code is needed (pip install candl-like): ACT DR6 is in the repository; SPT-3G D1 is
#: downloaded once with python -m Kosmulator_main.get_spt_data (spt_candl_data has no licence
#: file, so Kosmulator does not host it). The data packages are a fallback.
_CANDL_DATA = {
    "CMB_ACT": ("ACT_DR6_CMB_only_v0/ACT_DR6_CMB_only.yaml", "candl_data", "ACT_DR6_TTTEEE"),
    "CMB_SPT": ("SPT3G_D1_TnE_lite/SPT3G_D1_TnE_lite.yaml", "spt_candl_data", "SPT3G_D1_TnE_lite"),
}
CANDL_DIR = "candl"                     # under constants.OBSERVATIONS_BASE
_LIKES: Dict[str, object] = {}          # per process


def candl_data_file(tag: str) -> str:
    """The candl data set (yaml) of a tag: Observations/candl first, else the data package's copy."""
    rel, mod_name, attr = _CANDL_DATA[tag]
    path = os.path.join(K.OBSERVATIONS_BASE, CANDL_DIR, rel)
    if os.path.exists(path):
        return path
    try:
        return str(getattr(importlib.import_module(mod_name), attr))
    except ImportError as e:
        if tag == "CMB_SPT":
            raise FileNotFoundError(
                f"CMB_SPT needs the SPT-3G D1 data in {os.path.dirname(path)}. Kosmulator does not host them "
                "(spt_candl_data has no licence file); download them once (21 MB, md5-checked) with:\n"
                "    python -m Kosmulator_main.get_spt_data\n"
                "and cite them (Observations/candl/README.md)") from e
        raise FileNotFoundError(f"{tag}: {path} is missing (and {mod_name} is not installed); the data are "
                                "part of Kosmulator's repository, see Observations/candl/README.md") from e


def candl_like(tag: str):
    """The candl likelihood of a tag, loaded once per process, without its internal priors."""
    if tag not in _LIKES:
        try:
            with contextlib.redirect_stdout(io.StringIO()):    # candl prints a notice when candl_data
                import candl                                   # is not installed: not needed here
        except ImportError as e:
            raise ImportError(f"{tag} needs candl ({e}); install with: pip install candl-like "
                              "(the data are in Observations/candl)") from e
        path = candl_data_file(tag)
        with contextlib.redirect_stdout(io.StringIO()):
            like = candl.Like(path)
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
        self.sigma = np.sqrt(np.diag(cov)[self.idx])          # for plots (item 57)
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
_ACT_LENS_COMPACT_FORMAT = 2       # raise when _act_lens_compact changes (new cache file)
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
    ddir = act_lens_dir()
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return alike.load_data(
                K.ACT_LENS_VARIANT, ddir=ddir, lens_only=bool(lens_only), like_corrections=not lens_only,
                nsims_act=K.ACT_LENS_NSIMS[0], nsims_planck=K.ACT_LENS_NSIMS[1],
                trim_lmax=ACT_LENS_TRIM_LMAX, version=K.ACT_LENS_VERSION)
    except OSError as e:
        if lens_only:
            raise
        raise FileNotFoundError(
            f"CMB_lensing_ACTPlanck with primary CMB in the group needs {act_lens_shipped_file()} (in git) "
            f"or ACT's correction matrices in {os.path.join(ddir, 'like_corrs')} (N1*.txt and *.npy, 2.8 GB; "
            "not in git, they are above GitHub's 100 MB file limit; ACT_dr6_likelihood_v1.2.tgz from "
            f"https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/data/, see "
            f"Observations/{K.ACT_LENS_DIR}/README.md). ({e})") from e


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
           "n_act": np.array(int(blocks[0][1].shape[0])),        # the first n_act bins are ACT's
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


def act_lens_shipped_file() -> str:
    """The compact corrected likelihood shipped in git (Observations/<ACT_LENS_DIR>/), so that a
    clone runs CMB_lensing_ACTPlanck with primary CMB without the 2.8 GB of correction matrices;
    written by write_act_lens_shipped from those matrices."""
    return os.path.join(K.OBSERVATIONS_BASE, K.ACT_LENS_DIR,
                        f"{K.ACT_LENS_VARIANT}_corrections_{K.ACT_LENS_VERSION}.npz")


def _act_lens_meta() -> Dict[str, np.ndarray]:
    """The settings the compact form depends on, stored with the shipped file and checked on reading."""
    return {"meta_format": np.array(_ACT_LENS_COMPACT_FORMAT), "meta_variant": np.array(K.ACT_LENS_VARIANT),
            "meta_version": np.array(K.ACT_LENS_VERSION), "meta_nsims": np.array(tuple(K.ACT_LENS_NSIMS)),
            "meta_trim_lmax": np.array(ACT_LENS_TRIM_LMAX)}


def _read_act_lens_shipped() -> Optional[Dict[str, np.ndarray]]:
    """The shipped compact form, or None when it is missing or was made with other settings."""
    path = act_lens_shipped_file()
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path) as z:
            c = {k: z[k] for k in z.files}
    except Exception as e:
        logger.warning("CMB_lensing_ACTPlanck: %s unreadable (%s)", path, e)
        return None
    want = _act_lens_meta()
    if any(k not in c or not np.array_equal(c[k], v) for k, v in want.items()):
        logger.warning("CMB_lensing_ACTPlanck: %s was made with other settings (%s); not used", path,
                       {k[5:]: c[k].tolist() for k in want if k in c})
        return None
    try:
        import act_dr6_lenslike as alike
        built, have = str(c.get("meta_package", "?")), str(getattr(alike, "__version__", "?"))
        if built != have:
            logger.warning("CMB_lensing_ACTPlanck: %s was made with act_dr6_lenslike %s, %s is installed (same "
                           "data release; rebuild with ground_cmb.write_act_lens_shipped() if ACT changed "
                           "load_data)", os.path.basename(path), built, have)
    except ImportError:
        pass
    return {k: v for k, v in c.items() if not k.startswith("meta_")}


def write_act_lens_shipped(path: Optional[str] = None) -> str:
    """
    Build the compact corrected likelihood from ACT's correction matrices (2.8 GB in
    Observations/<ACT_LENS_DIR>/<version>/like_corrs) and write it, with the settings and the
    act_dr6_lenslike version, to act_lens_shipped_file() (or path). For maintainers, after a new
    ACT release or a change of _act_lens_compact (raise _ACT_LENS_COMPACT_FORMAT):
        python -c "from Kosmulator_main import ground_cmb as G; print(G.write_act_lens_shipped())"
    """
    import act_dr6_lenslike as alike
    c = _act_lens_compact(act_lens_package_data(False))
    c.update(_act_lens_meta())
    c["meta_package"] = np.array(str(getattr(alike, "__version__", "?")))
    path = path or act_lens_shipped_file()
    _write_npz(path, c)
    return path


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
    corrections (primary CMB in the group): the cache file of this data folder, else the file
    shipped in git (act_lens_shipped_file), else built from the 2.8 GB of correction matrices and
    cached (the first process does it at start-up, MCMC_setup, so spawned workers only read a
    3 MB file)."""
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
                c = _read_act_lens_shipped()            # in git: a clone needs no 2.8 GB
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
    c = act_lens_data(lens_only)
    delta = c["data"] - act_lens_theory(cl, lens_only)
    return -0.5 * float(delta @ c["cinv"] @ delta)


def act_lens_theory(cl: Dict[str, np.ndarray], lens_only: bool) -> np.ndarray:
    """The binned C_L^kappa kappa the likelihood compares with the data (corrections included)."""
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
    return th


class MusePhiPhi:
    """
    SPT-3G 2019 - 2020 lensing power spectrum from MUSE (Ge et al. 2025, arXiv:2411.06000), the
    phi phi band powers only: a numpy port of muse3glike 1.0.0 (March 2025)
    spt3g_2yr_delensed_ee_optimal_pp_muse(components=["ϕϕ"]), which marginalises the delensed EE
    band powers and the systematics parameters (the phi phi block of the covariance). As in the
    package: Gaussian in arctanh(theory / scale - 1.01), the covariance inverse plus 1/0.8^2 on
    the diagonal (the package's volume term for its uniform prior on the band powers, "between 0
    and 2 times the Planck theory" in its README; exactly, finite for 0.01 < theory / scale < 2.01),
    theory outside that range rejected. The MUSE band powers need no correction from the CMB spectra.
    """

    def __init__(self, path: str):
        import re
        import h5py
        with h5py.File(path, "r") as f:
            axis = f["axis"].asstr()[()]
            m = re.search(r"ϕϕ\s*=\s*(\d+):(\d+)", axis)
            if m is None:
                raise ValueError(f"{path}: no phi phi block in the axis {axis!r}")
            sl = slice(int(m.group(1)) - 1, int(m.group(2)))          # Julia 1-based, inclusive
            s_full = np.asarray(f["scale"][:], float)
            cov = np.asarray(f["covariance"][:], float)
            data = np.asarray(f["data"][:], float)
            self.W = np.asarray(f["BPWF"]["ϕϕ"][:], float).T           # (bins, L), column j is L = j + 1
            ell = np.asarray(f["BPWF"]["ℓ"][:])
        if not np.array_equal(ell, np.arange(1, self.W.shape[1] + 1)):
            raise ValueError(f"{path}: band-power window multipoles are not 1 - {self.W.shape[1]}")
        self.s = s_full[sl]
        n = self.s.size
        cov_s = ((cov / s_full).T / s_full)[sl, sl]                     # covariance of data / scale
        cov_t = np.linalg.inv(np.linalg.inv(cov_s) + np.eye(n) / 0.8 ** 2)
        self.chol = np.linalg.cholesky(cov_t)
        self.d_t = np.arctanh(data[sl] / self.s - 1.01)
        self.d = data[sl]                                        # C_L^phiphi band powers (plots)
        self.sigma = np.sqrt(np.diag(cov)[sl])                   # untransformed: for plots only
        self.L_eff = self.W @ np.arange(1, self.W.shape[1] + 1) / self.W.sum(axis=1)
        self.n_data = int(n)
        self.lmax = int(np.nonzero(np.abs(self.W).sum(axis=0))[0][-1] + 1)   # 3073 in the March 2025 file

    def theory(self, pp: np.ndarray) -> np.ndarray:
        """The band powers of C_L^phiphi (CLASS's, from L = 0) through the windows."""
        pp = np.asarray(pp, float)
        if pp.size - 1 < self.lmax:
            raise ValueError(f"CMB_lensing_SPT needs C_L^phiphi to L = {self.lmax}, got {pp.size - 1}")
        x = np.zeros(self.W.shape[1])
        m = min(x.size, pp.size - 1)
        x[:m] = pp[1:m + 1]
        return self.W @ x

    def loglike(self, pp: np.ndarray) -> float:
        """pp: CLASS's dimensionless C_L^phiphi from L = 0."""
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.arctanh(self.theory(pp) / self.s - 1.01)
        if not np.all(np.isfinite(t)):
            return -np.inf                       # outside the band-power prior (the package returns NaN)
        z = np.linalg.solve(self.chol, self.d_t - t)
        return -0.5 * float(z @ z)


def spt_muse_file() -> str:
    """The SPT-3G MUSE likelihood file: Observations/<SPT_MUSE_DIR>/<SPT_MUSE_FILE>, else the dat
    folder of an installed muse3glike (found without importing it: it would use jax if present)."""
    cands = [os.path.join(K.OBSERVATIONS_BASE, K.SPT_MUSE_DIR, K.SPT_MUSE_FILE)]
    try:
        import importlib.util
        spec = importlib.util.find_spec("muse3glike")
        if spec is not None and spec.submodule_search_locations:
            cands.append(os.path.join(list(spec.submodule_search_locations)[0], "dat", K.SPT_MUSE_FILE))
    except Exception:
        pass
    for p in cands:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"SPT-3G MUSE lensing data not found ({cands[0]}). It is the file src/muse3glike/dat/{K.SPT_MUSE_FILE} "
        "of muse_3g_like_march_2025.zip (https://pole.uchicago.edu/public/data/ge25/)")


def spt_muse() -> MusePhiPhi:
    if "spt_muse" not in _NATIVE:
        _NATIVE["spt_muse"] = MusePhiPhi(spt_muse_file())
    return _NATIVE["spt_muse"]


def lensing_loglike(tag: str, pd: Dict[str, float], model_name: str, lens_only: bool,
                    floor: float = -1e10) -> float:
    """ln L of a lensing tag of item 61 on the point's single CLASS run (lens_only: ACT + Planck
    only; the SPT MUSE band powers have no correction from the CMB spectra)."""
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
        if tag == "CMB_lensing_ACTPlanck":
            v = act_lens_loglike(cl, lens_only)
        else:
            v = spt_muse().loglike(cl["pp"])
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
    if tag == "CMB_lensing_SPT":
        return spt_muse().n_data
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
    return float(like.log_like(_candl_pars(tag, like, dl, pd)))


def _candl_pars(tag: str, like, dl: Dict[str, np.ndarray], pd: Dict[str, float]) -> dict:
    """candl's input for one point: the theory D_l over its multipole range and the nuisances."""
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
    return pars


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
    elif tag == "CMB_lensing_SPT":
        spt_muse()
    else:
        raise KeyError(tag)


# ---------------------------------------------------------------------------
# Band powers for the best-fit plots (item 57)
# ---------------------------------------------------------------------------
#: Lensing panels show 10^7 [L (L + 1)]^2 C_L^phiphi / (2 pi), Planck's convention
PP_PLOT_SCALE = 1e7


def _read_fits_vector(path: str) -> np.ndarray:
    """The data of a one-dimensional FITS image (clik's lensing folders), without astropy."""
    with open(path, "rb") as fh:
        raw = fh.read()
    hdr, pos, done = {}, 0, False
    while not done:
        block = raw[pos:pos + 2880].decode("ascii", "replace")
        pos += 2880
        for i in range(0, 2880, 80):
            card = block[i:i + 80]
            if card[:8].strip() == "END":
                done = True
                break
            if card[8:10] == "= ":
                hdr[card[:8].strip()] = card[10:].split("/")[0].strip()
    n = int(hdr.get("NAXIS1", 0)) if int(hdr.get("NAXIS", 0)) else 0
    dt = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}[int(hdr["BITPIX"])]
    return np.frombuffer(raw, dtype=dt, count=n, offset=pos).astype(float)


def planck_lensing_theory(folder: str, cl: Dict[str, np.ndarray], A_planck: float = 1.0):
    """
    Planck 2018 lensing (a clik_lensing folder) for the plots: (effective L, data, errors,
    binned theory), in [L (L + 1)]^2 C_L^phiphi / (2 pi). The binned theory is clik's (clipy
    0.15's _clik_lensing written out): bins @ (C_L^phiphi [L (L + 1)]^2 / 2 pi) - cor0, plus
    cors times the current spectra when the folder has them (N1 and normalisation).
    """
    d = os.path.join(folder, "clik_lensing")
    mdb = {}
    for line in open(os.path.join(d, "_mdb")):
        parts = line.split()
        if len(parts) >= 3:
            mdb[parts[0]] = parts[2]
    nb, lmax = int(mdb["nbins"]), int(mdb["lmax"])
    rd = lambda name: _read_fits_vector(os.path.join(d, name))
    pp_hat = rd("pp_hat")
    bins = rd("bins").reshape(nb, -1)
    siginv = rd("siginv").reshape(nb, nb)
    cor0 = rd("cor0") if os.path.exists(os.path.join(d, "cor0")) else 0.0
    hascl = rd("hascl").astype(int) if os.path.exists(os.path.join(d, "hascl")) else np.zeros(6, int)
    L = np.arange(lmax + 1, dtype=float)
    m2, m1 = (L * (L + 1)) ** 2 / (2 * np.pi), L * (L + 1) / (2 * np.pi)
    pp = np.asarray(cl["pp"], float)[:lmax + 1]
    th = bins @ (pp * m2) - cor0
    if os.path.exists(os.path.join(d, "cors")):
        t2 = (K.T_CMB_DEFAULT * 1e6) ** 2
        cmb = [np.asarray(cl[x], float)[:lmax + 1] * t2 if x in cl else np.zeros(lmax + 1)
               for x, h in zip(("tt", "ee", "bb", "te", "tb", "eb"), hascl) if h]
        fid = rd("cl_fid")
        fphi = pp if int(mdb.get("ren1", 1)) else fid[:lmax + 1]
        fcmb = (np.concatenate(cmb) if cmb else np.zeros(0)) if int(mdb.get("renorm", 1)) else fid[lmax + 1:]
        cors = rd("cors").reshape(nb, -1)
        calib = 1.0 / float(A_planck) ** 2
        th = th + cors @ np.concatenate([fphi * m2, fcmb * calib * np.tile(m1, len(cmb))])
    w = bins.sum(axis=1)
    L_eff = (bins @ L) / np.where(w != 0, w, 1.0)
    return L_eff, pp_hat, np.sqrt(np.diag(np.linalg.inv(siginv))), th, siginv


def _plik_lite_full() -> PlikLiteCut:
    if "plite" not in _NATIVE:
        _NATIVE["plite"] = PlikLiteCut(os.path.join(K.OBSERVATIONS_BASE, "plik_lite_v22_TTTEEE.clik"))
    return _NATIVE["plite"]


def _plik_band_powers(pl: PlikLiteCut, dl: Dict[str, np.ndarray], A: float, label: str, specs) -> List[dict]:
    out = []
    off = 0
    for spec, keep in zip(("TT", "TE", "EE"), pl.used):
        n = keep.size
        if n and spec in specs:
            lb = 0.5 * (pl.blmin[keep] + pl.blmax[keep])
            f = lb * (lb + 1) / (2 * np.pi)                     # binned C_l -> D_l at the bin centre
            th = np.array([np.dot(dl[spec][pl.blmin[i]:pl.blmax[i] + 1], pl.weights[pl.blmin[i]:pl.blmax[i] + 1])
                           for i in keep]) / A ** 2
            sl = slice(off, off + n)
            out.append({"spec": spec, "label": label, "ell": lb, "y": pl.x[sl] * f, "err": pl.sigma[sl] * f,
                        "model": th * f})
        off += n
    return out


def _lowl_band_powers(spec: str, cl: Dict[str, np.ndarray], label: str) -> List[dict]:
    """Planck's low-l band powers with their asymmetric errors (Observations/CMB_TT.dat); the
    residual uses the error on the model's side. The Commander likelihood is not Gaussian in
    D_l, so these residuals show the fit, not the likelihood's terms."""
    path = K.DEFAULT_CMB_FILES[spec]
    if not os.path.exists(path):
        return []
    t = np.loadtxt(path)
    t = t[(t[:, 0] >= 2) & (t[:, 0] <= 29)]
    ell = t[:, 0]
    lo, hi = (t[:, 2], t[:, 3]) if t.shape[1] > 3 else (t[:, 2], t[:, 2])
    c = np.asarray(cl[spec.lower()], float)
    model = ell * (ell + 1) / (2 * np.pi) * c[ell.astype(int)] * (K.T_CMB_DEFAULT * 1e6) ** 2
    y = t[:, 1]
    return [{"spec": spec, "label": label, "ell": ell, "y": y, "err": 0.5 * (lo + hi),
             "err_lohi": np.vstack([lo, hi]), "model": model,
             "resid": (y - model) / np.where(model > y, hi, lo)}]


def _simall_table() -> np.ndarray:
    """SimAll's low-l EE table (the CMB_lowl clik folder's probEE: ln P on 3000 steps of 1e-4
    muK^2 in D_l^EE, l = 2 - 29), as (steps, multipoles) like TableEE.prob."""
    if "simall" not in _NATIVE:
        d = os.path.join(K.OBSERVATIONS_BASE, K.PLANCK_LOWL_FILES["CMB_lowl"], "clik", "lkl_0")
        mdb = {}
        for line in open(os.path.join(d, "_mdb")):
            parts = line.split()
            if len(parts) >= 3:
                mdb[parts[0]] = parts[2]
        nell, nsteps = int(mdb["nell"]), int(mdb["nstepsEE"])
        if float(mdb["stepEE"]) != TableEE.STEP or int(mdb["lmin"]) != TableEE.LMIN:
            raise ValueError(f"{d}: SimAll table with step {mdb['stepEE']}, l_min {mdb['lmin']}")
        _NATIVE["simall"] = _read_fits_vector(os.path.join(d, "probEE")).reshape(nell, nsteps).T
    return _NATIVE["simall"]


def _table_band_powers(prob: np.ndarray, dl_ee: np.ndarray, A: float, label: str) -> List[dict]:
    """
    A low-l EE likelihood tabulated per multipole (SimAll, SRoll2: ln L = sum over l of ln P_l(D_l))
    as band powers: the most probable D_l and its 68% interval from each l's table, and as
    residual sign(y - model) sqrt(2 [ln P_l(y) - ln P_l(model)]), whose squares add up to the
    likelihood's chi^2 above its maximum. The model is D_l / A_planck^2, as the likelihood reads it.
    """
    nl = prob.shape[1]
    ell = np.arange(TableEE.LMIN, TableEE.LMIN + nl, dtype=float)
    grid = (np.arange(prob.shape[0]) + 0.5) * TableEE.STEP          # step centres (the likelihood floors)
    j = np.arange(nl)
    ipk = np.argmax(prob, axis=0)
    y = grid[ipk]
    p = np.exp(prob - prob[ipk, j])
    cdf = np.cumsum(p, axis=0)
    cdf /= cdf[-1]
    lo = np.array([np.interp(0.16, cdf[:, k], grid) for k in j])
    hi = np.array([np.interp(0.84, cdf[:, k], grid) for k in j])
    model = np.asarray(dl_ee, float)[TableEE.LMIN:TableEE.LMIN + nl] / float(A) ** 2
    im = np.floor(model / TableEE.STEP).astype(int)
    inside = (im >= 0) & (im < prob.shape[0])
    lp_model = np.where(inside, prob[np.clip(im, 0, prob.shape[0] - 1), j], -np.inf)
    with np.errstate(invalid="ignore"):
        resid = np.sign(y - model) * np.sqrt(np.maximum(2.0 * (prob[ipk, j] - lp_model), 0.0))
    return [{"spec": "EE", "label": label, "ell": ell, "y": y, "err": 0.5 * (hi - lo),
             "err_lohi": np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)]), "model": model,
             "resid": resid}]


def band_powers(tag: str, cl: Dict[str, np.ndarray], pd: Dict[str, float], group) -> List[dict]:
    """
    Data, errors and the binned model of one CMB likelihood at a point, for the best-fit
    plots (item 57): a list of {"spec": TT | TE | EE | PP, "label", "ell" (effective
    multipoles), "y", "err", "model", optionally "err_lohi" (asymmetric error bars) and
    "resid" (the residual in sigma when it is not (y - model) / err)}, in D_l [muK^2] or, for
    PP, 10^7 [L (L + 1)]^2 C_L^phiphi / (2 pi). Binned models are the likelihood's own
    (windows, calibrations, lensing corrections). For the Gaussian likelihoods (candl,
    plik_lite, ACT + Planck lensing, Planck 2018 lensing) the band powers with their
    covariance give the likelihood's ln L; Planck high-l is shown in the plik_lite bins
    whichever plik the group uses. SimAll and SRoll2: their tables per multipole (residuals
    add up to their chi^2); SPT MUSE: Gaussian in a transformed variable, so (y - model) /
    err is close to, not equal to, its terms; low-l TT: Planck's band powers. A
    combination: use cmb_components(tag).
    """
    t2 = (K.T_CMB_DEFAULT * 1e6) ** 2
    A = float(pd.get("A_planck", 1.0))
    if tag in ("CMB_hil", "CMB_hil_TT"):
        return _plik_band_powers(_plik_lite_full(), _dl(cl), A, "Planck 2018 high-l (plik_lite bins)",
                                 ("TT",) if tag == "CMB_hil_TT" else ("TT", "TE", "EE"))
    if tag == "CMB_planck_ACTcut":
        return _plik_band_powers(planck_act_cut(), _dl(cl), A, "Planck 2018 (cut for ACT)", ("TT", "TE", "EE"))
    if tag == "CMB_lowl_TT":
        return _lowl_band_powers("TT", cl, "Planck low-l TT")
    if tag == "CMB_lowl":
        return _table_band_powers(_simall_table(), _dl(cl)["EE"], A, "Planck low-l EE (SimAll)")
    if tag == "CMB_lowl_sroll2":
        return _table_band_powers(sroll2().prob, _dl(cl)["EE"], A, "Planck low-l EE (SRoll2)")
    if tag in ("CMB_ACT", "CMB_SPT"):
        like = candl_like(tag)
        model = np.asarray(like.bin_model_specs(like.get_model_specs(_candl_pars(tag, like, _dl(cl), pd))), float)
        data = np.asarray(like.data_bandpowers, float)
        err = np.sqrt(np.diag(np.asarray(like.covariance, float)))
        ell = np.asarray(like.effective_ells, float)
        name = "ACT DR6" if tag == "CMB_ACT" else "SPT-3G D1"
        out = []
        for i, sp in enumerate(like.spec_order):
            sl = slice(int(like.bins_start_ix[i]), int(like.bins_stop_ix[i]))
            out.append({"spec": sp.split()[0], "label": name, "ell": ell[sl], "y": data[sl], "err": err[sl],
                        "model": model[sl]})
        return out
    lens_only = not K.group_has_primary_cmb(group)
    if tag == "CMB_lensing_ACTPlanck":
        c = act_lens_data(lens_only)
        th = act_lens_theory(cl, lens_only)
        B = np.asarray(c["B"], float)
        L_eff = B @ np.arange(B.shape[1]) / B.sum(axis=1)
        err = np.sqrt(np.diag(np.linalg.inv(np.asarray(c["cinv"], float))))
        f = PP_PLOT_SCALE * 2 / np.pi                            # C^kk -> [L(L+1)]^2 C^phiphi / 2 pi
        n_act = int(c["n_act"])
        return [{"spec": "PP", "label": lab, "ell": L_eff[sl], "y": c["data"][sl] * f, "err": err[sl] * f,
                 "model": th[sl] * f}
                for lab, sl in (("ACT DR6 lensing", slice(0, n_act)), ("Planck PR4 lensing", slice(n_act, None)))
                if L_eff[sl].size]
    if tag == "CMB_lensing_SPT":
        # The MUSE band powers are of C_L^phiphi in wide bins over which [L (L + 1)]^2 C_L^phiphi
        # changes much less than C_L^phiphi: converted with the model's ratio of the two binned
        # (W [L (L + 1)]^2 C_L / 2 pi over W C_L), not at the effective L, so the binned model sits on
        # the curve. (data - model) / error is unchanged.
        m = spt_muse()
        pp = np.asarray(cl["pp"], float)
        Lw = np.arange(pp.size, dtype=float)
        th = m.theory(pp)
        f = PP_PLOT_SCALE * m.theory((Lw * (Lw + 1)) ** 2 / (2 * np.pi) * pp) / th
        return [{"spec": "PP", "label": "SPT-3G MUSE lensing", "ell": m.L_eff, "y": m.d * f, "err": m.sigma * f,
                 "model": th * f}]
    if tag == "CMB_lensing":
        name = ("smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_CMBmarged.clik_lensing" if lens_only
                else "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing")
        L, d, e, th, _ = planck_lensing_theory(os.path.join(K.OBSERVATIONS_BASE, name), cl, A)
        return [{"spec": "PP", "label": "Planck 2018 lensing", "ell": L, "y": d * PP_PLOT_SCALE,
                 "err": e * PP_PLOT_SCALE, "model": th * PP_PLOT_SCALE}]
    return []
