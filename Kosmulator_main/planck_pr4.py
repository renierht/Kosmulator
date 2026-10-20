# SPDX-License-Identifier: LGPL-3.0-or-later
#
# Derived from Cobaya's planck_2018_CamSpec_python.py (A. Lewis, from the CamSpec Fortran code by
# G. Efstathiou, S. Gratton and A. Lewis), Cobaya copyright (C) 2017-2026 Jesus Torrado and Antony Lewis.
# Changes for Kosmulator (C) 2026 Renier T. Hough: a NumPy implementation of the TTTEEE likelihood with
# Kosmulator's parameter names, a shared float64 inverse-covariance cache and binned band powers.
#
# Unlike the rest of Kosmulator (MIT), this file is free software under the GNU Lesser General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version, as Cobaya is. It is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See LICENSES/LGPL-3.0.txt and
# LICENSES/GPL-3.0.txt in the repository. Cobaya's licence also asks that publications using results of
# the code are submitted to arXiv (no later than to a journal) and that bugs are reported when confirmed.
"""
Planck PR4 (NPIPE) high-l TT, TE, EE: the CamSpec 12.6cl likelihood (Rosenberg, Gratton &
Efstathiou 2022, MNRAS 517, 4620, arXiv:2205.10869), Kosmulator tag CMB_hil_PR4. It is the
high-l Planck likelihood of DESI DR2 (arXiv:2503.14738, Sect. IV B), next to the PR3 plik one
(CMB_hil).

A NumPy implementation of the Gaussian CamSpec likelihood as Cobaya distributes it
(planck_NPIPE_highl_CamSpec.TTTEEE, Cobaya 3.6.2, A. Lewis's Python port of the CamSpec
code), with Cobaya's settings: the spectra 143x143 (l = 30 - 2000), 217x217 and 143x217
(500 - 2500), TE and EE (30 - 2000), 9915 multipoles; power-law foreground residuals
A_f (l / 1500)^n_f in each TT spectrum (use_fg_residual_model = 0, amp_100 = 0); the
calibrations A_planck (all spectra, as y_cal^2), calTE and calEE. The nuisance priors are
Cobaya's (constants.GROUND_NUISANCE_SAMPLED). Checked against Cobaya's class at fixed spectra
(kit section DESIDR2).

Data: Cobaya's planck_native_data release v1, CamSpec_NPIPE.zip (85.4 MB, 0.5 GB unpacked), not in git
(the covariance is 509 MB). A run with CMB_hil_PR4 that finds no data downloads them at start-up, saying why
and from where (ensure_data), into Observations/planck_NPIPE_CamSpec/CamSpec_NPIPE/, and checks them (md5);
``python -m Kosmulator_main.planck_pr4 --get`` does the same by hand. A copy elsewhere is used through
KOSM_CAMSPEC_PR4_DIR or a Cobaya packages folder (COBAYA_PACKAGES_PATH). The
inverse covariance (9915 x 9915) is computed once (about 1 GB of memory for a minute) and kept
next to the data as a float64 .npy (786 MB), or in ~/.cache/kosmulator/camspec_pr4 (KOSM_CACHE_DIR)
when that folder cannot be written; every process maps it read-only, so the pool workers share one
copy in memory. Cobaya keeps its inverse in float32; the chi^2 of the two differ by about 0.003 at
chi^2 = 1.1e4.

Derived from Cobaya's planck_2018_CamSpec_python.py (Torrado & Lewis 2021, arXiv:2005.05290), so this
file is LGPL-3.0-or-later (header above); Cobaya itself is not needed.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np

from Kosmulator_main import constants as K

logger = logging.getLogger("Kosmulator.planck_pr4")

#: The spectra of the TT,TE,EE likelihood (Cobaya's TTTEEE.yaml)
USE_CL = ("143x143", "217x217", "143x217", "TE", "EE")
DATASET = "CamSpec_NPIPE_12_6_cl.dataset"
SUBDIR = os.path.join("planck_NPIPE_CamSpec", "CamSpec_NPIPE")
URL = "https://github.com/CobayaSampler/planck_native_data/releases/download/v1/CamSpec_NPIPE.zip"
ZIP_MB, UNPACKED_GB = 85.4, 0.5                 # MB = 1e6 bytes
#: md5 of the files CMB_hil_PR4 reads (planck_native_data v1, CamSpec_NPIPE.zip): checked after a download
#: and by --check
MD5 = {
    "CamSpec_NPIPE_12_6_cl.dataset": "f95aac05fddcd93562ee0bfe43cf58df",
    "like_NPIPE_12.6_unified_cov.bin": "843063c3ac30984fe4f2470d0ce38d8f",
    "like_NPIPE_12.6_unified_spectra.txt": "e0e22a85436a399380a65e855520e2ae",
    "like_NPIPE_12.6_unified_data_ranges.txt": "71607015ae0e6d0debc5d6d15e06373e",
}
#: Fixed inputs (Cobaya's params_TT_CamSpec.yaml): the 100x100 spectrum is not used
_FIXED = {"use_fg_residual_model": 0, "cal0": 1.0, "cal2": 1.0, "amp_100": 0.0, "n_100": 1.0}
_POWERLAW_PIVOT = 1500.0


def _candidate_dirs() -> List[str]:
    out = []
    env = os.environ.get("KOSM_CAMSPEC_PR4_DIR")
    if env:
        out.append(env)
    out.append(os.path.join(K.OBSERVATIONS_BASE, SUBDIR))
    pk = os.environ.get("COBAYA_PACKAGES_PATH")
    if pk:
        out.append(os.path.join(pk, "data", SUBDIR))
    return out


def data_dir() -> str:
    """The folder with CamSpec_NPIPE_12_6_cl.dataset (FileNotFoundError with what to do)."""
    for d in _candidate_dirs():
        if os.path.isfile(os.path.join(d, DATASET)):
            return d
    raise FileNotFoundError(
        "CMB_hil_PR4 needs the CamSpec PR4 data (CamSpec_NPIPE.zip of Cobaya's planck_native_data release): "
        "a run with CMB_hil_PR4 downloads them, or python -m Kosmulator_main.planck_pr4 --get (to Observations/"
        + SUBDIR.replace(os.sep, "/") + "), or set KOSM_CAMSPEC_PR4_DIR (or COBAYA_PACKAGES_PATH) to a copy you "
        "have. Looked in: " + ", ".join(_candidate_dirs()))


def _read_ini(path: str) -> Dict[str, str]:
    out = {}
    for line in open(path, encoding="utf-8-sig"):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _symmetrise(a: np.ndarray, block: int = 1024) -> None:
    """a = (a + a^T) / 2 in place, block by block (no second n x n array)."""
    n = a.shape[0]
    for i in range(0, n, block):
        for j in range(i, n, block):
            m = 0.5 * (a[i:i + block, j:j + block] + a[j:j + block, i:i + block].T)
            a[i:i + block, j:j + block] = m
            a[j:j + block, i:i + block] = m.T


class CamSpecPR4:
    """The CamSpec PR4 TT,TE,EE likelihood on the data of one folder (see the module docstring)."""

    def __init__(self, folder: str):
        self.folder = folder
        ini = _read_ini(os.path.join(folder, DATASET))
        spectra = np.loadtxt(os.path.join(folder, ini["cl_hat_file"]))
        cov_names = ini["covmat_cl"].split()
        names, lmin, lmax = [], [], []
        for line in open(os.path.join(folder, ini["data_ranges"]), encoding="utf-8-sig"):
            items = line.split()
            if items:
                names.append(items[0])
                lmin.append(int(items[1]))
                lmax.append(int(items[2]))
        if names != cov_names:
            raise ValueError(f"CamSpec PR4: data ranges {names} differ from covmat_cl {cov_names}")
        self.cl_names = names
        data_vector, used, self.ell_ranges, self.used_sizes = [], [], [], []
        nX = 0
        for i, (nm, lo, hi) in enumerate(zip(names, lmin, lmax)):
            n = hi - lo + 1 if hi >= lo else 0
            if nm in USE_CL and n > 0:
                data_vector.append(spectra[lo:hi + 1, i])
                used.append(np.arange(nX, nX + n))
                self.ell_ranges.append(np.arange(lo, hi + 1))
                self.used_sizes.append(n)
            else:
                self.ell_ranges.append(np.zeros(0, int))
                self.used_sizes.append(0)
            nX += n
        self.used_idx = np.concatenate(used)
        self.data_vector = np.concatenate(data_vector)
        self.n_data = int(self.data_vector.size)
        self.lmax = int(max(lmax[i] for i, nm in enumerate(names) if nm in USE_CL))
        self.ls = np.arange(self.lmax + 1, dtype=float)
        self._cov_file = os.path.join(folder, ini["covmat_fiducial"])
        self._cov_dtype = np.float64 if "64.bin" in self._cov_file else np.float32
        self._nX = nX
        self.covinv = self._load_covinv()

    # -- inverse covariance, computed once and memory-mapped by every process ---------------
    def _cache_key(self) -> str:
        """The covariance file's size and the md5 of its first and last MiB (not its date, so that
        unpacking the same zip again finds the same cache)."""
        size = os.path.getsize(self._cov_file)
        md5 = hashlib.md5(repr((USE_CL, size, self.n_data)).encode())
        with open(self._cov_file, "rb") as fh:
            md5.update(fh.read(1 << 20))
            fh.seek(max(0, size - (1 << 20)))
            md5.update(fh.read(1 << 20))
        return md5.hexdigest()[:12]

    def _cache_dirs(self) -> List[str]:
        """Next to the data first; the user's cache folder when that one cannot be written (a read-only
        Cobaya packages folder on a cluster); KOSM_CACHE_DIR before both when set."""
        dirs = [os.path.join(self.folder, ".kosmulator_cache"),
                os.path.join(os.path.expanduser("~"), ".cache", "kosmulator", "camspec_pr4")]
        if os.environ.get("KOSM_CACHE_DIR"):
            dirs.insert(0, os.path.join(os.environ["KOSM_CACHE_DIR"], "camspec_pr4"))
        return dirs

    def _load_covinv(self) -> np.ndarray:
        name = f"covinv_TTTEEE_{self._cache_key()}.npy"
        dirs = self._cache_dirs()
        for d in dirs:                                   # an existing cache anywhere is used
            if os.path.exists(os.path.join(d, name)):
                return self._map(os.path.join(d, name))
        for d in dirs:
            try:
                os.makedirs(d, exist_ok=True)
                if os.access(d, os.W_OK):
                    path = os.path.join(d, name)
                    self._build_covinv(path)
                    return self._map(path)
            except OSError as e:
                logger.info("CMB_hil_PR4: cannot keep the inverse covariance in %s (%s)", d, e)
        raise OSError(f"CMB_hil_PR4: no writable folder for the inverse covariance among {dirs}")

    def _map(self, path: str) -> np.ndarray:
        arr = np.load(path, mmap_mode="r")
        if arr.shape != (self.n_data, self.n_data):
            raise ValueError(f"{path}: shape {arr.shape}, expected {(self.n_data, self.n_data)}")
        return arr

    def _build_covinv(self, path: str) -> None:
        """Invert the used block of the covariance (float64, symmetric) into path, once: under a file lock,
        written to a temporary file in the same folder and renamed. Peak memory about 1 GB."""
        import tempfile
        from scipy import linalg as sla
        with open(path + ".lock", "w") as fh:
            try:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)          # another process may be writing it
            except Exception:
                pass
            if os.path.exists(path):
                return
            logger.info("CMB_hil_PR4: inverting the CamSpec PR4 covariance (%d x %d, once; stored as %s)",
                        self.n_data, self.n_data, path)
            raw = np.memmap(self._cov_file, dtype=self._cov_dtype, mode="r")
            if raw.size != self._nX ** 2:
                raise ValueError(f"CamSpec PR4 covariance has {raw.size} entries, expected {self._nX ** 2}")
            raw = raw.reshape(self._nX, self._nX)
            idx = self.used_idx
            a = np.empty((idx.size, idx.size), dtype=np.float64)
            for i in range(0, idx.size, 512):                     # the used block, row by row
                a[i:i + 512] = raw[idx[i:i + 512]][:, idx]
            del raw
            # a is symmetric, so a.T is the same matrix in Fortran order, which LAPACK inverts in place
            a = np.ascontiguousarray(sla.inv(a.T, overwrite_a=True, check_finite=False).T)
            _symmetrise(a)                                        # exactly symmetric for dsymv
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".covinv_", suffix=".npy")
            try:
                with os.fdopen(fd, "wb") as out:
                    np.save(out, a)
                um = os.umask(0)
                os.umask(um)
                os.chmod(tmp, 0o666 & ~um)                        # mkstemp makes it private (0600)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

    # -- likelihood --------------------------------------------------------------------------
    def foregrounds(self, pd: Dict[str, float]) -> np.ndarray:
        """Power-law residuals of the four TT spectra (100, 143, 217, 143x217), D_l in muK^2."""
        amp = [_FIXED["amp_100"], pd.get("amp_143", 0.0), pd.get("amp_217", 0.0), pd.get("amp_143x217", 0.0)]
        tilt = [_FIXED["n_100"], pd.get("n_143", 1.0), pd.get("n_217", 1.0), pd.get("n_143x217", 1.0)]
        x = self.ls / _POWERLAW_PIVOT
        return np.array([float(a) * x ** float(n) for a, n in zip(amp, tilt)])

    @staticmethod
    def cals(pd: Dict[str, float]) -> np.ndarray:
        """Calibration of each spectrum (100, 143, 217, 143x217, TE, EE), A_planck^2 included."""
        cal_planck = float(pd.get("A_planck", 1.0)) ** 2
        cal0, cal2 = _FIXED["cal0"], _FIXED["cal2"]
        return np.array([cal0, 1.0, cal2, np.sqrt(cal2), float(pd.get("calTE", 1.0)),
                         float(pd.get("calEE", 1.0))]) * cal_planck

    def residual(self, dl: Dict[str, np.ndarray], pd: Dict[str, float]) -> np.ndarray:
        """Data minus model over the 9915 multipoles (dl: D_l in muK^2 from l = 0, keys TT, TE, EE)."""
        for s in ("TT", "TE", "EE"):
            if dl[s].size <= self.lmax:
                raise ValueError(f"CMB_hil_PR4 needs CLASS spectra to l = {self.lmax}, got {dl[s].size - 1}")
        cals = self.cals(pd)
        fg = self.foregrounds(pd)
        delta = self.data_vector.copy()
        ix = 0
        for i, (cal, n) in enumerate(zip(cals, self.used_sizes)):
            if n == 0:
                continue
            ells = self.ell_ranges[i]
            if i <= 3:
                delta[ix:ix + n] -= (dl["TT"][ells] + fg[i][ells]) / cal
            elif i == 4:
                delta[ix:ix + n] -= dl["TE"][ells] / cal
            else:
                delta[ix:ix + n] -= dl["EE"][ells] / cal
            ix += n
        return delta

    def chi2(self, dl: Dict[str, np.ndarray], pd: Dict[str, float]) -> float:
        from scipy.linalg import blas
        d = self.residual(dl, pd)
        # symmetric product as Cobaya's chi_squared (the transposed C-order map is Fortran order)
        return float(blas.dsymv(alpha=1.0, a=self.covinv.T, x=d, lower=0).dot(d))

    def loglike(self, dl: Dict[str, np.ndarray], pd: Dict[str, float]) -> float:
        return -0.5 * self.chi2(dl, pd)

    # -- band powers for the plots ---------------------------------------------------------------
    def band_powers(self, dl: Dict[str, np.ndarray], pd: Dict[str, float], width: int = 50) -> List[dict]:
        """For the best-fit figures (ground_cmb.band_powers format): calibrated, foreground-subtracted
        data in bins of `width` multipoles with errors from the covariance's diagonal, and the theory
        binned with the same weights. TT is the inverse-variance mean of 143x143, 217x217 and
        143x217. The likelihood itself uses every multipole with the full covariance."""
        cov = np.memmap(self._cov_file, dtype=self._cov_dtype, mode="r", shape=(self._nX, self._nX))
        diag = np.asarray(cov[self.used_idx, self.used_idx], dtype=float)
        cals, fg = self.cals(pd), self.foregrounds(pd)
        parts = {"TT": [], "TE": [], "EE": []}
        ix = 0
        for i, n in enumerate(self.used_sizes):
            if n == 0:
                continue
            ells = self.ell_ranges[i]
            d = self.data_vector[ix:ix + n] * cals[i]
            var = diag[ix:ix + n] * cals[i] ** 2
            spec = "TT" if i <= 3 else ("TE" if i == 4 else "EE")
            if i <= 3:
                d = d - fg[i][ells]
            parts[spec].append((ells, d, var))
            ix += n
        out = []
        for spec, items in parts.items():
            if not items:
                continue
            hi = max(int(e[-1]) for e, _, _ in items)
            lo = min(int(e[0]) for e, _, _ in items)
            num, den = np.zeros(hi + 1), np.zeros(hi + 1)
            for e, d, v in items:
                num[e] += d / v
                den[e] += 1.0 / v
            th = np.asarray(dl[spec], float)[:hi + 1]
            ell_c, y, err, model = [], [], [], []
            for a in range(lo, hi + 1, width):
                b = min(a + width, hi + 1)
                w = den[a:b]
                if w.sum() <= 0:
                    continue
                ell_c.append(float((np.arange(a, b) * w).sum() / w.sum()))
                y.append(float(num[a:b].sum() / w.sum()))
                err.append(float(1.0 / np.sqrt(w.sum())))
                model.append(float((th[a:b] * w).sum() / w.sum()))
            out.append({"spec": spec, "label": "Planck PR4 high-l (CamSpec)", "ell": np.array(ell_c),
                        "y": np.array(y), "err": np.array(err), "model": np.array(model)})
        return out


_LIKE: Optional[CamSpecPR4] = None


def camspec() -> CamSpecPR4:
    """The CamSpec PR4 likelihood of this process (loaded once)."""
    global _LIKE
    if _LIKE is None:
        _LIKE = CamSpecPR4(data_dir())
    return _LIKE


def download_notice(dest: str) -> str:
    """Why the CamSpec PR4 data are downloaded, from where and to where (shown before every download)."""
    return (
        "CMB_hil_PR4 (the Planck PR4 CamSpec high-l TT, TE, EE likelihood of Rosenberg, Gratton & Efstathiou\n"
        "2022) needs its data: the binned spectra and their 9915 x 9915 covariance. They are not part of\n"
        "Kosmulator's repository, because the covariance alone is 509 MB (GitHub takes files up to 100 MB).\n"
        f"Kosmulator downloads them once, {ZIP_MB} MB ({UNPACKED_GB} GB unpacked), from the release in which Cobaya's\n"
        "authors publish the Planck likelihood data (no Cobaya installation is needed):\n"
        f"    {URL}\n"
        f"into {os.path.abspath(dest)}\n"
        "and checks them (md5). If you already have these data (e.g. from Cobaya's packages), set\n"
        "KOSM_CAMSPEC_PR4_DIR to their CamSpec_NPIPE folder instead.")


def verify(folder: str) -> List[str]:
    """Files of MD5 that are missing or differ in folder (empty when the data are the expected ones)."""
    bad = []
    for name, want in MD5.items():
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            bad.append(f"{name} (missing)")
            continue
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            bad.append(f"{name} (md5 {h.hexdigest()}, expected {want})")
    return bad


def get_data(dest: Optional[str] = None) -> str:
    """Download CamSpec_NPIPE.zip (with download_notice), unpack it into dest (Observations/planck_NPIPE_CamSpec/)
    and check the files (md5). The zip and the unpacked files first go to temporary names in dest; only data
    that pass the check are moved to dest/CamSpec_NPIPE, so a failed or broken download leaves nothing a
    later run would use."""
    import shutil
    import tempfile
    import time
    import urllib.request
    import zipfile
    dest = dest or os.path.join(K.OBSERVATIONS_BASE, "planck_NPIPE_CamSpec")
    os.makedirs(dest, exist_ok=True)
    print(download_notice(dest), flush=True)
    fd, tmp = tempfile.mkstemp(dir=dest, prefix=".CamSpec_NPIPE_", suffix=".zip")
    unpack = tempfile.mkdtemp(dir=dest, prefix=".CamSpec_NPIPE_unpack_")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(URL, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
            got, shown = 0, -1
            for chunk in iter(lambda: r.read(1 << 20), b""):
                out.write(chunk)
                got += len(chunk)
                pct = int(100 * got / total) if total else -1
                if pct >= 0 and pct // 10 != shown:
                    shown = pct // 10
                    print(f"  downloaded {got / 1e6:5.1f} of {total / 1e6:.1f} MB ({pct}%)", flush=True)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(unpack)
        bad = verify(os.path.join(unpack, "CamSpec_NPIPE"))
        if bad:
            raise ValueError(f"the downloaded CamSpec PR4 data are not the expected ones: {'; '.join(bad)}")
        folder = os.path.join(dest, "CamSpec_NPIPE")
        if os.path.exists(folder):                    # an incomplete copy (no .dataset) is kept aside
            os.rename(folder, f"{folder}.incomplete_{time.strftime('%Y%m%d_%H%M%S')}")
        os.rename(os.path.join(unpack, "CamSpec_NPIPE"), folder)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
        shutil.rmtree(unpack, ignore_errors=True)
    print(f"CamSpec PR4 data in {folder} (md5 checked)", flush=True)
    return folder


def ensure_data() -> str:
    """The data folder (data_dir); when there is none, the data are downloaded with get_data, which says why
    and from where. A failed download (no internet, e.g. on a cluster node) stops with what to do."""
    try:
        return data_dir()
    except FileNotFoundError:
        pass
    dest = os.path.join(K.OBSERVATIONS_BASE, "planck_NPIPE_CamSpec")
    try:
        return get_data(dest)
    except Exception as e:
        raise FileNotFoundError(
            f"CMB_hil_PR4: the CamSpec PR4 data could not be downloaded ({type(e).__name__}: {e}). Where there is "
            f"internet (e.g. a cluster's login node) run: python -m Kosmulator_main.planck_pr4 --get, or download "
            f"{URL} yourself, unpack it into {os.path.abspath(dest)} and run python -m Kosmulator_main.planck_pr4 "
            "--check; or set KOSM_CAMSPEC_PR4_DIR to a CamSpec_NPIPE folder you already have.") from e


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Planck PR4 CamSpec data for CMB_hil_PR4")
    ap.add_argument("--get", action="store_true", help="download, unpack and check the data (Observations/)")
    ap.add_argument("--check", action="store_true", help="check the data (md5) and load the likelihood (and "
                                                          "build its inverse covariance)")
    a = ap.parse_args(argv)
    if a.get:
        get_data()
    if a.check or not a.get:
        folder = data_dir()
        bad = verify(folder)
        if bad:
            print(f"CMB_hil_PR4 data in {folder} are not the expected ones: {'; '.join(bad)}")
            return 1
        lk = camspec()
        print(f"CMB_hil_PR4: {lk.n_data} multipoles, l_max {lk.lmax}, data in {lk.folder} (md5 checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
