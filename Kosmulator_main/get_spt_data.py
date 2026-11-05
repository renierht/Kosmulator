"""
Restore (download and check) the SPT-3G data Kosmulator includes in Observations/.

The SPT-3G D1 lite data and the SPT-3G MUSE lensing file are part of Kosmulator's repository
(public data with their sources and citations; the releases have no licence file). If they are
missing or changed in your copy, this script fetches them from the SPT releases and checks every
file against the md5 of the version Kosmulator was tested with.

    python -m Kosmulator_main.get_spt_data                     # SPT-3G D1 T&E lite (CMB_SPT, CMB_SPA)
    python -m Kosmulator_main.get_spt_data --no-lite --muse-zip PATH   # SPT-3G MUSE lensing (CMB_lensing_SPT)

- SPT-3G D1 T&E lite: the six files of spt_candl_data's SPT3G_D1_TnE_v0/lite and its reference
  test, from github.com/SouthPoleTelescope/spt_candl_data at commit SPT_CANDL_COMMIT, into
  Observations/candl/SPT3G_D1_TnE_lite/ and Observations/candl/tests/.
- SPT-3G MUSE: src/muse3glike/dat/90_150_220.h5 of muse_3g_like_march_2025.zip, which you
  download from https://pole.uchicago.edu/public/data/ge25/ and pass with --muse-zip, into
  Observations/spt3g_muse_lensing/.

What to cite is printed at the end (and listed in Observations/DATA_SOURCES.md).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
import zipfile
from typing import Dict, Tuple

from Kosmulator_main import constants as K

SPT_CANDL_COMMIT = "f22611a9671d50af6135f048a2ea3f43ef4c2f62"      # spt_candl_data 3.0.1
SPT_CANDL_BASE = (f"https://raw.githubusercontent.com/SouthPoleTelescope/spt_candl_data/{SPT_CANDL_COMMIT}/"
                  "spt_candl_data/")

#: source path in spt_candl_data -> (destination under Observations/candl, md5)
SPT_LITE_FILES: Dict[str, Tuple[str, str]] = {
    "SPT3G_D1_TnE_v0/lite/SPT3G_D1_TnE_lite.yaml":
        ("SPT3G_D1_TnE_lite/SPT3G_D1_TnE_lite.yaml", "6d18cec0786a13555456976d34acfa0d"),
    "SPT3G_D1_TnE_v0/lite/SPT3G_D1_TnE_lite_bdp.txt":
        ("SPT3G_D1_TnE_lite/SPT3G_D1_TnE_lite_bdp.txt", "7ab63d52d032f1c03a9baaab96ca8620"),
    "SPT3G_D1_TnE_v0/lite/SPT3G_D1_TnE_lite_cov.txt":
        ("SPT3G_D1_TnE_lite/SPT3G_D1_TnE_lite_cov.txt", "d3b399a7e5f0d457c3d535e45643b1fc"),
    "SPT3G_D1_TnE_v0/lite/windows/TT_lxl_window_functions.txt":
        ("SPT3G_D1_TnE_lite/windows/TT_lxl_window_functions.txt", "f14cbf8d99939972248db20c9aa1eaee"),
    "SPT3G_D1_TnE_v0/lite/windows/TE_lxl_window_functions.txt":
        ("SPT3G_D1_TnE_lite/windows/TE_lxl_window_functions.txt", "b29b89ed9d1a48b7e42eb0a77f035392"),
    "SPT3G_D1_TnE_v0/lite/windows/EE_lxl_window_functions.txt":
        ("SPT3G_D1_TnE_lite/windows/EE_lxl_window_functions.txt", "324a1c65cef8645b775f9f3e54843f43"),
    "tests/SPT3G_D1_TnE_lite_test.yaml":
        ("tests/SPT3G_D1_TnE_lite_test.yaml", "a894591a63cee0e6cb1f68de991d5baf"),
    "tests/SPT3G_D1_TnE_test_spec.txt":
        ("tests/SPT3G_D1_TnE_test_spec.txt", "a74582ec6fb8e573bd77d2b78b7fb3a9"),
}
MUSE_ZIP_MEMBER = "src/muse3glike/dat/" + K.SPT_MUSE_FILE
MUSE_MD5 = "8839ad236d98022ba54d4de8a0ead995"

def _cite(*keys):
    from Kosmulator_main.citations import REFS
    return "; ".join(REFS[k] for k in keys)


CITATIONS = (
    "candl: " + _cite("balkenhol24"),
    "SPT-3G D1 T&E (CMB_SPT, CMB_SPA; SPT-lite is Appendix F of Camphuis et al.): "
    + _cite("camphuis26", "balkenhol25", "quan26"),
    "SPT-3G MUSE lensing (CMB_lensing_SPT): " + _cite("ge25"),
)


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _place(data: bytes, dest: str, md5: str, what: str) -> None:
    """Write data to dest through a temporary file after checking its md5 (complete or absent)."""
    got = hashlib.md5(data).hexdigest()
    if got != md5:
        raise ValueError(f"{what}: md5 {got}, expected {md5} (not the version Kosmulator was tested with)")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".part_", dir=os.path.dirname(dest))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        um = os.umask(0)
        os.umask(um)
        os.chmod(tmp, 0o666 & ~um)                  # mkstemp makes it 0600
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def spt_lite_missing(obs_base: str = None) -> list:
    """The SPT-3G D1 lite files (destinations) that are absent or differ from the tested version."""
    root = os.path.join(obs_base or K.OBSERVATIONS_BASE, "candl")
    return [dst for dst, md5 in SPT_LITE_FILES.values()
            if not os.path.isfile(os.path.join(root, dst)) or _md5(os.path.join(root, dst)) != md5]


def get_spt_lite(obs_base: str = None, base_url: str = SPT_CANDL_BASE, force: bool = False) -> int:
    """Download the SPT-3G D1 T&E lite data set; returns the number of files written."""
    root = os.path.join(obs_base or K.OBSERVATIONS_BASE, "candl")
    n = 0
    for src, (dst, md5) in SPT_LITE_FILES.items():
        path = os.path.join(root, dst)
        if not force and os.path.isfile(path) and _md5(path) == md5:
            continue
        url = base_url + src
        print(f"  {url}")
        with urllib.request.urlopen(url, timeout=120) as r:
            _place(r.read(), path, md5, src)
        n += 1
    return n


def get_spt_muse(zip_path: str, obs_base: str = None) -> str:
    """Take the MUSE likelihood file out of muse_3g_like_march_2025.zip; returns its path."""
    dest = os.path.join(obs_base or K.OBSERVATIONS_BASE, K.SPT_MUSE_DIR, K.SPT_MUSE_FILE)
    with zipfile.ZipFile(zip_path) as z:
        names = [m for m in z.namelist() if m.endswith(MUSE_ZIP_MEMBER)]
        if not names:
            raise ValueError(f"{zip_path} has no {MUSE_ZIP_MEMBER} (is it muse_3g_like_march_2025.zip?)")
        _place(z.read(names[0]), dest, MUSE_MD5, names[0])
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download the SPT-3G data Kosmulator supports into Observations/.")
    ap.add_argument("--muse-zip", default=None,
                    help="muse_3g_like_march_2025.zip from https://pole.uchicago.edu/public/data/ge25/ (CMB_lensing_SPT)")
    ap.add_argument("--no-lite", action="store_true", help="skip the SPT-3G D1 T&E lite data (CMB_SPT, CMB_SPA)")
    ap.add_argument("--force", action="store_true", help="download again even when the files are there")
    a = ap.parse_args(argv)
    if a.no_lite and not a.muse_zip:
        print("Nothing to do: --no-lite without --muse-zip")
        return 0
    if not a.no_lite:
        print("SPT-3G D1 T&E lite (spt_candl_data, commit %s) -> %s" % (SPT_CANDL_COMMIT[:7],
              os.path.join(K.OBSERVATIONS_BASE, "candl")))
        try:
            n = get_spt_lite(force=a.force)
        except Exception as e:
            print(f"Download failed ({e}). By hand: copy spt_candl_data/SPT3G_D1_TnE_v0/lite/ of "
                  f"https://github.com/SouthPoleTelescope/spt_candl_data (commit {SPT_CANDL_COMMIT[:7]}) to "
                  "Observations/candl/SPT3G_D1_TnE_lite/ (see Observations/candl/README.md).")
            return 1
        print(f"  {n} file(s) written, all {len(SPT_LITE_FILES)} checked (md5)")
    if a.muse_zip:
        print("SPT-3G MUSE lensing ->", get_spt_muse(os.path.expanduser(a.muse_zip)), "(md5 checked)")
    print("\nPlease cite, when you use these data:")
    for c in (CITATIONS[:2] if not a.no_lite else ()) + (CITATIONS[2:] if a.muse_zip else ()):
        print("  - " + c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
