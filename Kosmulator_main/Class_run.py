"""
Class_run.py

Builds and caches CLASS (classy) binaries per model and provides
helpers for Planck clik likelihoods (CMB high-ℓ, low-ℓ, lensing).

Public API (used elsewhere in Kosmulator):
  - ensure_class_ready(model_name, ...)
  - get_Class()
  - init_clik_worker(...)
  - preload_clik_hil / preload_clik_lowl / preload_clik_hilTT
  - preload_clik_lensing_raw / preload_clik_lensing_cmbmarg
  - get_clik_hil / get_clik_lowl / get_clik_hilTT
  - get_clik_lensing / get_clik_lensing_cmbmarg
"""

from __future__ import annotations

import hashlib
import json
import glob
import logging
import os
import platform
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import clik
import sysconfig

from Kosmulator_main import utils as U

logger = logging.getLogger("Kosmulator.Class_run")

OBS_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Observations")

# ---------------------------------------------------------------------------
# Global state for clik paths / instances (per process)
# ---------------------------------------------------------------------------

# Paths (set by preload_*); handles are per-process
_clik_path_hil: str | None = None
_clik_path_lowl: str | None = None
_clik_path_lensing: str | None = None
_clik_path_lensing_cmbmarg: str | None = None
_clik_path_hilTT: str | None = None

# Per-process clik instances
_clik_pid: int | None = None
_clik_instance_hil: Optional[clik.clik] = None
_clik_instance_lowl: Optional[clik.clik] = None
_clik_instance_lensing: Optional[clik.clik_lensing] = None
_clik_instance_lensing_cmbmarg: Optional[clik.clik_lensing] = None
_clik_hilTT: Optional[clik.clik] = None

# CLASS build signature cache
_current_class_model: Optional[str] = None
_current_class_hash: Optional[str] = None

# One-time logs for lensing
_DID_LOG_LENSING_RAW = False
_DID_LOG_LENSING_CMBMARG = False


# ---------------------------------------------------------------------------
# Helpers for describing the build environment (used in CLASS cache key)
# ---------------------------------------------------------------------------


def _first_line(cmd: str) -> str:
    """Return the first line of a shell command, or 'unknown' on failure."""
    try:
        out = subprocess.check_output(
            shlex.split(cmd), stderr=subprocess.STDOUT, text=True
        )
        return out.splitlines()[0].strip()
    except Exception:
        return "unknown"


def _python_abi_sig() -> dict:
    """Describe the Python ABI relevant to compiled extensions."""
    return {
        "py": sys.version.split()[0],
        "soabi": sysconfig.get_config_var("SOABI") or "",
        "plat": sysconfig.get_platform(),
    }


def _toolchain_sig() -> dict:
    """Describe the toolchain/libc used to build CLASS."""
    env = os.environ
    return {
        "CCv": _first_line((env.get("CC") or "gcc") + " --version"),
        "CXXv": _first_line((env.get("CXX") or "g++") + " --version"),
        "LDv": _first_line((env.get("LD") or "ld") + " --version"),
        "CONDA_PREFIX": env.get("CONDA_PREFIX", ""),
        "SYSROOT": env.get("SYSROOT", "") or env.get("LDFLAGS", ""),
        "libc": " ".join(platform.libc_ver()),
    }


def _source_tree_hash(model_dir: str) -> str:
    """Stable hash of all build-relevant sources for this CLASS model."""
    h = hashlib.sha256()
    roots = ["source", "tools", "external", "include", "python"]

    # Include Makefile and python/setup.py
    for extra in ("Makefile", os.path.join("python", "setup.py")):
        p = os.path.join(model_dir, extra)
        if os.path.exists(p):
            h.update(extra.encode())
            with open(p, "rb") as f:
                h.update(f.read())

    for root in roots:
        r = os.path.join(model_dir, root)
        if not os.path.isdir(r):
            continue
        for dp, _, fns in os.walk(r):
            for fn in sorted(fns):
                if not fn.endswith((".c", ".h", ".cc", ".cpp", ".hpp", ".py", ".txt")):
                    continue
                fp = os.path.join(dp, fn)
                rel = fp[len(model_dir) :].encode()
                h.update(rel)
                try:
                    with open(fp, "rb") as f:
                        h.update(f.read())
                except Exception:
                    st = os.stat(fp)
                    h.update(str((st.st_mtime_ns, st.st_size)).encode())
    return h.hexdigest()


def _full_signature(model_dir: str) -> dict:
    """Full build signature = source tree + Python ABI + toolchain/libc."""
    return {
        "src": _source_tree_hash(model_dir),
        "pyabi": _python_abi_sig(),
        "tool": _toolchain_sig(),
    }


def _sig_hash(sig_dict: dict) -> str:
    """Hash of the full build signature."""
    return hashlib.sha256(json.dumps(sig_dict, sort_keys=True).encode()).hexdigest()


def _mpi_rank0() -> bool:
    """Return True on MPI rank 0, or always True if mpi4py is unavailable."""
    try:
        from mpi4py import MPI  # local import to avoid hard dependency at import-time

        return MPI.COMM_WORLD.Get_rank() == 0
    except Exception:
        return True  # no MPI → act as rank 0


# ---------------------------------------------------------------------------
# CLASS cache directory helpers
# ---------------------------------------------------------------------------


def _cache_dir_for(model_name: str, src_hash: str) -> str:
    """Cache directory for a given model and signature hash."""
    return os.path.join("./Class", ".cache", model_name, src_hash)


def _find_cached_so(model_name: str, src_hash: str) -> Optional[str]:
    """Find a cached classy .so for this model/signature, if it exists."""
    cd = _cache_dir_for(model_name, src_hash)
    if not os.path.isdir(cd):
        return None
    for dp, _, fns in os.walk(cd):
        for fn in fns:
            if fn.endswith(".so"):
                return os.path.join(dp, fn)
    return None


def _write_stamp(model_name: str, sig_hash: str, sig: dict, artifact_path: str) -> None:
    """Write a small JSON describing the build in the cache directory."""
    d = _cache_dir_for(model_name, sig_hash)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, ".build_stamp.json"), "w") as f:
        json.dump({"signature": sig, "artifact": os.path.basename(artifact_path)}, f, indent=2)


def _lock_path(model_name: str, sig_hash: str) -> str:
    """Filesystem lock path used to serialise CLASS builds across ranks."""
    return os.path.join(_cache_dir_for(model_name, sig_hash), ".lock")


@contextmanager
def _with_lock(model_name: str, sig_hash: str, timeout_s: int = 120):
    """
    Acquire a simple filesystem lock for this model/signature.

    Only one rank/process will hold the lock at a time. Others wait until
    the lock disappears, or until timeout_s is exceeded.
    """
    lp = _lock_path(model_name, sig_hash)
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    start = time.time()
    while True:
        try:
            fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            try:
                yield  # we hold the lock
            finally:
                try:
                    os.remove(lp)
                except OSError:
                    pass
            return
        except FileExistsError:
            if time.time() - start > timeout_s:
                raise RuntimeError("Timeout waiting for CLASS cache lock.")
            time.sleep(0.1)


def _wait_for_artifact(model_name: str, sig_hash: str, timeout_s: int = 180) -> Optional[str]:
    """Wait for another rank/process to build the .so and put it in cache."""
    start = time.time()
    while time.time() - start < timeout_s:
        p = _find_cached_so(model_name, sig_hash)
        if p:
            return p
        time.sleep(0.1)
    return None


# ---------------------------------------------------------------------------
# CLASS build + load helpers
# ---------------------------------------------------------------------------


def find_classy_so(model_name: str) -> str:
    """
    Search under ./Class/<model_name> for a built classy .so.

    We try SOABI-specific names first, then fall back to generic *.so.
    """
    base = os.path.join("./Class", model_name)
    soabi = sysconfig.get_config_var("SOABI") or ""
    patterns = [
        os.path.join(base, "python", "build", "lib.*", f"classy.{soabi}.so"),
        os.path.join(base, "build", "lib.*", f"classy.{soabi}.so"),
        os.path.join(base, "**", f"classy.{soabi}.so"),
        # fallbacks
        os.path.join(base, "python", "build", "lib.*", "classy.cpython-*.so"),
        os.path.join(base, "build", "lib.*", "classy.cpython-*.so"),
        os.path.join(base, "**", "classy.cpython-*.so"),
        os.path.join(base, "**", "*.so"),
    ]
    for pat in patterns:
        hits = sorted(Path(base).glob(pat.replace(base + os.sep, ""))) if "**" in pat else sorted(
            glob.glob(pat, recursive=True)
        )
        if hits:
            return str(hits[0])
    raise FileNotFoundError(f"No classy .so found under {base}")


def _ensure_libclass(model_dir: str) -> None:
    """
    Build only the libclass.a archive, not the 'class' executable.

    Many CLASS Makefiles already provide a 'libclass.a' target.
    """
    try:
        subprocess.run(["make", "-C", model_dir, "libclass.a"], check=True)
    except subprocess.CalledProcessError:
        # Fallback: try a minimal build that may produce libclass.a
        subprocess.run(["make", "-C", model_dir, "class", "-k"], check=False)
        if not os.path.exists(os.path.join(model_dir, "libclass.a")):
            raise


def _strip_mvec_from_setup(py_dir: str) -> None:
    """
    Remove -lmvec and 'mvec' from python/setup.py (conda toolchain lacks libmvec).
    """
    import re
    import shutil as _shutil

    p = os.path.join(py_dir, "setup.py")
    if not os.path.isfile(p):
        return

    with open(p, "r", encoding="utf-8") as f:
        s = f.read()

    s2 = s.replace("-lmvec", "")
    s2 = re.sub(r'(["\'])mvec\1\s*,?\s*', "", s2)  # drop "mvec" or 'mvec'
    s2 = re.sub(r",\s*,", ",", s2)  # clean accidental double commas

    if s2 != s:
        _shutil.copy2(p, p + ".bak")
        with open(p, "w", encoding="utf-8") as f:
            f.write(s2)


def _build_python_extension(model_dir: str) -> None:
    """Build the Python extension for CLASS in model_dir/python."""
    model_dir_path = Path(model_dir)
    _ensure_libclass(str(model_dir_path))
    subprocess.run(["make", "-C", str(model_dir_path), "libclass.a"], check=True)
    _strip_mvec_from_setup(str(model_dir_path / "python"))
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=str(model_dir_path / "python"),
        check=True,
    )


def run_model(model_name: str) -> None:
    """
    Build the CLASS library and Python extension for this model.

    Called internally from ensure_class_ready when we have to rebuild.
    """
    model_dir = Path("./Class") / model_name
    built = False
    for target in ["libclass.a", "lib"]:
        try:
            subprocess.run(["make", "-C", str(model_dir), target], check=True)
            built = True
            break
        except subprocess.CalledProcessError:
            continue

    if not built:
        subprocess.run(["make", "-C", str(model_dir)], check=True)

    _build_python_extension(str(model_dir))


def load_classy_so(so_path: str, *args, **kwargs):
    """
    Load a model-specific CLASS Python extension from `so_path` as module 'classy'.

    This matches the extension's compiled name (PyInit_classy), avoiding
    PyInit__classy mismatches that can happen when loading under a submodule.
    """
    import importlib.machinery
    import importlib.util
    import os
    import sys

    so_dir = os.path.dirname(os.path.abspath(so_path))
    if so_dir not in sys.path:
        sys.path.insert(0, so_dir)

    # Drop any existing 'classy' modules (pip or previous rebuilds)
    for key in list(sys.modules.keys()):
        if key == "classy" or key.startswith("classy."):
            sys.modules.pop(key, None)

    mod_name = "classy"

    # IMPORTANT: make it a *package* so importlib.resources.files("classy") works
    spec = importlib.util.spec_from_file_location(
        mod_name,
        so_path,
        loader=importlib.machinery.ExtensionFileLoader(mod_name, so_path),
        submodule_search_locations=[so_dir],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for '{mod_name}' at: {so_path}")

    mod = importlib.util.module_from_spec(spec)

    # Register early (helps some importlib/resource edge cases)
    sys.modules[mod_name] = mod

    spec.loader.exec_module(mod)  # type: ignore[arg-type]

    # Extra belt-and-braces: ensure it's package-like
    if not hasattr(mod, "__path__"):
        mod.__path__ = [so_dir]  # type: ignore[attr-defined]

    if not hasattr(mod, "Class"):
        raise ImportError("Loaded 'classy' extension lacks attribute 'Class'")

    return mod


def _copy_built_so_to_cache(model_name: str, src_hash: str) -> Optional[str]:
    """Copy the just-built classy .so AND required runtime data into the cache directory."""
    try:
        import shutil as _shutil

        model_dir = os.path.join("./Class", model_name)
        so_path = find_classy_so(model_name)

        dest = _cache_dir_for(model_name, src_hash)
        os.makedirs(dest, exist_ok=True)

        # 1) copy the .so
        target_so = os.path.join(dest, os.path.basename(so_path))
        _shutil.copy2(so_path, target_so)

        # 2) copy CLASS runtime data dirs needed at runtime (BBN tables live here)
        src_external = os.path.join(model_dir, "external")
        dst_external = os.path.join(dest, "external")
        if os.path.isdir(src_external):
            # dirs_exist_ok=True requires py3.8+
            _shutil.copytree(src_external, dst_external, dirs_exist_ok=True)

        # (optional) if you ever need it:
        # src_include = os.path.join(model_dir, "include")
        # dst_include = os.path.join(dest, "include")
        # if os.path.isdir(src_include):
        #     _shutil.copytree(src_include, dst_include, dirs_exist_ok=True)

        return target_so

    except Exception:
        return None


def ensure_class_ready(
    model_name: str,
    *,
    force: bool = False,
    no_rebuild: bool = False,
    announce: bool = False,
) -> bool:
    """
    Ensure the correct classy for `model_name` is loaded.

    - Cache key includes: source tree, Python ABI, toolchain/libc.
    - Rank 0 rebuilds behind a lock; workers wait for the artifact to appear.
    - If the same model+signature is already active in this process, this is a no-op.
    """
    global _current_class_model, _current_class_hash

    model_dir = os.path.join("./Class", model_name)
    sig = _full_signature(model_dir)
    sig_hash = _sig_hash(sig)

    # Fast path: already loaded in this process
    if (not force) and (_current_class_model == model_name) and (_current_class_hash == sig_hash):
        return True

    # Try cache first (any rank can try to load)
    cached = _find_cached_so(model_name, sig_hash)
    if cached and not force:
        try:
            load_classy_so(cached, model_name, sig_hash)
            from classy import Class  # type: ignore

            _ = Class()
            _current_class_model, _current_class_hash = model_name, sig_hash
            msg = f"[CLASS] Using cached binary for {model_name} ({sig_hash[:8]}) @ {os.path.basename(cached)}"
            if announce:
                print(msg, flush=True)
            else:
                logger.debug(msg)
            return True
        except Exception as e:
            logger.warning(
                "[CLASS] Cache import failed for %s: %s — will rebuild if allowed.",
                model_name,
                e,
            )

        # Try existing model-local build (pre-cache behaviour)
    if not force:
        try:
            local_so = find_classy_so(model_name)
            load_classy_so(local_so, model_name, sig_hash)
            from classy import Class  # type: ignore
            _ = Class()
            _current_class_model, _current_class_hash = model_name, sig_hash
            # Optionally promote to cache for next time
            try:
                built_cached = _copy_built_so_to_cache(model_name, sig_hash)
                if built_cached:
                    _write_stamp(model_name, sig_hash, sig, built_cached)
            except Exception:
                pass
            msg = f"[CLASS] Using existing local binary for {model_name} ({sig_hash[:8]})"
            if announce:
                print(msg, flush=True)
            else:
                logger.debug(msg)
            return True
        except Exception:
            pass

    if no_rebuild:
        return False

    # Rebuild path: only rank 0 does the build; others wait for the artifact.
    if _mpi_rank0():
        with _with_lock(model_name, sig_hash):
            cached2 = _find_cached_so(model_name, sig_hash)
            if cached2 and not force:
                path = cached2
            else:
                try:
                    subprocess.run(["make", "clean", "-C", model_dir], check=False)
                except Exception:
                    pass

                run_model(model_name)
                built_cached = _copy_built_so_to_cache(model_name, sig_hash)
                path = built_cached or find_classy_so(model_name)
                _write_stamp(model_name, sig_hash, sig, path)
    else:
        path = _wait_for_artifact(model_name, sig_hash)
        if not path:
            raise RuntimeError("Worker timed out waiting for CLASS build artifact.")

    # Load the artifact we now have, and sanity-check
    try:
        load_classy_so(path, model_name, sig_hash)
        from classy import Class  # type: ignore

        _ = Class()
        _current_class_model, _current_class_hash = model_name, sig_hash
        if path.endswith(".so"):
            msg = f"[CLASS] Ready for {model_name} ({sig_hash[:8]}) @ {os.path.basename(path)}"
            if announce:
                print(msg, flush=True)
            else:
                logger.debug(msg)
        return True
    except Exception as e:
        logger.warning("[CLASS] Post-build import failed for %s: %s", model_name, e)
        return False


def get_Class():
    """
    Return the classy.Class constructor for the currently loaded CLASS model.

    This is a thin wrapper so the rest of Kosmulator can just call get_Class().
    """
    import importlib

    classy = importlib.import_module("classy")
    return classy.Class


# ---------------------------------------------------------------------------
# clik preload / worker initialisation
# ---------------------------------------------------------------------------


def init_clik_worker(
    hil_path: str | None = None,
    lowl_path: str | None = None,
    hiltt_path: str | None = None,
    lens_raw_path: str | None = None,
    lens_cmbmarg_path: str | None = None,
) -> None:
    """
    Initialiser for worker processes that need Planck clik likelihoods.

    - Forces single-threaded BLAS/OpenMP per worker.
    - Optionally preloads paths for CMB high-ℓ, low-ℓ, TT-only, and lensing.
    """
    # keep workers single-threaded
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[var] = "1"

    if hil_path:
        preload_clik_hil(hil_path)
    if lowl_path:
        preload_clik_lowl(lowl_path)
    if hiltt_path:
        preload_clik_hilTT(hiltt_path)
    if lens_raw_path:
        preload_clik_lensing_raw(lens_raw_path)
    if lens_cmbmarg_path:
        preload_clik_lensing_cmbmarg(lens_cmbmarg_path)


def _ensure_process_local() -> None:
    """Reset per-process clik instances when pid changes (e.g. after fork)."""
    global _clik_pid, _clik_instance_hil, _clik_instance_lowl, _clik_instance_lensing
    global _clik_instance_lensing_cmbmarg, _clik_hilTT

    pid = os.getpid()
    if _clik_pid != pid:
        _clik_pid = pid
        _clik_instance_hil = None
        _clik_instance_lowl = None
        _clik_instance_lensing = None
        _clik_instance_lensing_cmbmarg = None
        _clik_hilTT = None


# ---------------------------------------------------------------------------
# clik preload helpers (paths usually come from Config/MCMC_setup)
# ---------------------------------------------------------------------------


def preload_clik_hil(path: str) -> None:
    """Preload path for high-ℓ TTTEEE likelihood."""
    global _clik_path_hil
    _clik_path_hil = U.fast_path_for_clik(path)


def preload_clik_lowl(path: str) -> None:
    """Preload path for low-ℓ EE likelihood."""
    global _clik_path_lowl
    _clik_path_lowl = U.fast_path_for_clik(path)


def preload_clik_lensing_raw(path: str) -> None:
    """
    Preload RAW lensing likelihood path (pp + TT/EE/TE).

    If `path` is a directory, it will be copied to a fast FS via
    utils.fast_path_for_clik.
    """
    global _clik_path_lensing, _clik_instance_lensing
    _clik_path_lensing = U.fast_path_for_clik(path)
    _clik_instance_lensing = None  # force fresh open on first get


def preload_clik_lensing_cmbmarg(path: str) -> None:
    """
    Preload CMB-marginalised lensing likelihood path (pp only).
    """
    global _clik_path_lensing_cmbmarg, _clik_instance_lensing_cmbmarg
    _clik_path_lensing_cmbmarg = U.fast_path_for_clik(path)
    _clik_instance_lensing_cmbmarg = None


def preload_clik_hilTT(hiltt_path: str) -> None:
    """Preload path for high-ℓ TT-only likelihood."""
    global _clik_path_hilTT, _clik_hilTT
    _clik_path_hilTT = U.fast_path_for_clik(hiltt_path)
    _clik_hilTT = None  # reset; will be (re)loaded lazily


# ---------------------------------------------------------------------------
# clik getters (used by Statistical_packages)
# ---------------------------------------------------------------------------


def get_clik_hil():
    """Return cached high-ℓ TTTEEE clik instance (silent)."""
    global _clik_instance_hil
    _ensure_process_local()
    if _clik_instance_hil is not None:
        return _clik_instance_hil
    if _clik_path_hil is None:
        raise RuntimeError("CMB_hil path not set")
    with U.quiet_cstdio():
        _clik_instance_hil = clik.clik(_clik_path_hil)
    return _clik_instance_hil


def get_clik_lowl():
    """Return cached low-ℓ EE clik instance (silent)."""
    global _clik_instance_lowl
    _ensure_process_local()
    if _clik_instance_lowl is not None:
        return _clik_instance_lowl
    if _clik_path_lowl is None:
        raise RuntimeError("CMB_lowl path not set")
    with U.quiet_cstdio():
        _clik_instance_lowl = clik.clik(_clik_path_lowl)
    return _clik_instance_lowl


def get_clik_lensing():
    """Planck 2018 lensing likelihood (RAW: expects pp + TT/EE/TE)."""
    global _clik_instance_lensing, _DID_LOG_LENSING_RAW
    _ensure_process_local()
    if _clik_instance_lensing is not None:
        return _clik_instance_lensing

    # Use preloaded path if provided, else fall back to default under Observations
    if _clik_path_lensing is not None:
        path_raw = _clik_path_lensing
    else:
        path_raw = os.path.join(
            OBS_BASE,
            "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing",
        )

    with U.quiet_cstdio():
        _clik_instance_lensing = clik.clik_lensing(path_raw)

    if not _DID_LOG_LENSING_RAW:
        logger.info("Loaded lensing (RAW) clik from '%s'", path_raw)
        _DID_LOG_LENSING_RAW = True

    return _clik_instance_lensing


def get_clik_lensing_cmbmarg():
    """Planck 2018 lensing likelihood (CMB-marged: expects pp only)."""
    global _clik_instance_lensing_cmbmarg, _DID_LOG_LENSING_CMBMARG
    _ensure_process_local()
    if _clik_instance_lensing_cmbmarg is not None:
        return _clik_instance_lensing_cmbmarg

    if _clik_path_lensing_cmbmarg is not None:
        path_marg = _clik_path_lensing_cmbmarg
    else:
        path_marg = os.path.join(
            OBS_BASE,
            "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_CMBmarged.clik_lensing",
        )

    with U.quiet_cstdio():
        _clik_instance_lensing_cmbmarg = clik.clik_lensing(path_marg)

    if not _DID_LOG_LENSING_CMBMARG:
        logger.info("Loaded lensing (CMB-marged) clik from '%s'", path_marg)
        _DID_LOG_LENSING_CMBMARG = True

    return _clik_instance_lensing_cmbmarg


def get_clik_hilTT():
    """Return cached high-ℓ TT-only clik instance (silent)."""
    global _clik_hilTT
    _ensure_process_local()
    if _clik_hilTT is not None:
        return _clik_hilTT
    if _clik_path_hilTT is None:
        raise RuntimeError("CMB_hil_TT path not set. Call preload_clik_hilTT first.")
    with U.quiet_cstdio():
        _clik_hilTT = clik.clik(_clik_path_hilTT)
    return _clik_hilTT
