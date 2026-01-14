import os, ctypes
from ctypes import c_double, c_int, Structure, POINTER, byref

class _BBNResult(Structure):
    _fields_ = [
        ("eta",     c_double),
        ("Yp",      c_double),
        ("D_H",     c_double),
        ("He3_H",   c_double),
        ("Li7_H",   c_double),
        ("Li6_Li7", c_double),
        ("status",  c_int),
    ]

# --- robust shared-lib loader ---
_here = os.path.dirname(__file__)
env_paths = [os.getenv("KOSMO_BBN_LIB"), os.getenv("ALTERBBN_SO")]

candidates = [p for p in (
    # 1) explicit env vars (highest priority)
    *env_paths,

    # 2) locations relative to this file
    os.path.join(_here, "libkosmo_bbn.so"),
    os.path.join(_here, "build", "libkosmo_bbn.so"),
    os.path.join(_here, "build", "lib", "libkosmo_bbn.so"),

    # 3) your known build location
    "/mnt/d/AlterBBN/libkosmo_bbn.so",

    # 4) last resort: current working dir
    os.path.join(os.getcwd(), "libkosmo_bbn.so"),
) if p]

_tried = []
_lib = None
for p in candidates:
    try:
        if os.path.exists(p):
            _lib = ctypes.CDLL(p)
            _loaded_path = p  # for debugging
            break
        else:
            _tried.append(f"{p} (missing)")
    except OSError as e:
        _tried.append(f"{p} (load error: {e})")

if _lib is None:
    # 5) fall back to linker search (LD_LIBRARY_PATH, rpath, etc.)
    try:
        _lib = ctypes.CDLL("libkosmo_bbn.so")
        _loaded_path = "libkosmo_bbn.so (linker search)"
    except OSError as e:
        msg = [
            "Could not load libkosmo_bbn.so.",
            "Tried paths:",
            *("  - " + s for s in _tried),
            f"Final linker search failed: {e}",
            "Set KOSMO_BBN_LIB=/full/path/to/libkosmo_bbn.so or add its folder to LD_LIBRARY_PATH.",
        ]
        raise OSError("\n".join(msg))

# --- function signature ---
try:
    _run = _lib.kosmo_bbn_run
except AttributeError as e:
    raise RuntimeError(f"Loaded {_loaded_path} but symbol 'kosmo_bbn_run' not found: {e}")

_run.argtypes = [c_double, c_double, c_double, POINTER(_BBNResult)]
_run.restype  = c_int

class AlterBBNError(RuntimeError):
    pass

def run_bbn(Omega_b_h2: float, N_eff: float = 3.046, tau_n: float = 879.4):
    """
    Call the AlterBBN wrapper.
    Returns dict with keys: eta, Yp, D_H, He3_H, Li7_H, Li6_Li7
    """
    out = _BBNResult()
    rc = _run(float(Omega_b_h2), float(N_eff), float(tau_n), byref(out))
    if rc != 0 or out.status == 0:
        raise AlterBBNError(f"AlterBBN call failed (rc={rc}, status={out.status})")
    return {
        "eta": out.eta, "Yp": out.Yp, "D_H": out.D_H,
        "He3_H": out.He3_H, "Li7_H": out.Li7_H, "Li6_Li7": out.Li6_Li7
    }

