"""
rd_Class: the sound horizon r_d from CLASS with the model's own expansion history.

Why
---
BAO/DESI groups with an early-time calibrator (BBN or Planck) compute r_d instead of sampling
it. Kosmulator did that with CLASS for flat LCDM (rd_helpers._rd_class_core), whatever the
model: right for LCDM, but a model whose dark sector still matters before recombination
(w0wa with w0 + wa near 0, f(T), early dark energy) has a different H(z > 1000) and so a
different r_d (w0 = -0.9, wa = 0.85: r_d 1.45% below the LCDM value).

How
---
One extra CLASS build, Class/rd_Class, made from Class/LCDM_v with a small hook: CLASS's
cosmological-constant slot carries a tabulated dark sector instead of a constant,

    rho_slot(a) = (3 H0^2 / 8 pi G) [X(a) + delta],   X(z) = E(z)^2 - Omega_m (1+z)^3 - Omega_r (1+z)^4,
    p_slot      = -rho_slot - (1/3) d rho_slot / d ln a,

with E(z) the model's own E(z) (User_defined_modules), Omega_m and Omega_r as that E(z) uses
them, and delta = Omega_Lambda(CLASS budget) - X(a = 1), of order 1e-6, so that H(a = 1) = H0
exactly. CLASS keeps its own photons, baryons, CDM and neutrinos (the massive neutrino is
relativistic at recombination, while the late-time E(z) treats it as matter: subtracting
Kosmulator's own Omega_m and Omega_r removes that), so only the model's extra part comes from
the table. For LCDM, X = 1 - Omega_m - Omega_r and rd_Class gives CLASS's LCDM r_d.

Assumptions: the model changes the expansion history only (no change to G for the
baryon-photon fluid, to N_eff, BBN or recombination physics), and its E(z) is valid up to
z = RD_TABLE_ZMAX (beyond that the slot is held constant, like a cosmological constant).

Build cache
-----------
The tree is made once (and again only if Class/LCDM_v's sources or the hook change) and
compiled once: the classy .so goes to Class/.cache/rd_Class/<signature>/ with the same
signature as every Kosmulator CLASS build (sources, Python ABI, toolchain), so later runs
load it without compiling. Once loaded in a process it is kept (no hashing per call). It is
loaded under its own handle, next to whichever model build is sys.modules["classy"].
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import logging
import os
import shutil
import tempfile
from functools import lru_cache
from typing import Callable, Optional, Tuple

import numpy as np

from Kosmulator_main import constants as K

logger = logging.getLogger("Kosmulator.rd_class")

RD_CLASS_NAME = "rd_Class"
RD_CLASS_SOURCE = "LCDM_v"
RD_HOOK_VERSION = "1"            # change when the C edits below change
RD_TABLE_ZMAX = 1.0e7            # the table covers 0 <= z <= RD_TABLE_ZMAX
RD_TABLE_ROWS = 2000             # rows, uniform in ln(1+z) (plus z = 0)
RD_TABLE_MAX_ROWS = 4096         # what the C reader accepts

_CLASS_ROOT = "./Class"
_MODULE = None                   # the loaded rd_Class classy (this process)
_MODULE_SO: Optional[str] = None


# ---------------------------------------------------------------------------
# The C hook: (file, anchor, replacement), each anchor must occur exactly once
# ---------------------------------------------------------------------------
_EDITS = [
    ("include/background.h",
     "  double Omega0_lambda;    /**< \\f$ \\Omega_{0_\\Lambda} \\f$: cosmological constant */\n",
     "  double Omega0_lambda;    /**< \\f$ \\Omega_{0_\\Lambda} \\f$: cosmological constant */\n"
     "  int rd_tab_size;         /**< Kosmulator rd_Class: rows of the tabulated dark sector, 0 = off */\n"
     "  double * rd_tab_loga;    /**< rd_Class: ln a, increasing, last row ln a = 0 */\n"
     "  double * rd_tab_X;       /**< rd_Class: X = rho_slot / H0^2 without the closure constant */\n"
     "  double * rd_tab_dX;      /**< rd_Class: dX / d ln a */\n"
     "  double * rd_tab_d2X;     /**< rd_Class: d^2 X / d ln a^2 */\n"
     "  double rd_tab_delta;     /**< rd_Class: Omega0_lambda - X(a = 1), budget closure */\n"),
    ("source/input.c",
     "  /* ** END OF BUDGET EQUATION ** */\n",
     "  /* ** END OF BUDGET EQUATION ** */\n"
     "\n"
     "  /** Kosmulator rd_Class: the Lambda slot carries a tabulated dark sector X(a) (file rd_table_file,\n"
     "      four columns: ln a increasing to 0, X, dX/dln a, d2X/dln a2) */\n"
     "  class_call(parser_read_string(pfc,\"rd_table_file\",&string1,&flag1,errmsg),errmsg,errmsg);\n"
     "  if (flag1 == _TRUE_) {\n"
     "    FILE * rd_fp; int rd_n = 0, rd_i; int rd_cap = " + str(RD_TABLE_MAX_ROWS) + ";\n"
     "    double rd_c1, rd_c2, rd_c3, rd_c4;\n"
     "    class_test(pba->Omega0_lambda == 0., errmsg, \"rd_table_file: Omega_Lambda must come from the budget closure\");\n"
     "    class_open(rd_fp, string1, \"r\", errmsg);\n"
     "    class_alloc(pba->rd_tab_loga, rd_cap*sizeof(double), errmsg);\n"
     "    class_alloc(pba->rd_tab_X, rd_cap*sizeof(double), errmsg);\n"
     "    class_alloc(pba->rd_tab_dX, rd_cap*sizeof(double), errmsg);\n"
     "    class_alloc(pba->rd_tab_d2X, rd_cap*sizeof(double), errmsg);\n"
     "    pba->rd_tab_size = rd_cap;  /* so that background_free_input frees the arrays from here on */\n"
     "    while (fscanf(rd_fp, \"%lf %lf %lf %lf\", &rd_c1, &rd_c2, &rd_c3, &rd_c4) == 4) {\n"
     "      if (rd_n >= rd_cap) { fclose(rd_fp); class_stop(errmsg, \"rd_table_file: more than %d rows\", rd_cap); }\n"
     "      pba->rd_tab_loga[rd_n] = rd_c1; pba->rd_tab_X[rd_n] = rd_c2;\n"
     "      pba->rd_tab_dX[rd_n] = rd_c3; pba->rd_tab_d2X[rd_n] = rd_c4; rd_n++;\n"
     "    }\n"
     "    fclose(rd_fp);\n"
     "    class_test(rd_n < 2, errmsg, \"rd_table_file %s: fewer than 2 rows\", string1);\n"
     "    for (rd_i = 1; rd_i < rd_n; rd_i++)\n"
     "      class_test(pba->rd_tab_loga[rd_i] <= pba->rd_tab_loga[rd_i-1], errmsg, \"rd_table_file: ln a must increase\");\n"
     "    class_test(fabs(pba->rd_tab_loga[rd_n-1]) > 1.e-12, errmsg, \"rd_table_file: the last row must be ln a = 0\");\n"
     "    pba->rd_tab_size = rd_n;\n"
     "    pba->rd_tab_delta = pba->Omega0_lambda - pba->rd_tab_X[rd_n-1];\n"
     "  }\n"),
    ("source/input.c",
     "  pba->Omega0_lambda = 1.-pba->Omega0_k-pba->Omega0_g-pba->Omega0_ur-pba->Omega0_b-pba->Omega0_cdm"
     "-pba->Omega0_ncdm_tot-pba->Omega0_dcdmdr - pba->Omega0_idr -pba->Omega0_idm;\n",
     "  pba->Omega0_lambda = 1.-pba->Omega0_k-pba->Omega0_g-pba->Omega0_ur-pba->Omega0_b-pba->Omega0_cdm"
     "-pba->Omega0_ncdm_tot-pba->Omega0_dcdmdr - pba->Omega0_idr -pba->Omega0_idm;\n"
     "  pba->rd_tab_size = 0; pba->rd_tab_loga = NULL; pba->rd_tab_X = NULL;   /* Kosmulator rd_Class: off */\n"
     "  pba->rd_tab_dX = NULL; pba->rd_tab_d2X = NULL; pba->rd_tab_delta = 0.;\n"),
    ("source/background.c",
     "int background_functions(\n",
     "/* Kosmulator rd_Class: the tabulated dark sector at ln a = la (cubic Hermite in X with the\n"
     "   tabulated dX; linear in dX and d2X); outside the table X is held at its end values */\n"
     "static void rd_tab_eval(struct background *pba, double la, double *X, double *dX, double *d2X) {\n"
     "  int n = pba->rd_tab_size, lo = 0, hi = n - 1, mid;\n"
     "  double h, t, h00, h10, h01, h11;\n"
     "  if (la <= pba->rd_tab_loga[0]) { *X = pba->rd_tab_X[0]; *dX = 0.; *d2X = 0.; return; }\n"
     "  if (la >= pba->rd_tab_loga[n-1]) { *X = pba->rd_tab_X[n-1]; *dX = pba->rd_tab_dX[n-1]; *d2X = pba->rd_tab_d2X[n-1]; return; }\n"
     "  while (hi - lo > 1) { mid = (lo + hi) / 2; if (pba->rd_tab_loga[mid] > la) hi = mid; else lo = mid; }\n"
     "  h = pba->rd_tab_loga[hi] - pba->rd_tab_loga[lo]; t = (la - pba->rd_tab_loga[lo]) / h;\n"
     "  h00 = (1. + 2.*t)*(1. - t)*(1. - t); h10 = t*(1. - t)*(1. - t); h01 = t*t*(3. - 2.*t); h11 = t*t*(t - 1.);\n"
     "  *X = h00*pba->rd_tab_X[lo] + h10*h*pba->rd_tab_dX[lo] + h01*pba->rd_tab_X[hi] + h11*h*pba->rd_tab_dX[hi];\n"
     "  *dX = (1. - t)*pba->rd_tab_dX[lo] + t*pba->rd_tab_dX[hi];\n"
     "  *d2X = (1. - t)*pba->rd_tab_d2X[lo] + t*pba->rd_tab_d2X[hi];\n"
     "}\n"
     "\n"
     "int background_functions(\n"),
    ("source/background.c",
     "  /* Lambda */\n"
     "  if (pba->has_lambda == _TRUE_) {\n"
     "    pvecback[pba->index_bg_rho_lambda] = pba->Omega0_lambda * pow(pba->H0,2);\n"
     "    rho_tot += pvecback[pba->index_bg_rho_lambda];\n"
     "    p_tot -= pvecback[pba->index_bg_rho_lambda];\n"
     "  }\n",
     "  /* Lambda */\n"
     "  if ((pba->has_lambda == _TRUE_) && (pba->rd_tab_size > 1)) {\n"
     "    /* Kosmulator rd_Class: rho = H0^2 [X(a) + delta], p = -rho - (1/3) d rho / d ln a */\n"
     "    double rd_X, rd_dX, rd_d2X, rd_H02 = pow(pba->H0,2);\n"
     "    rd_tab_eval(pba, log(a), &rd_X, &rd_dX, &rd_d2X);\n"
     "    pvecback[pba->index_bg_rho_lambda] = rd_H02 * (rd_X + pba->rd_tab_delta);\n"
     "    rho_tot += pvecback[pba->index_bg_rho_lambda];\n"
     "    p_tot += -pvecback[pba->index_bg_rho_lambda] - rd_H02 * rd_dX / 3.;\n"
     "    dp_dloga += -rd_H02 * rd_dX - rd_H02 * rd_d2X / 3.;\n"
     "  }\n"
     "  else if (pba->has_lambda == _TRUE_) {\n"
     "    pvecback[pba->index_bg_rho_lambda] = pba->Omega0_lambda * pow(pba->H0,2);\n"
     "    rho_tot += pvecback[pba->index_bg_rho_lambda];\n"
     "    p_tot -= pvecback[pba->index_bg_rho_lambda];\n"
     "  }\n"),
    ("source/background.c",
     "int background_free_input(\n"
     "                          struct background *pba\n"
     "                          ) {\n"
     "\n"
     "  int k;\n",
     "int background_free_input(\n"
     "                          struct background *pba\n"
     "                          ) {\n"
     "\n"
     "  int k;\n"
     "\n"
     "  if (pba->rd_tab_size > 0) {   /* Kosmulator rd_Class */\n"
     "    free(pba->rd_tab_loga); free(pba->rd_tab_X); free(pba->rd_tab_dX); free(pba->rd_tab_d2X);\n"
     "    pba->rd_tab_size = 0;\n"
     "  }\n"),
]

# Build products and caches not copied from Class/LCDM_v
_IGNORE = shutil.ignore_patterns("*.o", "*.so", "*.a", "build", "__pycache__", ".cache", "class",
                                 "classy.c", "classy.cpp", "*.bak")


def _tree_stamp(src_dir: str) -> str:
    from Kosmulator_main import Class_run as CR
    edits = "".join(f + a + r for f, a, r in _EDITS)
    return hashlib.sha256((CR._source_tree_hash(src_dir) + RD_HOOK_VERSION + edits).encode()).hexdigest()


def make_tree(root: str = _CLASS_ROOT) -> str:
    """Class/rd_Class from Class/LCDM_v plus the hook; remade only when either changed."""
    src = os.path.join(root, RD_CLASS_SOURCE)
    dst = os.path.join(root, RD_CLASS_NAME)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"rd_Class is made from {src}, which is missing")
    stamp = _tree_stamp(src)
    stamp_file = os.path.join(dst, ".rd_class_stamp")
    if os.path.isfile(stamp_file) and open(stamp_file).read().strip() == stamp:
        return dst
    tmp = dst + f".tmp{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, ignore=_IGNORE, symlinks=True)
    for rel, anchor, repl in _EDITS:
        p = os.path.join(tmp, rel)
        s = open(p, encoding="utf-8").read()
        if s.count(anchor) != 1:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError(f"rd_Class: the hook does not fit {RD_CLASS_SOURCE}/{rel} "
                               f"({s.count(anchor)} matches for its anchor)")
        open(p, "w", encoding="utf-8").write(s.replace(anchor, repl))
    with open(os.path.join(tmp, ".rd_class_stamp"), "w") as f:
        f.write(stamp + "\n")
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)
    logger.info("[rd_Class] source tree made from %s (hook version %s)", RD_CLASS_SOURCE, RD_HOOK_VERSION)
    return dst


def _load_private(so_path: str):
    """Load a classy .so without touching sys.modules['classy'] (the model's own CLASS build)."""
    so_dir = os.path.dirname(os.path.abspath(so_path))
    spec = importlib.util.spec_from_file_location(
        "classy", so_path, loader=importlib.machinery.ExtensionFileLoader("classy", so_path),
        submodule_search_locations=[so_dir])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # type: ignore[union-attr]
    if not hasattr(mod, "Class"):
        raise ImportError(f"{so_path} has no Class")
    return mod


def ensure_ready(announce: bool = False, build: bool = True):
    """
    The rd_Class classy module, loaded once per process. Order: already loaded -> cached .so
    for the current sources (Class/.cache/rd_Class/<signature>) -> compile (only if `build`;
    the main process does this before any worker pool exists) -> None if not possible.
    """
    global _MODULE, _MODULE_SO
    if _MODULE is not None:
        return _MODULE
    from Kosmulator_main import Class_run as CR
    tree = make_tree()
    sig = CR._full_signature(tree)
    sig_hash = CR._sig_hash(sig)
    so = CR._find_cached_so(RD_CLASS_NAME, sig_hash)
    if so is None:
        if not build:
            return None
        if CR._mpi_rank0():
            with CR._with_lock(RD_CLASS_NAME, sig_hash, timeout_s=900):
                so = CR._find_cached_so(RD_CLASS_NAME, sig_hash)
                if so is None:
                    msg = "[rd_Class] compiling CLASS with the tabulated E(z) hook (once; cached afterwards)"
                    print(msg, flush=True) if announce else logger.info(msg)
                    CR.run_model(RD_CLASS_NAME)
                    so = CR._copy_built_so_to_cache(RD_CLASS_NAME, sig_hash) or CR.find_classy_so(RD_CLASS_NAME)
                    CR._write_stamp(RD_CLASS_NAME, sig_hash, sig, so)
        else:
            so = CR._wait_for_artifact(RD_CLASS_NAME, sig_hash, timeout_s=900)
            if so is None:
                raise RuntimeError("rd_Class: timed out waiting for rank 0's build")
    _MODULE = _load_private(so)
    _MODULE_SO = so
    msg = f"[rd_Class] using {os.path.relpath(so)} ({sig_hash[:8]})"
    print(msg, flush=True) if announce else logger.debug(msg)
    return _MODULE


# ---------------------------------------------------------------------------
# The table and r_d
# ---------------------------------------------------------------------------
_Z = np.concatenate([[0.0], np.expm1(np.linspace(np.log1p(1e-4), np.log1p(RD_TABLE_ZMAX), RD_TABLE_ROWS - 1))])


def _model_E(model_func: Callable, z: np.ndarray, p: dict) -> np.ndarray:
    try:
        E = np.asarray(model_func(z, p), dtype=float)
        if E.shape == z.shape:
            return E
    except Exception:
        pass
    return np.array([float(model_func(float(zi), p)) for zi in z])   # scalar-only user models


def dark_sector_table(model_func: Callable, p: dict) -> Tuple[np.ndarray, ...]:
    """ln a, X, dX/dln a, d2X/dln a2 (ln a increasing to 0) for the model's E(z) at p."""
    from scipy.interpolate import CubicSpline
    import User_defined_modules as UDM
    from Kosmulator_main.utils import ensure_background_params
    q = ensure_background_params(dict(p))
    E = _model_E(model_func, _Z, q)
    if not np.all(np.isfinite(E)) or np.any(E <= 0):
        raise ValueError("the model's E(z) is not finite and positive up to z = %.0e" % RD_TABLE_ZMAX)
    Om = float(q["Omega_m"])
    Or = float(UDM.radiation_density(q))
    zp1 = 1.0 + _Z
    std = Om * zp1 ** 3 + Or * zp1 ** 4
    X = E * E - std
    # The model must contain matter and radiation as E(z) is written in Kosmulator (the table
    # replaces only the rest): X may not cancel them at early times
    early = (_Z > 1e3) & (_Z < 1e5)
    if np.min(X[early] / std[early]) < -0.5:
        raise ValueError("the model's E(z)^2 falls below Omega_m (1+z)^3 + Omega_r (1+z)^4 at z > 1000 "
                         "(does it include radiation, User_defined_modules.radiation_density?)")
    la = -np.log(zp1)[::-1]
    Xr = X[::-1]
    cs = CubicSpline(la, Xr)
    return la, Xr, cs(la, 1), cs(la, 2)


def _table_path() -> str:
    d = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
    return os.path.join(d, f"kosm_rd_table_{os.getpid()}.txt")


def write_table(model_func: Callable, p: dict) -> str:
    la, X, dX, d2X = dark_sector_table(model_func, p)
    path = _table_path()
    with open(path, "w") as f:
        f.write("\n".join(f"{a:.17e} {b:.17e} {c:.17e} {d:.17e}" for a, b, c, d in zip(la, X, dX, d2X)))
        f.write("\n")
    return path


def class_inputs(p: dict) -> dict:
    """CLASS inputs of the standard species, as rd_helpers.compute_rd_class sets them."""
    from Kosmulator_main.utils import ensure_background_params
    q = ensure_background_params(dict(p))
    h = float(q["H_0"]) / 100.0
    obh2 = float(q["Omega_bh^2"])
    if "Omega_dh^2" in q:
        ocdmh2 = float(q["Omega_dh^2"])
    else:
        ocdmh2 = max(float(q["Omega_m"]) * h * h - obh2 - K.omega_nu(), 0.0)
    nu = K.class_neutrino_params()
    m_ncdm = float(nu.get("m_ncdm", 0.0))
    n_eff = float(q.get("N_eff", K.NU_N_EFF))
    pars = {"h": h, "omega_b": obh2, "omega_cdm": ocdmh2,
            "N_ur": float(q.get("N_ur", n_eff - (K.NU_NEFF_PER_MASSIVE if m_ncdm else 0.0))),
            "T_cmb": float(q.get("T_CMB", K.T_CMB_DEFAULT)), "Omega_k": float(q.get("Omega_k", 0.0)),
            "YHe": "BBN", "output": ""}
    if m_ncdm:
        pars.update({"N_ncdm": 1, "m_ncdm": m_ncdm})
    if "tau_n" in q:
        pars["tau_neutron"] = float(q["tau_n"])
    return pars


def rd_model(model_name: str, p: dict, model_func: Optional[Callable] = None) -> float:
    """r_d [Mpc] from rd_Class with the model's E(z) at p (cached per point)."""
    key = tuple(sorted((k, float(v)) for k, v in p.items() if isinstance(v, (int, float, np.floating))))
    return _rd_model_cached(model_name, key, model_func)


@lru_cache(maxsize=4096)
def _rd_model_cached(model_name: str, key: tuple, model_func: Optional[Callable]) -> float:
    mod = ensure_ready(build=False)
    if mod is None:
        raise RuntimeError("rd_Class is not built (MCMC_setup builds it at start-up)")
    if model_func is None:
        import User_defined_modules as UDM
        model_func = UDM.Get_model_function(model_name)
    p = dict(key)
    pars = class_inputs(p)
    pars["rd_table_file"] = write_table(model_func, p)
    c = mod.Class()
    try:
        c.set(pars)
        c.compute()
        return float(c.rs_drag())
    finally:
        try:
            c.struct_cleanup()
            c.empty()
        except Exception:
            pass
