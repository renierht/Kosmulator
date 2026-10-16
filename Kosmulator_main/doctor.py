"""
kosmulator-doctor: check, and with --setup finish, a Kosmulator installation.

    kosmulator-doctor                   what is installed, what is missing and the command that adds it
                                        (a few seconds; nothing is built or changed)
    kosmulator-doctor --setup           the same, then builds CLASS for LCDM_v and rd_Class, loads every
                                        likelihood code with its data and makes a short test run
                                        (a few minutes, once after installing)
    kosmulator-doctor --build-class M   builds Class/M now, so that its compiler errors show before a run

Everything is done inside the Kosmulator folder: CLASS builds go to Class/.cache (as in a run), their
compiler output to Class/<model>/build.log, and the test run's outputs are removed afterwards
(--keep-test keeps them). The exit status is 0 when everything required is in place.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

OK, MISSING, WARN, INFO, SKIP = "OK", "MISSING", "WARN", "INFO", "SKIP"

# Core packages (requirements.txt): import name, pip name
CORE = [("numpy", "numpy"), ("scipy", "scipy"), ("matplotlib", "matplotlib"), ("h5py", "h5py"),
        ("pandas", "pandas"), ("getdist", "getdist"), ("emcee", "emcee"), ("zeus", "zeus-mcmc"),
        ("tqdm", "tqdm"), ("threadpoolctl", "threadpoolctl")]

QUICK_TEST_SUFFIX = "kosmulator_doctor_test"
QUICK_TEST_OUTPUTS = ("MCMC_Chains", os.path.join("Plots", "Saved_Plots"), "Statistical_analysis_tables")

# Planck 2018 best-fit LCDM (Planck 2018 VI, TT,TE,EE+lowE+lensing), for the r_d check only
RD_CHECK_POINT = {"H_0": 67.36, "Omega_m": 0.3153, "Omega_bh^2": 0.02237}


@dataclass
class Result:
    feature: str
    name: str
    status: str
    detail: str = ""
    fix: str = ""
    required: bool = False


class Report:
    """Prints each result as it comes (the setup takes minutes) and keeps them for the summary."""

    def __init__(self) -> None:
        self.items: List[Result] = []
        self._section = ""

    def section(self, title: str) -> None:
        self._section = title
        print(f"\n{title}", flush=True)

    def add(self, feature: str, name: str, status: str, detail: str = "", fix: str = "",
            required: bool = False) -> Result:
        r = Result(feature, name, status, detail, fix, required)
        self.items.append(r)
        line = f"  [{status:^7}] {name}"
        if detail:
            line += f": {detail}"
        print(textwrap.fill(line, width=100, subsequent_indent=" " * 12, break_on_hyphens=False,
                            break_long_words=False), flush=True)
        if fix and status in (MISSING, WARN):
            print(textwrap.indent(fix.strip(), " " * 12 + "> "), flush=True)
        return r

    def required_ok(self) -> bool:
        return all(r.status != MISSING for r in self.items if r.required)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _version(mod_name: str, dist_name: Optional[str] = None) -> Optional[str]:
    """Version of an importable module, or None if it does not import."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):   # candl prints a notice on import
            m = importlib.import_module(mod_name)
    except Exception:
        return None
    v = getattr(m, "__version__", None)
    if v is None and dist_name:
        try:
            from importlib.metadata import version
            v = version(dist_name)
        except Exception:
            v = None
    return str(v) if v is not None else "?"


def _first_line(cmd: List[str], env: Optional[dict] = None) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except Exception:
        return "?"


def _in_conda() -> bool:
    return bool(os.environ.get("CONDA_PREFIX"))


def _install_hint(conda: str = "", pip: str = "", apt: str = "", mac: str = "") -> str:
    """The install command(s) for this kind of environment."""
    lines = []
    if conda and _in_conda():
        lines.append(conda)
    if pip:
        lines.append(pip)
    if apt and sys.platform.startswith("linux") and not (conda and _in_conda()):
        lines.append(apt)
    if mac and sys.platform == "darwin":
        lines.append(mac)
    return "\n".join(lines)


@contextlib.contextmanager
def _output_to(path: Path):
    """Send everything written to stdout/stderr (this process and the build commands it starts)
    to `path` for the duration of the block."""
    sys.stdout.flush()
    sys.stderr.flush()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        saved = os.dup(1), os.dup(2)
        try:
            os.dup2(f.fileno(), 1)
            os.dup2(f.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def _tail(path: Path, n: int = 12) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-n:])


def _build_failure_hint(log: Path) -> str:
    """The likely fix for a failed CLASS build, from its log."""
    text = ""
    try:
        text = log.read_text(errors="replace")
    except Exception:
        pass
    checks = [
        (("No such file or directory: 'gcc'", "gcc: not found", "gcc: command not found"),
         "no C compiler called gcc: " + _compiler_fix()),
        (("Python.h: No such file",), "Python headers missing: " + _install_hint(
            conda="conda install -c conda-forge python", apt="sudo apt install python3-dev")),
        (("No module named 'Cython'", "No module named Cython"), "pip install cython"),
        (("No module named 'setuptools'",), "pip install setuptools"),
        (("numpy/arrayobject.h",), "NumPy headers missing: pip install --force-reinstall numpy"),
    ]
    for needles, fix in checks:
        if any(s in text for s in needles):
            return fix
    return f"see the full log: {log}"


def _compiler_fix() -> str:
    return _install_hint(conda="conda install -c conda-forge c-compiler cxx-compiler make",
                         apt="sudo apt install build-essential python3-dev",
                         mac="xcode-select --install")


def find_root() -> Optional[Path]:
    """The Kosmulator folder (with Kosmulator.py, Class/ and Kosmulator_main/): the one this package
    is in (pip install -e .), else the current folder or one above it."""
    here = Path(__file__).resolve().parent.parent
    for c in [here, Path.cwd(), *Path.cwd().parents]:
        if (c / "Kosmulator.py").is_file() and (c / "Class").is_dir() and (c / "Kosmulator_main").is_dir():
            return c
    return None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_core(rep: Report) -> None:
    rep.section("Core (required for every run)")
    v = sys.version_info
    ok_py = (3, 9) <= v[:2] < (3, 13)
    rep.add("core", "Python", OK if ok_py else MISSING, f"{sys.version.split()[0]} ({sys.executable})",
            fix="Kosmulator needs Python 3.9 - 3.12 (tested: 3.11); conda env create -f environment.yml "
                "makes a 3.11 environment", required=True)
    have, missing = [], []
    for mod, dist in CORE:
        ver = _version(mod, dist)
        (have if ver else missing).append(f"{mod} {ver}" if ver else dist)
    if have:
        rep.add("core", "Packages", OK, ", ".join(have))
    if missing:
        rep.add("core", "Packages missing", MISSING, ", ".join(missing),
                fix='pip install -e .   (from the Kosmulator folder; installs the core packages)', required=True)


def check_build_tools(rep: Report) -> bool:
    """What Kosmulator needs to compile CLASS (per model, from Class/<model>). True if all present."""
    rep.section("CLASS build tools (required for CMB groups and for r_d in BAO/DESI + BBN/CMB groups)")
    from Kosmulator_main import Class_run as CR
    env = CR._build_env()
    ok_all = True
    gcc = shutil.which("gcc", path=env.get("PATH"))
    gxx = shutil.which("g++", path=env.get("PATH"))
    if gcc and gxx:
        rep.add("class", "C/C++ compiler", OK, _first_line([gcc, "--version"], env))
    else:
        ok_all = False
        rep.add("class", "C/C++ compiler (gcc, g++)", MISSING, "not found", fix=_compiler_fix(), required=True)
    for tool in ("make", "ar"):
        if shutil.which(tool, path=env.get("PATH")):
            rep.add("class", tool, OK, _first_line([tool, "--version"], env))
        else:
            ok_all = False
            rep.add("class", tool, MISSING, "not found", fix=_install_hint(
                conda=f"conda install -c conda-forge {'make' if tool == 'make' else 'binutils'}",
                apt="sudo apt install build-essential", mac="xcode-select --install"), required=True)
    for mod, dist in (("Cython", "cython"), ("setuptools", "setuptools")):
        ver = _version(mod, dist)
        if ver:
            rep.add("class", mod, OK, ver)
        else:
            ok_all = False
            rep.add("class", mod, MISSING, "not installed", fix=f"pip install {dist}", required=True)
    py_h = Path(sysconfig.get_paths()["include"]) / "Python.h"
    if py_h.is_file():
        rep.add("class", "Python headers", OK, str(py_h.parent))
    else:
        ok_all = False
        rep.add("class", "Python headers", MISSING, f"{py_h} not found", required=True,
                fix=_install_hint(conda="conda install -c conda-forge python",
                                  apt=f"sudo apt install python{sys.version_info[0]}.{sys.version_info[1]}-dev"))
    if gcc and ok_all:
        # A one-line C program, so that a compiler without its C library is caught here
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "t.c"
            src.write_text("#include <stdio.h>\nint main(void){return 0;}\n")
            r = subprocess.run([gcc, str(src), "-o", str(Path(d) / "t")], capture_output=True, text=True,
                               env=env)
        if r.returncode == 0:
            rep.add("class", "Compiler test", OK, "a test C program compiles")
        else:
            ok_all = False
            rep.add("class", "Compiler test", MISSING, (r.stderr or r.stdout).strip()[-300:],
                    fix=_compiler_fix(), required=True)
    return ok_all


def _class_models(root: Path) -> List[str]:
    return sorted(p.name for p in (root / "Class").iterdir()
                  if p.is_dir() and not p.name.startswith(".") and p.name != "rd_Class"
                  and (p / "Makefile").is_file())


def _cached(model: str, model_dir: str) -> bool:
    from Kosmulator_main import Class_run as CR
    try:
        return CR._find_cached_so(model, CR._sig_hash(CR._full_signature(model_dir))) is not None
    except Exception:
        return False


def build_class(rep: Report, root: Path, model: str, verbose: bool = False) -> bool:
    """Build (or find in the cache) Class/<model> and load it, as a run would."""
    from Kosmulator_main import Class_run as CR
    model_dir = os.path.join(".", "Class", model)
    if not os.path.isfile(os.path.join(model_dir, "Makefile")):
        rep.add("class", f"CLASS {model}", MISSING, f"Class/{model} is not a CLASS source folder", required=True)
        return False
    if _cached(model, model_dir):
        ok = CR.ensure_class_ready(model, no_rebuild=True)
        rep.add("class", f"CLASS {model}", OK if ok else MISSING,
                "built earlier (cache)" if ok else "cached build does not load", required=True)
        if ok:
            return True
    log = root / "Class" / model / "build.log"
    print(f"  ... building CLASS for {model} (about 1 - 3 min; output in Class/{model}/build.log)", flush=True)
    t0 = time.time()
    ok, err = False, ""
    ctx = contextlib.nullcontext() if verbose else _output_to(log)
    try:
        with ctx:
            ok = bool(CR.ensure_class_ready(model))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    if ok:
        rep.add("class", f"CLASS {model}", OK, f"built in {time.time() - t0:.0f} s")
        return True
    rep.add("class", f"CLASS {model}", MISSING, "build failed" + (f" ({err})" if err else ""),
            fix=_build_failure_hint(log) + ("\nlast lines of the log:\n" + _tail(log) if log.is_file() else ""),
            required=True)
    return False


def build_rd_class(rep: Report, root: Path, verbose: bool = False) -> bool:
    from Kosmulator_main import rd_class as RD
    tree = Path(RD.make_tree())
    if _cached(RD.RD_CLASS_NAME, str(tree)):
        mod = RD.ensure_ready(build=False)
        rep.add("class", "rd_Class (r_d with the model's E(z))", OK if mod else MISSING,
                "built earlier (cache)" if mod else "cached build does not load", required=True)
        return mod is not None
    log = tree / "build.log"
    print("  ... building rd_Class (about 1 - 3 min; output in Class/rd_Class/build.log)", flush=True)
    t0 = time.time()
    mod, err = None, ""
    ctx = contextlib.nullcontext() if verbose else _output_to(log)
    try:
        with ctx:
            mod = RD.ensure_ready(build=True)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    if mod is not None:
        rep.add("class", "rd_Class (r_d with the model's E(z))", OK, f"built in {time.time() - t0:.0f} s")
        return True
    rep.add("class", "rd_Class", MISSING, "build failed" + (f" ({err})" if err else ""),
            fix=_build_failure_hint(log) + ("\nlast lines of the log:\n" + _tail(log) if log.is_file() else ""),
            required=True)
    return False


def check_rd(rep: Report) -> None:
    """r_d from rd_Class (tabulated LCDM E(z)) against CLASS's own LCDM r_d at the same point."""
    try:
        import User_defined_modules as UDM
        from Kosmulator_main import rd_class as RD
        rd_tab = RD.rd_model("LCDM_v", dict(RD_CHECK_POINT), UDM.Get_model_function("LCDM_v"))
        from classy import Class  # the LCDM_v build, loaded by build_class
        pars = RD.class_inputs(dict(RD_CHECK_POINT))
        c = Class()
        c.set(pars)
        c.compute()
        rd_class = float(c.rs_drag())
        c.struct_cleanup()
        c.empty()
    except Exception as e:
        rep.add("class", "r_d check", MISSING, f"{type(e).__name__}: {e}", required=True)
        return
    rel = abs(rd_tab / rd_class - 1.0)
    ok = rel < 1e-3 and 140.0 < rd_class < 155.0
    rep.add("class", "r_d check (Planck 2018 LCDM)", OK if ok else MISSING,
            f"rd_Class {rd_tab:.3f} Mpc, CLASS {rd_class:.3f} Mpc (relative difference {rel:.1e})",
            fix="rd_Class and CLASS disagree; please report this with Class/rd_Class/build.log", required=True)


def check_class(rep: Report, root: Path, setup: bool, tools_ok: bool, verbose: bool) -> None:
    rep.section("CLASS (built by Kosmulator per model from Class/<model>; no separate CLASS install needed)")
    models = _class_models(root)
    if not setup:
        from Kosmulator_main import rd_class as RD
        rd_dir = os.path.join(".", "Class", RD.RD_CLASS_NAME)
        stamp = Path(rd_dir, ".rd_class_stamp")
        rd_fresh = stamp.is_file() and stamp.read_text().strip() == RD._tree_stamp(
            os.path.join(".", "Class", RD.RD_CLASS_SOURCE))
        for m, built in (("LCDM_v", _cached("LCDM_v", os.path.join(".", "Class", "LCDM_v"))),
                         ("rd_Class", rd_fresh and _cached(RD.RD_CLASS_NAME, rd_dir))):
            rep.add("class", f"CLASS {m}", OK if built else INFO,
                    "built (cache)" if built else "not built yet: built on first use, or now with --setup")
    elif tools_ok:
        if build_class(rep, root, "LCDM_v", verbose) and build_rd_class(rep, root, verbose):
            check_rd(rep)
    else:
        rep.add("class", "CLASS builds", SKIP, "skipped: build tools missing (above)")
    others = [m for m in models if m != "LCDM_v"]
    if others:
        state = ", ".join(f"{m} ({'built' if _cached(m, os.path.join('.', 'Class', m)) else 'not built yet'})"
                          for m in others)
        rep.add("class_other", "Other models", INFO, f"{state}; each is built on its first CMB/r_d run, or "
                "now with kosmulator-doctor --build-class <model>")


def check_bbn(rep: Report, root: Path) -> None:
    rep.section("BBN")
    from Kosmulator_main import constants as K
    grid = root / "Observations" / K.BBN_GRID_RELATIVE
    have_lib = False
    try:
        from Kosmulator_main.Statistical_packages import _import_run_bbn
        with _output_to(Path(os.devnull)):
            _import_run_bbn()
        have_lib = True
    except Exception:
        pass
    if grid.is_file():
        try:
            import numpy as np
            with np.load(grid) as z:
                n = z["obh2"].size
                lo, hi = float(z["obh2"].min()), float(z["obh2"].max())
            rep.add("bbn", "AlterBBN D/H grid (BBN_DH_AlterBBN)", OK,
                    f"{grid.relative_to(root)}, {n} points, Omega_bh^2 {lo:.4f} - {hi:.4f}")
        except Exception as e:
            rep.add("bbn", "AlterBBN D/H grid", WARN, f"{grid} unreadable ({e})",
                    fix=f"git checkout -- {grid.relative_to(root)}")
    elif have_lib:
        rep.add("bbn", "AlterBBN D/H grid", INFO, "not present; AlterBBN is installed, so the first "
                "BBN_DH_AlterBBN run builds it")
    else:
        rep.add("bbn", "AlterBBN D/H grid", WARN,
                f"{grid.relative_to(root)} missing and AlterBBN not installed: BBN_DH_AlterBBN would use "
                "the D/H power-law approximation instead (AlterBBN's D/H is about 3% below it)",
                fix=f"git checkout -- {grid.relative_to(root)}")
    rep.add("bbn_live", "Live AlterBBN (optional)", OK if have_lib else INFO,
            "libkosmo_bbn.so found" if have_lib else
            "not installed; only needed for live AlterBBN calls or a new grid (README, AlterBBN section)")


def _timed(fn: Callable[[], object]):
    t0 = time.time()
    fn()
    return time.time() - t0


def check_cmb(rep: Report, root: Path, load: bool) -> None:
    rep.section("CMB likelihoods (optional: only for groups with CMB data)")
    from Kosmulator_main import Class_run as CR
    from Kosmulator_main import ground_cmb as G
    obs = root / "Observations"
    planck = [("CMB_hil", "plik_rd12_HM_v22b_TTTEEE.clik", "clik"),
              ("CMB_hil_TT", "plik_rd12_HM_v22_TT.clik", "clik"),
              ("CMB_hil lite", "plik_lite_v22_TTTEEE.clik", "clik"),
              ("CMB_lowl", "simall_100x143_offlike5_EE_Aplanck_B.clik", "clik"),
              ("CMB_lowl_TT", "commander_dx12_v3_2_29.clik", "clik"),
              ("CMB_lensing", "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing", "clik_lensing")]
    cmb_fix = 'pip install -e ".[cmb]"   (clipy, candl, act_dr6_lenslike)'
    # Planck: clik (compiled) or clipy (pure Python), as Class_run chooses
    if CR.CLIK_BACKEND is None:
        rep.add("planck", "Planck (clik or clipy)", MISSING, "neither is installed", fix=cmb_fix)
    else:
        ver = getattr(CR.clik, "__version__", "")
        rep.add("planck", "Planck likelihood code", OK, f"{CR.CLIK_BACKEND} {ver}".strip())
        lowl = CR.clipy_lowl_problem()
        if lowl:
            rep.add("planck", "CMB_lowl through clipy", WARN, lowl, fix='pip install "numpy>=2"')
        for tag, folder, kind in planck:
            path = obs / folder
            if not path.exists():
                rep.add("planck", f"{tag} data", MISSING, f"{path.relative_to(root)} missing",
                        fix=f"git checkout -- Observations/{folder}")
                continue
            if not load or (tag == "CMB_lowl" and lowl):
                continue
            try:
                ctor = getattr(CR.clik, kind)
                with _output_to(Path(os.devnull)):          # clik prints from C
                    dt = _timed(lambda: ctor(str(path)))
                rep.add("planck", f"{tag}", OK, f"{folder} loads ({dt:.1f} s)")
            except Exception as e:
                rep.add("planck", f"{tag}", MISSING, f"{folder} does not load: {type(e).__name__}: {e}")
        if not load:
            rep.add("planck", "Planck data", OK, "all six .clik folders present (loaded by --setup)")
    # ACT DR6 and SPT-3G D1 through candl
    ver = _version("candl", "candl-like")
    if not ver:
        rep.add("candl", "ACT DR6 / SPT-3G D1 (candl)", MISSING, "candl not installed", fix=cmb_fix)
    else:
        rep.add("candl", "candl", OK, ver)
        if load:
            for tag in ("CMB_ACT", "CMB_SPT"):
                try:
                    dt = _timed(lambda: G.candl_like(tag))
                    rep.add("candl", tag, OK, f"loads ({dt:.1f} s)")
                except Exception as e:
                    rep.add("candl", tag, MISSING, f"{type(e).__name__}: {e}")
    # ACT DR6 + Planck PR4 lensing
    ver = _version("act_dr6_lenslike", "act_dr6_lenslike")
    if not ver:
        rep.add("act_lens", "ACT DR6 lensing (act_dr6_lenslike)", MISSING, "not installed", fix=cmb_fix)
    else:
        try:
            d = G.act_lens_dir()
            rep.add("act_lens", "act_dr6_lenslike", OK, f"{ver}; data in {os.path.relpath(d, root)}")
            if load:
                for lens_only in (True, False):
                    what = "lensing only" if lens_only else "with primary CMB"
                    try:
                        dt = _timed(lambda: G.act_lens_data(lens_only))
                        rep.add("act_lens", f"CMB_lensing_ACTPlanck ({what})", OK, f"loads ({dt:.1f} s)")
                    except Exception as e:
                        rep.add("act_lens", f"CMB_lensing_ACTPlanck ({what})", MISSING,
                                f"{type(e).__name__}: {e}")
        except Exception as e:
            rep.add("act_lens", "ACT DR6 lensing data", MISSING, str(e))
    # SPT-3G MUSE lensing (Kosmulator's own reader)
    try:
        f = G.spt_muse_file()
        msg = os.path.relpath(f, root)
        if load:
            dt = _timed(G.spt_muse)
            msg += f" loads ({dt:.1f} s)"
        rep.add("spt_lens", "CMB_lensing_SPT (MUSE)", OK, msg)
    except Exception as e:
        rep.add("spt_lens", "CMB_lensing_SPT (MUSE)", MISSING, f"{type(e).__name__}: {e}".splitlines()[0],
                fix="git checkout -- Observations/spt3g_muse_lensing")


def check_mpi(rep: Report, run: bool) -> None:
    rep.section("Parallel runs")
    rep.add("pool", "Local process pool (--num_cores)", OK, f"{os.cpu_count()} cores")
    mpi_fix = _install_hint(conda="conda install -c conda-forge mpich mpi4py",
                            pip='pip install -e ".[mpi]"   (needs an MPI library, e.g. sudo apt install mpich)')
    v_mpi, v_sch = _version("mpi4py", "mpi4py"), _version("schwimmbad", "schwimmbad")
    mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
    if not (v_mpi and v_sch and mpiexec):
        miss = [n for n, v in (("mpi4py", v_mpi), ("schwimmbad", v_sch), ("mpiexec", mpiexec)) if not v]
        rep.add("mpi", "MPI (optional: --use_mpi, clusters)", INFO, "not available: " + ", ".join(miss),
                fix=mpi_fix)
        return
    if not run:
        rep.add("mpi", "MPI", OK, f"mpi4py {v_mpi}, schwimmbad {v_sch}, {mpiexec}")
        return
    code = "from mpi4py import MPI; c = MPI.COMM_WORLD; print(c.Get_rank(), c.Get_size())"
    try:
        r = subprocess.run([mpiexec, "-n", "2", sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=120)
        sizes = sorted(line.split()[1] for line in r.stdout.splitlines() if len(line.split()) == 2)
        if r.returncode == 0 and sizes == ["2", "2"]:
            rep.add("mpi", "MPI", OK, f"mpi4py {v_mpi}, schwimmbad {v_sch}; mpiexec -n 2 test passed")
        elif r.returncode == 0:
            rep.add("mpi", "MPI", WARN, "mpiexec started separate one-process jobs: mpi4py and mpiexec use "
                    "different MPI libraries", fix=mpi_fix)
        else:
            rep.add("mpi", "MPI", WARN, f"mpiexec -n 2 failed: {(r.stderr or r.stdout).strip()[-300:]}",
                    fix=mpi_fix)
    except Exception as e:
        rep.add("mpi", "MPI", WARN, f"mpiexec -n 2 test: {type(e).__name__}: {e}", fix=mpi_fix)


def check_latex(rep: Report, run: bool) -> None:
    rep.section("Plots")
    fix = _install_hint(apt="sudo apt install texlive-latex-extra texlive-fonts-recommended cm-super dvipng",
                        mac="brew install --cask mactex")
    missing = [t for t in ("latex", "dvipng") if not shutil.which(t)]
    if missing:
        rep.add("latex", "LaTeX labels (optional: --latex_enabled)", INFO, "not found: " + ", ".join(missing),
                fix=fix)
        return
    if not run:
        rep.add("latex", "LaTeX labels (--latex_enabled)", OK, "latex and dvipng found")
        return
    code = ("import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys;"
            "plt.rcParams['text.usetex']=True; f=plt.figure(); f.text(.5,.5,r'$\\Omega_m h^2$');"
            "f.savefig(sys.argv[1])")
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, "-c", code, os.path.join(d, "t.png")], capture_output=True,
                           text=True, timeout=180)
    if r.returncode == 0:
        rep.add("latex", "LaTeX labels (--latex_enabled)", OK, "a test plot with LaTeX labels renders")
    else:
        rep.add("latex", "LaTeX labels (--latex_enabled)", WARN,
                "latex found, but a test plot failed: " + (r.stderr.strip().splitlines() or ["?"])[-1], fix=fix)


QUICK_TEST_SCRIPT = '''\
import os, sys
sys.path.insert(0, os.getcwd())
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    import Kosmulator as KZ
    KZ.model_names = ["LCDM_v"]
    KZ.reference_model = "LCDM_v"
    KZ.observations = [["CC", "DESI_DR2", "PantheonP"]]
    KZ.nwalkers, KZ.nsteps, KZ.burn = 16, 300, 50
    sys.argv = ["Kosmulator.py", "--output_suffix", sys.argv[1], "--num_cores", sys.argv[2]]
    KZ.main()
'''


def quick_test(rep: Report, root: Path, keep: bool) -> None:
    rep.section("Test run (LCDM_v with CC + DESI_DR2 + PantheonP; 16 walkers, 300 steps)")
    suffix = QUICK_TEST_SUFFIX
    if any((root / d / suffix).exists() for d in QUICK_TEST_OUTPUTS):
        suffix = f"{QUICK_TEST_SUFFIX}_{os.getpid()}"
    tmp = Path(tempfile.mkdtemp(prefix="kosmulator_doctor_"))
    script, log = tmp / "kosmulator_test_run.py", tmp / "test_run.log"
    script.write_text(QUICK_TEST_SCRIPT)
    cores = str(max(1, min(4, os.cpu_count() or 1)))
    print(f"  ... running (under a minute on 2 cores; log {log})", flush=True)
    t0 = time.time()
    try:
        with open(log, "w") as f:
            r = subprocess.run([sys.executable, str(script), suffix, cores], cwd=str(root), stdout=f,
                               stderr=subprocess.STDOUT, timeout=1800)
        code = r.returncode
    except subprocess.TimeoutExpired:
        code = "timeout (30 min)"
    dt = time.time() - t0
    group = root / "MCMC_Chains" / suffix / "LCDM_v" / "CC_DESI_DR2_PantheonP"
    expected = {
        "chain": list(group.glob("*.h5")),
        "corner plot": list((root / "Plots" / "Saved_Plots" / suffix).rglob("corner_*.png")),
        "statistics": list((root / "Statistical_analysis_tables" / suffix).rglob("stats_summary.txt")),
    }
    absent = [k for k, v in expected.items() if not v]
    if code == 0 and not absent:
        rep.add("test", "Test run", OK, f"finished in {dt:.0f} s; chain, corner plot and statistics written"
                + (f" (kept: --output_suffix {suffix})" if keep else " (and removed)"))
        if not keep:
            for d in QUICK_TEST_OUTPUTS:
                shutil.rmtree(root / d / suffix, ignore_errors=True)
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        why = f"exit {code}" if code != 0 else "missing " + ", ".join(absent)
        rep.add("test", "Test run", MISSING, f"failed ({why}) after {dt:.0f} s",
                fix=f"log: {log}\nlast lines:\n{_tail(log)}", required=True)


def summary(rep: Report, setup: bool) -> None:
    rep.section("Summary")
    features = [
        ("core", "Core runs (SNe, BAO/DESI, CC, OHD, growth)", True),
        ("class", "CLASS (CMB groups; r_d in BAO/DESI + BBN/CMB groups)", True),
        ("test", "Test run", True),
        ("bbn", "BBN_DH_AlterBBN (grid)", False),
        ("planck", "Planck CMB", False),
        ("candl", "ACT DR6 / SPT-3G D1 (CMB_ACT, CMB_SPT, CMB_PACT, CMB_SPA)", False),
        ("act_lens", "ACT DR6 + Planck PR4 lensing", False),
        ("spt_lens", "SPT-3G MUSE lensing", False),
        ("mpi", "MPI runs", False),
        ("bbn_live", "Live AlterBBN", False),
        ("latex", "LaTeX labels", False),
    ]
    optional_installs = ("mpi", "bbn_live", "latex")
    for key, label, _ in features:
        items = [r for r in rep.items if r.feature == key]
        if not items:
            continue
        st = {r.status for r in items}
        if MISSING in st:
            word = "NOT READY" if any(r.required for r in items if r.status == MISSING) else \
                "not available (optional; command above)"
        elif WARN in st:
            word = "ready, with a warning (above)"
        elif SKIP in st:
            word = "skipped"
        elif INFO in st and key in optional_installs:
            word = "not installed (optional)"
        elif INFO in st and key == "class":
            word = "build tools ready; CLASS is built on first use (or now with --setup)"
        elif INFO in st and key == "bbn":
            word = "grid built on first use"
        else:
            word = "ready"
        print(f"  {label}: {word}")
    print()
    if rep.required_ok():
        print("Overall: READY" + ("" if setup else "   (kosmulator-doctor --setup builds CLASS and makes a "
                                                     "test run)"))
    else:
        print("Overall: NOT READY: fix the MISSING lines above (each has the command after '>') and run "
              "kosmulator-doctor again")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="kosmulator-doctor",
        description="Check a Kosmulator installation; --setup also builds CLASS and makes a test run.")
    p.add_argument("--setup", action="store_true",
                   help="Build CLASS for LCDM_v and rd_Class, load every likelihood code and its data, test "
                        "MPI and LaTeX, and make a short test run (a few minutes; run once after installing).")
    p.add_argument("--build-class", metavar="MODEL", action="append", default=[],
                   help="Build Class/MODEL now (repeatable), e.g. a model with its own CLASS.")
    p.add_argument("--no-test-run", action="store_true", help="With --setup: skip the test run.")
    p.add_argument("--keep-test", action="store_true", help="With --setup: keep the test run's outputs.")
    p.add_argument("--verbose-build", action="store_true",
                   help="Show the compiler output instead of writing it to Class/<model>/build.log.")
    # Flags of the earlier kosmulator-doctor, kept so that old commands still run
    for flag in ("--no-class", "--no-clik", "--no-clik-env", "--no-planck-assets", "--alterbbn"):
        p.add_argument(flag, action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--test-load", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)

    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    from Kosmulator_main import __version__
    root = find_root()
    print(f"Kosmulator {__version__}: installation check" + (" and setup" if a.setup else ""))
    print(f"Python {sys.version.split()[0]} ({sys.executable}); {platform.system()} {platform.machine()}")
    if root is None:
        print("\nThe Kosmulator folder (with Kosmulator.py, Class/ and Observations/) was not found. Run "
              "kosmulator-doctor from inside it, and install Kosmulator with pip install -e . (editable).")
        return 1
    print(f"Kosmulator folder: {root}")
    os.chdir(root)                       # Kosmulator's paths (./Class, ./Observations) are relative
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rep = Report()
    check_core(rep)
    if not rep.required_ok():
        summary(rep, a.setup)
        return 1

    if a.build_class:
        tools_ok = check_build_tools(rep)
        rep.section("CLASS builds")
        for m in a.build_class:
            if tools_ok:
                build_class(rep, root, m, a.verbose_build)
        print()
        return 0 if rep.required_ok() else 1

    tools_ok = True
    if not a.no_class:
        tools_ok = check_build_tools(rep)
        check_class(rep, root, a.setup, tools_ok, a.verbose_build)
    check_bbn(rep, root)
    if not (a.no_clik or a.no_planck_assets):
        check_cmb(rep, root, load=a.setup or a.test_load)
    check_mpi(rep, run=a.setup)
    check_latex(rep, run=a.setup)
    if a.setup and not a.no_test_run:
        if rep.required_ok():
            quick_test(rep, root, a.keep_test)
        else:
            rep.section("Test run")
            rep.add("test", "Test run", SKIP, "skipped: fix the MISSING lines above first")
    summary(rep, a.setup)
    return 0 if rep.required_ok() else 1


if __name__ == "__main__":
    sys.exit(main())
