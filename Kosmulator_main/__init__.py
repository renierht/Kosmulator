from __future__ import annotations

import argparse
import importlib
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


__version__ = "2.0.0"


@dataclass
class CheckItem:
    name: str
    ok: bool
    detail: str = ""
    hint: str = ""


def _try_import(modname: str) -> Tuple[bool, str]:
    try:
        m = importlib.import_module(modname)
        return True, getattr(m, "__file__", "") or "(built-in)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _path_ok(p: Optional[str]) -> Tuple[bool, str]:
    if not p:
        return False, "not set"
    pp = Path(p).expanduser().resolve()
    return pp.exists(), str(pp)


def _repo_root_guess() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def find_observations_dir() -> Optional[Path]:
    env = os.environ.get("KOSMULATOR_OBSERVATIONS")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists() and p.is_dir():
            return p

    root = _repo_root_guess()
    p = (root / "Observations").resolve()
    if p.exists() and p.is_dir():
        return p

    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        q = parent / "Observations"
        if q.exists() and q.is_dir():
            return q.resolve()

    return None


def check_planck_clik_assets(
    *,
    test_load: bool = False,
    hi_l_name: str = "plik_rd12_HM_v22b_TTTEEE.clik",
    low_l_name: str = "simall_100x143_offlike5_EE_Aplanck_B.clik",
) -> List[CheckItem]:
    items: List[CheckItem] = []

    obs = find_observations_dir()
    if obs is None:
        items.append(
            CheckItem(
                name="Locate Observations/ directory",
                ok=False,
                detail="Not found",
                hint=(
                    "Run Kosmulator from the cloned repo, or set:\n"
                    "  export KOSMULATOR_OBSERVATIONS=/path/to/Observations"
                ),
            )
        )
        return items

    items.append(CheckItem(name="Locate Observations/ directory", ok=True, detail=str(obs)))

    hi_l = obs / hi_l_name
    low_l = obs / low_l_name

    items.append(
        CheckItem(
            name="Planck high-l .clik directory exists",
            ok=hi_l.exists(),
            detail=str(hi_l),
            hint="If missing: ensure Observations/ contains the Planck *.clik directories.",
        )
    )
    items.append(
        CheckItem(
            name="Planck low-l .clik directory exists",
            ok=low_l.exists(),
            detail=str(low_l),
            hint="If missing: ensure Observations/ contains the Planck *.clik directories.",
        )
    )

    if not test_load:
        return items

    ok_clik, d_clik = _try_import("clik")
    if not ok_clik:
        items.append(
            CheckItem(
                name="clik available for likelihood load test",
                ok=False,
                detail=d_clik,
                hint="Build/install CLIK and ensure conda activation sources clik_profile.sh.",
            )
        )
        return items

    try:
        import clik as _clik  # type: ignore
    except Exception as e:
        items.append(
            CheckItem(
                name="Import clik for likelihood load test",
                ok=False,
                detail=f"{type(e).__name__}: {e}",
                hint="Re-check CLIKROOT/PYTHONPATH/LD_LIBRARY_PATH from clik_profile.sh.",
            )
        )
        return items

    if hi_l.exists():
        try:
            _clik.clik(str(hi_l))
            items.append(CheckItem(name="clik can load high-l likelihood", ok=True, detail=str(hi_l)))
        except Exception as e:
            items.append(
                CheckItem(
                    name="clik can load high-l likelihood",
                    ok=False,
                    detail=f"{type(e).__name__}: {e}",
                    hint="Check missing shared-library deps (ldd on clik .so) and env vars from clik_profile.sh.",
                )
            )

    if low_l.exists():
        try:
            _clik.clik(str(low_l))
            items.append(CheckItem(name="clik can load low-l likelihood", ok=True, detail=str(low_l)))
        except Exception as e:
            items.append(
                CheckItem(
                    name="clik can load low-l likelihood",
                    ok=False,
                    detail=f"{type(e).__name__}: {e}",
                    hint="Check missing shared-library deps (ldd on clik .so) and env vars from clik_profile.sh.",
                )
            )

    return items


def check_installation(
    *,
    check_class: bool = True,
    check_clik: bool = True,
    check_clik_env: bool = True,
    check_planck_assets: bool = True,
    planck_test_load: bool = False,
    check_alterbbn: bool = False,
) -> Dict[str, Any]:
    items: List[CheckItem] = []

    items.append(
        CheckItem(
            name="python>=3.9",
            ok=sys.version_info >= (3, 9),
            detail=f"{sys.version.split()[0]} ({sys.executable})",
            hint="Recommended: Python 3.11 (best compatibility in our tested setup).",
        )
    )

    core_mods = [
        "numpy",
        "scipy",
        "matplotlib",
        "h5py",
        "pandas",
        "astropy",
        "emcee",
        "zeus",
        "getdist",
    ]
    for mod in core_mods:
        ok, detail = _try_import(mod)
        items.append(CheckItem(name=f"import {mod}", ok=ok, detail=detail,
                              hint=f"Install missing dependency: {mod} (see README)."))

    # CLASS
    if check_class:
        ok, detail = _try_import("classy")
        items.append(
            CheckItem(
                name="import classy (CLASS python wrapper)",
                ok=ok,
                detail=detail,
                hint="Build CLASS + install wrapper: `cd CLASS && make -j && python -m pip install .`",
            )
        )

    # CLIK
    clik_import_ok = True
    if check_clik:
        ok, detail = _try_import("clik")
        clik_import_ok = ok
        items.append(
            CheckItem(
                name="import clik (Planck/CLIK)",
                ok=ok,
                detail=detail,
                hint="Build/install CLIK and ensure activation sources `$CLIKROOT/bin/clik_profile.sh`.",
            )
        )
        if ok:
            ok_lkl, d_lkl = _try_import("clik.lkl")
            ok_len, d_len = _try_import("clik.lkl_lensing")
            items.append(CheckItem(name="import clik.lkl", ok=ok_lkl, detail=d_lkl,
                                   hint="If this fails: lkl module may need the activation symlink fix."))
            items.append(CheckItem(name="import clik.lkl_lensing", ok=ok_len, detail=d_len,
                                   hint="If this fails: lkl_lensing module may need the activation symlink fix."))

    # CLIKROOT check: only “required” if clik import failed
    if check_clik_env:
        ok_root, d_root = _path_ok(os.environ.get("CLIKROOT"))
        if check_clik and not clik_import_ok:
            items.append(
                CheckItem(
                    name="CLIKROOT is set and exists (required when clik is missing)",
                    ok=ok_root,
                    detail=d_root,
                    hint=(
                        "Set CLIKROOT and source clik_profile.sh on conda activate:\n"
                        "  export CLIKROOT=/path/to/plc-3.1\n"
                        "  source \"$CLIKROOT/bin/clik_profile.sh\""
                    ),
                )
            )
        else:
            # clik works, so CLIKROOT is optional / informational
            items.append(
                CheckItem(
                    name="CLIKROOT is set and exists (optional)",
                    ok=ok_root,
                    detail=d_root,
                    hint="Optional: set CLIKROOT if you want a standard, reproducible CLIK activation path.",
                )
            )

    # Planck assets in Observations/
    if check_planck_assets:
        items.extend(check_planck_clik_assets(test_load=planck_test_load))

    # Optional AlterBBN
    if check_alterbbn:
        ok_py, d_py = _try_import("alterbbn_ctypes")
        items.append(CheckItem(
            name="import alterbbn_ctypes (AlterBBN wrapper)",
            ok=ok_py,
            detail=d_py,
            hint="Optional: only needed for BBN_DH_AlterBBN. Ensure alterbbn_ctypes.py is available.",
        ))
        lib = os.environ.get("KOSMO_BBN_LIB")
        ok_lib, d_lib = _path_ok(lib)
        items.append(CheckItem(
            name="KOSMO_BBN_LIB is set and exists",
            ok=ok_lib,
            detail=d_lib,
            hint="Optional: export KOSMO_BBN_LIB=/path/to/libkosmo_bbn.so for BBN_DH_AlterBBN.",
        ))

    # Overall OK: core python deps + (optionally) CLASS/CLIK imports
    required_names = {"python>=3.9", *[f"import {m}" for m in core_mods]}
    if check_class:
        required_names.add("import classy (CLASS python wrapper)")
    if check_clik:
        required_names.update({"import clik (Planck/CLIK)", "import clik.lkl", "import clik.lkl_lensing"})

    overall_ok = all(it.ok for it in items if it.name in required_names)

    return {
        "ok": overall_ok,
        "version": __version__,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "items": [it.__dict__ for it in items],
    }


def print_installation_report(report: Dict[str, Any]) -> None:
    print(f"Kosmulator {report.get('version')} — environment report")
    print(f"Python: {report.get('python')}  @  {report.get('executable')}")
    print("-" * 72)

    for it in report.get("items", []):
        status = "OK" if it["ok"] else "MISSING"
        print(f"[{status:7}] {it['name']}")
        if it.get("detail"):
            print(f"         {it['detail']}")
        if not it["ok"] and it.get("hint"):
            hint = textwrap.indent(it["hint"].strip(), "         ")
            print(hint)
        print()

    print("-" * 72)
    print("Overall:", "OK" if report.get("ok") else "NOT READY")


def main_doctor() -> None:
    parser = argparse.ArgumentParser(prog="kosmulator-doctor", add_help=True)
    parser.add_argument("--no-class", action="store_true", help="Skip CLASS/classy checks.")
    parser.add_argument("--no-clik", action="store_true", help="Skip CLIK/clik checks.")
    parser.add_argument("--no-clik-env", action="store_true", help="Skip checking CLIKROOT env var.")
    parser.add_argument("--no-planck-assets", action="store_true", help="Skip checking Observations/*.clik directories.")
    parser.add_argument("--test-load", action="store_true", help="Attempt to load Planck .clik likelihoods via clik.clik(path).")
    parser.add_argument("--alterbbn", action="store_true", help="Also check optional AlterBBN wrapper and KOSMO_BBN_LIB.")
    args = parser.parse_args()

    rep = check_installation(
        check_class=not args.no_class,
        check_clik=not args.no_clik,
        check_clik_env=not args.no_clik_env,
        check_planck_assets=not args.no_planck_assets,
        planck_test_load=args.test_load,
        check_alterbbn=args.alterbbn,
    )
    print_installation_report(rep)

