import os
import sys
import argparse
import logging
import platform
import textwrap
import time as _time
import re
import ast
import shutil
import tempfile
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
from collections import defaultdict, Counter
from contextlib import contextmanager
from Kosmulator_main import constants as K

import numpy as np
from scipy import linalg as la
try:
    # SciPy >= 1.10+ prefers the new name
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    # Older SciPy
    from scipy.integrate import cumtrapz

from Kosmulator_main.constants import (
    DEFAULT_PLOT_COLORS,
    DEFAULT_PLOTS_BASE,
    GAMMA_FS8_SINGLETON,
    DEFAULT_CMB_FILES,
    CMB_ELL_MAX_PLOT,
    CMB_LENSING_LMIN,
    CMB_LENSING_LMAX,
    PLANCK_NUISANCE_DEFAULTS,
    PLANCK_TT_ONLY_NUISANCE,
    PLANCK_TTTEEE_NUISANCE,
    C_KM_S,
)

# Optional third-party (import safely)
try:
    import emcee
except Exception:
    emcee = None  # type: ignore

try:
    import h5py
except Exception:
    h5py = None  # type: ignore

# Zeus is optional; callbacks imported inside functions too
try:
    import zeus  # type: ignore
except Exception:
    zeus = None  # type: ignore

# Optional: MPI + schwimmbad
try:
    from mpi4py import MPI  # type: ignore
except Exception:
    MPI = None  # type: ignore

try:
    from schwimmbad import MPIPool  # type: ignore
except Exception:
    MPIPool = None  # type: ignore


log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# Patterns we consider "noisy" and collapse during init
INIT_COLLAPSE_KEYS = (
    "Added ",
    "BAO/DESI singleton",
    "BAO/DESI combo",
    "BAO/DESI uncalibrated",
    "H_0 is not constrained by",
    "r_d calibrated by early-time dataset(s)",
    "fσ8 singleton",
    "fσ₈ singleton",
    "fσ8 run solo",
    "fσ₈ run solo",
    "fsigma8 run solo",
    "f solo",
    "Observation f alone is not ideal",
    "Auto-calibrating r_d",
    "[Config] Normalised group ",
    "Bumping nwalker from",
    "JLA run solo",
    "Pantheon run solo",
    "Pantheon+ (uncal) run solo",
    "Union3 run solo",
    "DESY5 run solo",
)

# Stable observation ordering for canonicalisation
_OBS_ORDER = [
    "BAO",
    "BBN_DH_AlterBBN",
    "BBN_DH",
    "BBN_PryMordial",
    "CC",
    "CMB_hil",
    "CMB_hil_TT",
    "CMB_lensing",
    "CMB_lowl",
    "DESI_DR1",
    "DESI_DR2",
    "f",
    "f_sigma_8",
    "JLA",
    "JLA_legacy",
    "OHD",
    "Pantheon",
    "PantheonP",
]
_OBS_RANK = {name: i for i, name in enumerate(_OBS_ORDER)}


# ───────────────────────────────────────────────────────────────────────────────
# 1) CLI
# ───────────────────────────────────────────────────────────────────────────────

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run Kosmulator MCMC simulation.")

    parser.add_argument(
        "--num_cores", type=int, default=8,
        help="Number of cores (default 8)",
    )
    parser.add_argument(
        "--use_mpi", action="store_true",
        help="Force use of MPI pool",
    )
    parser.add_argument(
        "--latex_enabled", action="store_true",
        help="Enable LaTeX rendering in plots",
    )
    parser.add_argument(
        "--plot_table", action="store_true",
        help="Generate parameter-table plots",
    )
    parser.add_argument(
        "--output_suffix", type=str, default="Test_run",
        help="Suffix for output directories and files",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume incomplete chains instead of loading only",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete any existing .h5 chains and run MCMC from scratch",
    )
    parser.add_argument(
        "--force_vectorisation", action="store_true",
        help="Treat all models as vectorised",
    )
    parser.add_argument(
        "--disable_vectorisation",
        action="store_true",
        help="Disable vectorised likelihood evaluation even if available (forces scalar evaluation).",
    )
    parser.add_argument(
        "--force_zeus", action="store_true",
        help="Force the Zeus sampler",
    )
    parser.add_argument(
        "--tau-consecutive", "--consecutive-required",
        dest="consecutive_required", type=int, default=2,
        help=(
            "Convergence rule (zeus and emcee): number of consecutive checks at which "
            "all its conditions must hold before the run stops. Default 2."
        ),
    )
    parser.add_argument(
        "--autocorr-buffer", type=int, default=None,
        help=(
            "Optional minimum run length: the convergence rule may end a run only after "
            "burn + this many steps. Default 0 (the rule's own tests decide)."
        ),
    )
    parser.add_argument(
        "--autocorr-check-every", type=int, default=100,
        help="Check the convergence rule (and redraw its plot) every N steps (default 100)",
    )
    parser.add_argument(
        "--init-log", choices=["terse", "normal", "verbose"], default="terse",
        help="Initialisation logging style (default: terse).",
    )
    parser.add_argument(
        "--corner-show-all-cmb-params",
        action="store_true",
        help=(
            "Corner plot: show ALL CMB parameters (including nuisance). "
            "Default: show only key cosmological."
        ),
    )
    parser.add_argument(
        "--corner-table-full", action="store_true",
        help=(
            "Corner plot top table: keep FULL parameter list (including CMB "
            "nuisances). Default off unless plot_table is False."
        ),
    )
    parser.add_argument(
        "--force_emcee",
        action="store_true",
        help="Force the emcee sampler (ignore Zeus even if available).",
    )
    parser.add_argument(
        "--engine-mode",
        choices=["mixed", "single", "fastest"],
        default="mixed",
        help=(
            "Sampler strategy: "
            "'mixed' (per-observation engine + cross-engine reuse), "
            "'single' (one engine for all observations), "
            "'fastest' (auto-choose per observation set)."
        ),
    )
    parser.add_argument(
        "--print_loglike",
        nargs="?",
        const=1,       # user passed flag without value => print every call
        default=None,  # flag absent => printing disabled
        type=int,
        help="Print likelihood diagnostics (components + TOTAL) for one walker. "
             "Optional N prints every Nth likelihood call (default if flag is present: 1, Higher N = less printouts).",
    )

    args = parser.parse_args()

    pll = getattr(args, "print_loglike", None)

    K.print_loglike = pll is not None
    K.print_loglike_every = max(1, int(pll)) if pll is not None else 1

    return args


# ───────────────────────────────────────────────────────────────────────────────
# 2) Pretty banners & small UX helpers
# ───────────────────────────────────────────────────────────────────────────────

def print_completion_banner(elapsed: str) -> None:
    print(f"\n\n\033[33m{'#'*75}\033[0m")
    print(f"\033[33m#### \033[0m")
    print(
        f"\033[33m#### All models processed successfully in a total time of "
        f"{elapsed}!!!\033[0m"
    )
    print(f"\033[33m#### \033[0m")
    print(f"\033[33m#### Thank you for using Kosmulator :D\033[0m")
    print(f"\033[33m#### \033[0m")
    print(f"\033[33m{'#'*75}\033[0m\n")


def print_init_banner(message: str) -> None:
    bar = "#" * 48
    print(f"\n\033[33m{bar}\033[0m", flush=True)
    print(f"\033[33m####\033[0m \033[1m{message}\033[0m", flush=True)
    print(f"\033[33m{bar}\033[0m", flush=True)

def _engine_mode_label_for_banner(
    model_name: str,
    engine_mode: str,
    can_vec: bool,
    touches_cmb_bbn: bool,
    *,
    force_emcee: bool = False,
    force_zeus: bool = False,
    single_engine_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Human-facing banner label.

    We do NOT promise a specific sampler here in 'mixed'/'fastest' because
    the engine can change per observation set.
    """
    # CLI overrides dominate
    if force_emcee:
        return "forced-emcee"
    if force_zeus:
        return "forced-zeus"

    engine_mode = (engine_mode or "mixed").lower()

    if engine_mode == "single":
        eng = (single_engine_map or {}).get(model_name, "auto")
        return f"single-{eng}"

    if engine_mode == "fastest":
        return "fastest"

    # mixed (or unknown → treat like mixed)
    # Optional hint: if this model ever touches CMB/BBN, it will almost certainly
    # involve emcee somewhere.
    if touches_cmb_bbn and can_vec:
        return "mixed (zeus+emcee)"
    if touches_cmb_bbn:
        return "mixed (emcee)"
    if can_vec:
        return "mixed (zeus)"
    return "mixed"


def print_model_banner(
    model_name: str,
    engine_mode: str,
    can_vec: bool,
    touches_cmb_bbn: bool,
    *,
    single_engine_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Big banner for each model. Reports the *actual* execution policy.
    """
    import Kosmulator_main.constants as K

    # Source of truth
    eng = None
    if single_engine_map and model_name in single_engine_map:
        eng = single_engine_map[model_name]
    else:
        eng = "unknown"

    eng = str(eng).lower()

    parts = []

    if eng == "zeus":
        parts.append("Zeus")
        if not can_vec:
            parts.append("scalar")
        if not can_vec:
            parts.append("pool")
    elif eng == "emcee":
        parts.append("EMCEE")
        parts.append("vectorised" if (can_vec and not touches_cmb_bbn) else "pool")
    else:
        parts.append(eng)

    # Engine mode context
    mode_label = engine_mode if engine_mode in ("mixed", "single", "fastest") else "auto"

    # Explicit overrides (read from K, never passed)
    if getattr(K, "force_zeus", False):
        parts.append("forced")
    elif getattr(K, "force_emcee", False):
        parts.append("forced")

    label = f"{' '.join(parts)} ({mode_label})"

    msg = f"Processing model: {model_name}  |  Engine: {label}"
    width = max(48, len(msg) + 6)
    bar = "#" * width

    print(f"\n\033[33m{bar}\033[0m", flush=True)
    print(f"\033[33m####\033[0m \033[1m{msg}\033[0m", flush=True)
    print(f"\033[33m{bar}\033[0m", flush=True)




def format_elapsed_time(seconds: float) -> str:
    """
    Format elapsed seconds as D:HH:MM:SS / H:MM:SS / M:SS / SS.
    """
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}:{h:02}:{m:02}:{s:02} days"
    if h:
        return f"{h}:{m:02}:{s:02} hours"
    if m:
        return f"{m}:{s:02} minutes"
    return f"{s} seconds"


def get_parallel_flag(use_mpi: bool, num_cores: int) -> bool:
    """
    Whether to attempt parallelism on this OS/config.
    """
    return (use_mpi or (num_cores and num_cores > 1)) and platform.system() != "Windows"


# ───────────────────────────────────────────────────────────────────────────────
# 3) Pool & MPI (robust + friendly fallbacks)
# ───────────────────────────────────────────────────────────────────────────────

_RUN_LOCK_HANDLE = None   # kept open for the lifetime of the run


def acquire_run_lock(suffix: str = ""):
    """
    Take an exclusive OS-level lock on MCMC_Chains[/<suffix>]/.kosmulator.lock
    for the lifetime of this process, so that two runs cannot write to the same
    chain files at once (that interleaves rows in the HDF5 files). The operating
    system releases the lock when the process ends, even after a crash, so there
    are no stale locks. Set KOSM_IGNORE_LOCK=1 to skip the check.

    Raises RuntimeError if another live run holds the lock. On file systems
    without lock support a warning is logged and the run continues.
    """
    global _RUN_LOCK_HANDLE
    log = logging.getLogger(__name__)
    if os.environ.get("KOSM_IGNORE_LOCK", "0") == "1":
        return None

    d = os.path.join("MCMC_Chains", suffix) if suffix else "MCMC_Chains"
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, ".kosmulator.lock")
    fh = open(path, "a+")

    locked_by_other = False
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, PermissionError):
        locked_by_other = True
    except OSError as e:
        if os.name == "nt" and getattr(e, "errno", None) in (13, 36):
            locked_by_other = True
        else:
            log.warning(
                "Could not lock %s (%s); continuing without protection against "
                "concurrent runs.", path, e,
            )
            fh.close()
            return None

    if locked_by_other:
        try:
            fh.seek(0)
            info = fh.read().strip()
        except Exception:
            info = ""
        fh.close()
        raise RuntimeError(
            f"Another Kosmulator run is writing to '{d}'"
            f"{' (' + info + ')' if info else ''}. Use a different --output_suffix, "
            "wait for that run to finish, or set KOSM_IGNORE_LOCK=1."
        )

    try:
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid {os.getpid()} on {platform.node()} since "
                 f"{_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.flush()
    except Exception:
        pass
    _RUN_LOCK_HANDLE = fh
    return fh


def get_pool(use_mpi: bool = False, num_cores: int | None = None, **pool_kwargs):
    """
    Create a parallel pool.

    - If running under MPI (explicit --use_mpi or env detection), return MPIPool and
      IGNORE num_cores and any multiprocessing-only kwargs (initializer, initargs, etc.).
    - Else, if num_cores>=2, create a local multiprocessing Pool (using 'spawn') and
      forward any **pool_kwargs to it (initializer, initargs, maxtasksperchild...).
    - Else return None (serial).
    """
    import multiprocessing as mp

    logger = logging.getLogger(__name__)

    # Best-effort rank info (safe even if not under MPI)
    try:
        from mpi4py import MPI as _MPI
        rank = _MPI.COMM_WORLD.Get_rank()
        world = _MPI.COMM_WORLD.Get_size()
    except Exception:
        rank, world = 0, 1

    # Detect MPI via flag or common env vars from mpiexec/slurm
    mpi_env = any(
        k in os.environ
        for k in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMI_RANK", "SLURM_NTASKS")
    )
    if use_mpi or mpi_env:
        try:
            from schwimmbad import MPIPool
        except Exception:
            if rank == 0:
                print(
                    "MPI requested/detected but schwimmbad is not installed "
                    "→ falling back to serial."
                )
            return None

        pool = MPIPool()
        if not pool.is_master():
            # Workers block here until work arrives; once finished, they exit
            pool.wait()
            sys.exit(0)

        if rank == 0:
            print(
                f"Using MPI Pool with schwimmbad (world={world}). "
                "Ignoring num_cores and multiprocessing kwargs."
            )
        return pool

    # ---- Local multiprocessing path (no MPI) ----
    if num_cores is None or num_cores < 2:
        if rank == 0:
            print("Running in series on 1 core.")
        return None

    # Prefer 'spawn' to avoid fork-related issues (esp. if user later mixes MPI)
    try:
        ctx = mp.get_context("spawn")
    except ValueError:
        ctx = mp.get_context()  # fallback

    #if rank == 0:
     #   forwarded = ", ".join(pool_kwargs.keys()) or "none"
     #   print()
      #  print(
       #     "Using local multiprocessing Pool (spawn) with "
        #    f"{num_cores} cores; forwarding kwargs: {forwarded}. "
         #   "Note: vectorised Zeus chains ignore this Pool and run single-core; "
          #  "emcee and non-vectorised Zeus use the worker processes."
        #)

    # One BLAS/OpenMP thread per worker. Spawned workers read these variables
    # when NumPy loads, which happens while unpickling the initializer, so
    # setting them inside the initializer is too late. Set them in the parent
    # just for the pool start-up, then restore the parent's values.
    _thread_vars = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
    _saved = {k: os.environ.get(k) for k in _thread_vars}
    try:
        for k in _thread_vars:
            os.environ[k] = "1"
        # Forward initializer/initargs/maxtasksperchild/etc.
        return ctx.Pool(processes=num_cores, **pool_kwargs)
    finally:
        for k, v in _saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def init_mpi():
    """Return (comm, rank) if MPI available, else (None, 0)."""
    try:
        comm = MPI.COMM_WORLD if MPI else None
        rank = comm.Get_rank() if comm else 0
    except Exception:
        comm, rank = None, 0
    return comm, rank


def mpi_broadcast(comm, rank, payload):
    """Broadcast a payload dict from rank 0 to all ranks."""
    if comm is None:
        return payload if rank == 0 else None
    return comm.bcast(payload if rank == 0 else None, root=0)


def _mpi_comm():
    try:
        from mpi4py import MPI as _MPI
        return _MPI.COMM_WORLD
    except Exception:
        return None


def mpi_rank() -> int:
    comm = _mpi_comm()
    return comm.Get_rank() if comm else 0


def is_rank0() -> bool:
    return mpi_rank() == 0


class Rank0OnlyFilter(logging.Filter):
    def filter(self, record):
        return is_rank0()


def install_rank0_logging():
    """Drop logs from non-master MPI ranks."""
    root = logging.getLogger()
    # Avoid stacking multiple filters if called twice
    if not any(isinstance(f, Rank0OnlyFilter) for f in root.filters):
        root.addFilter(Rank0OnlyFilter())
        
def is_main_process() -> bool:
    try:
        return mp.current_process().name == "MainProcess"
    except Exception:
        return True


# ───────────────────────────────────────────────────────────────────────────────
# 4) Plot settings / paths (non-noisy LaTeX detection)
# ───────────────────────────────────────────────────────────────────────────────

def group_colours(n: int, base=None) -> List[str]:
    """
    n distinct colours: `base` (default DEFAULT_PLOT_COLORS) without entries that
    are the same colour under another name, then tab20 colours and golden-angle
    hues, each kept only if it is clearly different (RGB distance > 0.25) from
    every colour already chosen.
    """
    import colorsys
    import matplotlib.colors as mcolors
    base = list(base if base is not None else DEFAULT_PLOT_COLORS)
    out, rgbs = [], []

    def add(c):
        try:
            rgb = np.array(mcolors.to_rgb(c))
        except ValueError:
            return
        if all(np.linalg.norm(rgb - r) > 0.25 for r in rgbs):
            out.append(c if isinstance(c, str) else mcolors.to_hex(c))
            rgbs.append(rgb)

    for c in base:
        if len(out) >= n:
            return out[:n]
        add(c)
    import matplotlib
    for c in matplotlib.colormaps["tab20"].colors:
        if len(out) >= n:
            return out[:n]
        add(mcolors.to_hex(c))
    k = 0
    while len(out) < n and k < 10000:
        h = (0.618033988749895 * k) % 1.0
        s, v = (0.85, 0.75) if k % 2 == 0 else (0.55, 0.95)
        add(mcolors.to_hex(colorsys.hsv_to_rgb(h, s, v)))
        k += 1
    while len(out) < n:                       # more groups than distinguishable colours
        out.append(out[len(out) % max(1, len(rgbs))])
    return out[:n]


def build_plot_settings(
    observations,
    suffix: str,
    latex_enabled: bool,
    plot_table: bool,
) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # noqa: F401

    colors = DEFAULT_PLOT_COLORS
    base_plots = DEFAULT_PLOTS_BASE
    plot_table = bool(plot_table)

    n_obs = len(observations)

    # One distinct colour per observation group: the default list first, then
    # further colours that differ from every colour already used (no repeats).
    color_schemes = group_colours(n_obs, colors)

    settings: Dict[str, Any] = {
        # Style
        "color_schemes": color_schemes,
        "line_styles": ["-", "--", ":", "-."],
        "marker_size": 4,
        "legend_font_size": 12,
        "title_font_size": 12,
        "label_font_size": 12,
        "tick_font_size": 5,
        # LaTeX & DPI
        "latex_enabled": latex_enabled,
        "plot_table": plot_table,
        "dpi": 200,
        # Table layout knobs (corner top band)
        "corner_top": 0.88,
        "table_vpad": 0.008,
        "table_height_base": 0.012,
        "table_height_per_row": 0.33,
        "table_font_min": 9,
        "table_font_max": 14,
        "cell_height_factor": 4.5,
        "table_max_band_fraction": 0.29,
        "table_headroom_fraction": 0.10,
        # Best-fit plot layout knobs
        "bestfit_hspace": 0.00,
        "bestfit_wspace": 0.15,
        "bestfit_figwidth_percol": 6.1,
        "bestfit_left_base": 0.15,
        "bestfit_left_step": 0.05,
        "bestfit_left_min": 0.03,
        "bestfit_right": 0.97,
        "bestfit_top": 0.985,
        "bestfit_bottom": 0.08,
        # In-panel r_d badge
        "rd_badge_pos": "upper right",
        # X-axis limits and theory grid
        "xpad_frac": 0.15,
        "model_xpad_frac": 0.12,
        "z_dense_points": 800,
        # Optional: whether to overlay per-point model on BAO/DESI
        "overlay_model_for_bao_desi": False,
        # Save locations (model-specific subfolders added at use sites)
        "autocorr_save_path": (
            os.path.join(base_plots, suffix) if suffix else base_plots
        ),
        "cornerplot_save_path": (
            os.path.join(base_plots, suffix, "corner")
            if suffix
            else os.path.join(base_plots, "corner")
        ),
        "bestfit_save_path": (
            os.path.join(base_plots, suffix, "best_fits")
            if suffix
            else os.path.join(base_plots, "best_fits")
        ),
        "output_suffix": suffix,
        # Bandpower files (tweak if your local paths differ)
        **DEFAULT_CMB_FILES,
        "cmb_ell_max_plot": CMB_ELL_MAX_PLOT,
        "cmb_lensing_Lmin": CMB_LENSING_LMIN,
        "cmb_lensing_Lmax": CMB_LENSING_LMAX,
        # Autocorr
        "autocorr_check_every": 100,
        "autocorr_buffer_after_burn": 0,
        # Misc plot options
        "legend_loc": "upper left",
        "legend_bbox_anchor": (0.02, 0.98),
        "bbn_annotation_pos": "lower left",
    }

    settings["cmb_bandpower_files"] = {
        k: settings[k] for k in ("TT", "TE", "EE") if k in settings
    }

    # Quiet LaTeX detection (no stdout spam)
    if latex_enabled:
        if shutil.which("latex"):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")
        else:
            logging.warning("LaTeX not found; falling back to default fonts")

    return settings


# ───────────────────────────────────────────────────────────────────────────────
# 5) Vectorisation detection
# ───────────────────────────────────────────────────────────────────────────────

def detect_vectorisation(models, get_model_fn, config, data, sample_n: int = 10):
    """
    Detect whether each model supports vectorised z input.
    Only rank 0 computes & logs; others receive a broadcasted dict.

    IMPORTANT:
    - This is purely about *model capability* (E(z) handling).
    - Dataset-specific constraints (CMB/BBN → prefer emcee, or
      non-vectorised Zeus) are handled later in run_mcmc via
      has_cmb / has_bbn and engine_mode.
    """

    comm = _mpi_comm()
    rank = comm.Get_rank() if comm else 0
    vectorised: Dict[str, bool] = {}

    # 0) Forced override
    if getattr(K, "force_vectorisation", False):
        vectorised = {mod: True for mod in models}
        if is_rank0():
            for mod in models:
                log.info("Model '%s' vectorised: True  (forced)", mod)
        if comm:
            vectorised = comm.bcast(vectorised, root=0)
        return vectorised

    if rank == 0:
        for mod in models:
            MODEL = get_model_fn(mod)
            obs_list = [
                obs
                for group in config[mod].get("observations", [])
                for obs in group
            ]

            # 1) Find a suitable observation that actually has a redshift grid
            z_vals = None
            tested_obs = None
            for obs in obs_list:
                # Skip scalar-only observations or string placeholders
                if obs not in data or isinstance(data[obs], str):
                    continue
                obs_data = data[obs]
                for key in ("z", "redshift", "zHD"):
                    if key in obs_data:
                        arr = np.asarray(obs_data[key], dtype=float)
                        if arr.size >= 2 and np.isfinite(arr).all():
                            z_vals = arr
                            tested_obs = obs
                            break
                if tested_obs is not None:
                    break

            if z_vals is None:
                vectorised[mod] = False
                log.info(
                    "Model '%s' vectorised: False  "
                    "(no redshift-bearing obs among %s)",
                    mod,
                    obs_list,
                )
                continue

            z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))
            if (
                not np.isfinite(z_min)
                or not np.isfinite(z_max)
                or z_max <= z_min
            ):
                z_test = np.linspace(0.0, 3.0, 20000)
            else:
                z_test = np.linspace(z_min, z_max, 20000)

            names = config[mod]["parameters"][0]
            tv = config[mod]["reference_values"][0]
            params = dict(zip(names, tv))

            try:
                t0 = _time.time()
                out = MODEL(z_test, params)
                dt = _time.time() - t0
                is_vec = (
                    isinstance(out, np.ndarray)
                    and out.shape == z_test.shape
                    and dt < 0.01
                )
            except Exception as e:
                log.warning(
                    "Vectorisation test failed for model %s on obs %s: %s",
                    mod,
                    tested_obs,
                    e,
                )
                is_vec = False

            vectorised[mod] = is_vec
            log.info("Model '%s' vectorised: %s", mod, is_vec)

    if comm:
        vectorised = comm.bcast(vectorised if rank == 0 else None, root=0)

    return vectorised



# ───────────────────────────────────────────────────────────────────────────────
# 6) Output directories / labels / Pantheon+ covariance
# ───────────────────────────────────────────────────────────────────────────────

def prepare_output(model_name: str, obs_key: str, suffix: str = "") -> str:
    """
    Create and return the output directory for a given model/observation key:
        MCMC_Chains[/suffix]/<model_name>/<obs_key>
    """
    base = (
        os.path.join("MCMC_Chains", suffix, model_name)
        if suffix
        else os.path.join("MCMC_Chains", model_name)
    )
    os.makedirs(base, exist_ok=True)
    out = os.path.join(base, obs_key)
    os.makedirs(out, exist_ok=True)
    return out


def compute_pantheon_cov(
    data,
    config_model,
    comm,
    rank,
    cov_file,
    obs_tag: str = "PantheonP",
):
    """
    Load Pantheon+ covariance, handle 1D flattened arrays (with or without a
    leading N header), select the submatrix for the kept SNe, return lower
    Cholesky factor L (L @ L.T = sub_cov). MPI-broadcast if needed.

    obs_tag controls which Pantheon-like dataset to use, e.g. "PantheonP"
    or "PantheonPS".
    """
    # Only do work if this tag is part of the model
    has_pant = any(
        obs_tag in grp for grp in config_model.get("observations", [])
    )
    if not has_pant or obs_tag not in data:
        return None

    # Indices to keep
    idx = data[obs_tag].get("indices", None)
    mask = data[obs_tag].get("mask", None)
    if idx is None and mask is not None:
        idx = np.where(mask)[0]
    if idx is None:
        n = len(data.get(obs_tag, {}).get("zHD", []))
        idx = np.arange(n, dtype=int)
    idx = np.asarray(idx, dtype=int)

    def _load_cov_matrix(path: str) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".npy", ".npz"):
            arr = np.load(path, allow_pickle=False)
            if hasattr(arr, "files"):  # .npz
                # pick the first array if multiple
                arr = arr[arr.files[0]]
        else:
            # Robust text loader (skips empty/comment lines automatically)
            arr = np.loadtxt(path)
        arr = np.asarray(arr, dtype=float)

        if arr.ndim == 2:
            return arr

        # 1D → try reshape to square
        size = arr.size
        N = int(np.sqrt(size))
        if N * N == size:
            return arr.reshape(N, N)

        # Handle common case: first element is N, followed by N^2 values
        M = size - 1
        N2 = int(np.sqrt(M))
        if N2 * N2 == M:
            return arr[1:].reshape(N2, N2)

        raise ValueError(
            f"Cannot reshape covariance: size={size} is not K^2 or 1+K^2"
        )

    L = None
    if rank == 0:
        cov_full = _load_cov_matrix(cov_file)
        N = cov_full.shape[0]

        if idx.max(initial=-1) >= N:
            raise ValueError(
                f"{obs_tag} index out of bounds: max(idx)={idx.max()} but cov has N={N}.\n"
                "Check that your Pantheon+ filtering/mask matches the covariance file."
            )

        # Subselect and sanitize
        sub = cov_full[np.ix_(idx, idx)]
        # Numerical symmetrization (just in case)
        sub = 0.5 * (sub + sub.T)

        # Cholesky with tiny jitter fallback if needed
        for eps in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
            try:
                L = la.cholesky(
                    sub + (eps * np.eye(sub.shape[0])),
                    lower=True,
                    check_finite=False,
                )
                break
            except la.LinAlgError:
                continue
        if L is None:
            raise ValueError(
                "Pantheon+ covariance is not positive definite even after jitter."
            )

    # Broadcast to workers
    if comm is not None:
        L = comm.bcast(L, root=0)
    return L


def apply_pantheon_cov(data: Dict[str, Any], obs_set: List[str], pantheon_cov, obs_tag: str = "PantheonP"):
    if obs_tag in obs_set and pantheon_cov is not None and obs_tag in data:
        data[obs_tag]["cov"] = pantheon_cov
    return data


def cleanup_pantheon_cov(data: Dict[str, Any], obs_tag: str = "PantheonP"):
    if obs_tag in data:
        data[obs_tag].pop("cov", None)
    return data


# ───────────────────────────────────────────────────────────────────────────────
# 7) Chain load-or-run orchestration
# ───────────────────────────────────────────────────────────────────────────────

def load_or_run_chain(
    output_dir: str,
    chain_file: str,
    overwrite: bool,
    CONFIG_model: Dict[str, Any],
    data: Dict[str, Any],
    MODEL_func,
    convergence: float,
    parallel: bool,
    pool,
    vectorised: bool,
    resumeChains: bool = False,
    **other_kwargs,
):
    """
    Orchestrate loading/resuming (emcee) or running a fresh chain
    (emcee or Zeus). Handles the emcee resume logic but also allows
    cross-engine reuse of existing chains.
    """
    from Kosmulator_main import Kosmulator_MCMC

    if h5py is None or emcee is None:
        raise RuntimeError(
            "h5py/emcee required for chain handling but not available."
        )

    # Canonical filenames for the two engines
    chain_path = os.path.join(output_dir, chain_file)           # emcee
    zeus_chain = chain_path.replace(".h5", "_zeus.h5")          # Zeus
    
    engine_mode = getattr(K, "engine_mode", "mixed")
    force_emcee = bool(getattr(K, "force_emcee", False))

    # Decide which engine this obs-set will actually use
    can_vec    = bool(vectorised)
    model_name = str(other_kwargs.get("model_name", ""))
    obs_types  = [str(t) for t in (other_kwargs.get("Type") or [])]
    obs_lower  = [t.lower() for t in obs_types]

    has_cmb = any(t.startswith("cmb_") for t in obs_lower)
    has_bbn = any("bbn" in t for t in obs_lower)

    # Reconstruct engine choice (mirrors Kosmulator_MCMC._choose_engine logic)
    if force_emcee:
        engine = "emcee"
    elif getattr(K, "force_zeus", False) and zeus is not None:
        engine = "zeus"
    else:
        if engine_mode in ("single", "mixed"):
            eng_map = getattr(K, "engine_for_model", {})
            eng = eng_map.get(model_name)
            if eng in ("zeus", "emcee"):
                engine = eng
            else:
                # Fallback: prefer Zeus if possible
                engine = "zeus" if (can_vec and zeus is not None) else "emcee"
        elif engine_mode == "fastest":
            if (not has_cmb) and (not has_bbn) and can_vec and (zeus is not None):
                engine = "zeus"
            else:
                engine = "emcee"
        else:
            # Unknown mode -> behave like mixed
            engine = "zeus" if (can_vec and zeus is not None) else "emcee"

    # One sampled parameter: Kosmulator_MCMC runs emcee instead of zeus (zeus's
    # differential move is unreliable in one dimension), so the chain is loaded or
    # resumed as an emcee chain (otherwise a rerun appended a new run to it).
    if engine == "zeus":
        try:
            if int(CONFIG_model["ndim"][other_kwargs.get("obs_index")]) == 1:
                engine = "emcee"
        except (KeyError, IndexError, TypeError, ValueError):
            pass

    is_zeus_run  = (engine == "zeus" and zeus is not None)
    is_emcee_run = not is_zeus_run
    
    # ------------------------------------------------------------------
    # Cross-engine reuse I:
    #   EMCEE run sees an existing Zeus chain → reuse it instead of
    #   recomputing, as long as we are not overwriting or resuming.
    # ------------------------------------------------------------------
    if (
        not force_emcee
        and engine_mode != "single"
        and is_emcee_run          # instead of "not vectorised"
        and not overwrite
        and not resumeChains
        and (not os.path.exists(chain_path))
        and os.path.exists(zeus_chain)
    ):
        print(
            "[INFO] EMCEE requested but found existing Zeus chain.\n"
            f"       Re-using samples from {zeus_chain}.\n"
        )
        burn = int(CONFIG_model.get("burn", 0) or 0)
        with h5py.File(zeus_chain, "r") as f:
            all_samples = f["samples"][:]    # (nsteps, nwalker, ndim)
            # Guard against silly values
            if burn >= all_samples.shape[0]:
                burn = 0
            # Step-major, aligned with `flat` below (older walker-major files are
            # reordered); None when the stored log_like cannot be aligned.
            loglike = zeus_flat_log_like(f, burn)

        flat = all_samples[burn:, :, :].reshape(-1, all_samples.shape[-1])
        if loglike is not None and loglike.shape[0] != flat.shape[0]:
            loglike = None
        return {"samples": flat, "loglike": loglike}

        
    # A) Existing chain, not overwriting (emcee only)
    # For vectorised/Zeus runs we *always* delegate loading/resume to
    # Kosmulator_MCMC.run_mcmc, which knows about "<name>_zeus.h5".
    if is_emcee_run and os.path.exists(chain_path) and not overwrite:
        # A0) Already converged (autocorr) → fast-path load
        with h5py.File(chain_path, "r") as h5f:
            if h5f.attrs.get("converged", False):
                print(f"[INFO] Chain flagged converged → loading: {chain_path}\n")
                return Kosmulator_MCMC.load_mcmc_results(
                    output_path=output_dir,
                    file_name=chain_file,
                    CONFIG=CONFIG_model,
                )

        # A1) Load without resuming
        print(f"[INFO] Loading chain from: {chain_path}\n")
        if not resumeChains:
            return Kosmulator_MCMC.load_mcmc_results(
                output_path=output_dir,
                file_name=chain_file,
                CONFIG=CONFIG_model,
            )

        # A2) Resume emcee chain ...
        backend = emcee.backends.HDFBackend(chain_path)
        completed = backend.iteration >= CONFIG_model["nsteps"]
        if completed:
            print(
                "[INFO] Chain already complete "
                f"({backend.iteration} ≥ {CONFIG_model['nsteps']}) → loading.\n"
            )
            return Kosmulator_MCMC.load_mcmc_results(
                output_path=output_dir,
                file_name=chain_file,
                CONFIG=CONFIG_model,
            )
        else:
            print(
                "[INFO] Resuming chain from "
                f"{chain_path} (step {backend.iteration}/{CONFIG_model['nsteps']})"
            )
            resumed = Kosmulator_MCMC.run_mcmc(
                data=data,
                saveChains=True,
                chain_path=chain_path,
                overwrite=overwrite,
                resumeChains=True,
                MODEL_func=MODEL_func,
                CONFIG=CONFIG_model,
                autoCorr=True,
                parallel=parallel,
                model_name=other_kwargs.get(
                    "model_name", CONFIG_model["model_name"]
                ),
                obs=other_kwargs.get("obs"),
                Type=other_kwargs.get("Type"),
                colors=other_kwargs.get("colors"),
                convergence=convergence,
                last_obs=other_kwargs.get("last_obs"),
                PLOT_SETTINGS=other_kwargs.get("PLOT_SETTINGS"),
                obs_index=other_kwargs.get("obs_index"),
                use_mpi=other_kwargs.get("use_mpi"),
                num_cores=other_kwargs.get("num_cores"),
                pool=pool,
                vectorised=vectorised,
                obs_key=other_kwargs.get("obs_key"),
            )
            # run_mcmc returns the bare samples; reload them with the saved
            # log-likelihoods, as for a fresh run, so DIC and WAIC are computed.
            if isinstance(resumed, dict):
                return resumed
            return Kosmulator_MCMC.load_mcmc_results(
                output_path=output_dir,
                file_name=chain_file,
                CONFIG=CONFIG_model,
            )

    # ------------------------------------------------------------------
    # Cross-engine reuse II:
    #   Zeus run sees an existing EMCEE chain → reuse it instead of
    #   recomputing, as long as we are not overwriting or resuming.
    # ------------------------------------------------------------------
    if (
        is_zeus_run
        and engine_mode != "single"
        and not overwrite
        and not resumeChains
        and os.path.exists(chain_path)    # EMCEE file exists
        and (not os.path.exists(zeus_chain))   # <-- added guard
    ):
        print(
            "[INFO] Zeus requested but found existing EMCEE chain.\n"
            f"       Re-using samples from {chain_path}.\n"
        )
        return Kosmulator_MCMC.load_mcmc_results(
            output_path=output_dir,
            file_name=chain_file,
            CONFIG=CONFIG_model,
        )
        
    # B) Fresh run (or Zeus load/resume)
    result = Kosmulator_MCMC.run_mcmc(
        data=data,
        saveChains=True,
        chain_path=chain_path,
        overwrite=overwrite,
        resumeChains=resumeChains,
        MODEL_func=MODEL_func,
        CONFIG=CONFIG_model,
        autoCorr=True,
        parallel=parallel,
        model_name=other_kwargs.get("model_name", CONFIG_model["model_name"]),
        obs=other_kwargs.get("obs"),
        Type=other_kwargs.get("Type"),
        colors=other_kwargs.get("colors"),
        convergence=convergence,
        last_obs=other_kwargs.get("last_obs"),
        PLOT_SETTINGS=other_kwargs.get("PLOT_SETTINGS"),
        obs_index=other_kwargs.get("obs_index"),
        use_mpi=other_kwargs.get("use_mpi"),
        num_cores=other_kwargs.get("num_cores"),
        pool=pool,
        vectorised=vectorised,
        obs_key=other_kwargs.get("obs_key"),
    )

    flat_samples = result.get("samples") if isinstance(result, dict) else result

    # Fresh runs return samples directly, but save log_like in the chain file.
    # Load it back here so post-processing can compute DIC as well as AIC/BIC.
    if not isinstance(result, dict):
        saved_path = zeus_chain if is_zeus_run else chain_path
        loglike = None
        if os.path.exists(saved_path):
            try:
                with h5py.File(saved_path, "r") as h5f:
                    if is_zeus_run and "samples" in h5f:
                        # zeus files: step-major, aligned with the returned samples
                        burn_z = int(CONFIG_model.get("burn", 0) or 0)
                        candidate = zeus_flat_log_like(h5f, burn_z)
                    elif "log_like" in h5f:
                        candidate = np.asarray(h5f["log_like"], dtype=float).reshape(-1)
                    else:
                        candidate = None
                    if candidate is not None and candidate.size == flat_samples.shape[0]:
                        loglike = candidate
            except Exception as exc:
                print(f"[WARNING] Could not load saved log_like: {exc}")
        result = {"samples": flat_samples, "loglike": loglike}

    if (
        flat_samples is None
        or flat_samples.size == 0
        or not np.any(np.isfinite(flat_samples))
    ):
        print(f"[WARNING] Samples for {chain_file} are empty or invalid!")
    return result


# ───────────────────────────────────────────────────────────────────────────────
# 8) Convergence diagnostics and stopping rule (zeus and emcee)
# ───────────────────────────────────────────────────────────────────────────────
#
# One rule for both engines, checked every `check_every` steps on the chain
# after burn-in (n_post steps x n_walkers):
#
#   1. n_post >= CONV_TAU_FACTOR x tau_max       (50 autocorrelation times)
#   2. ESS = n_post x n_walkers / tau_max >= CONV_ESS_MIN
#   3. tau_max changed by less than `convergence` (relative) since the previous check
#   4. split-Rhat of every parameter < CONV_RHAT_MAX
#
# tau_max is the largest integrated autocorrelation time over the sampled
# parameters, estimated with zeus's default method (autocorr_time_mk). The run
# may stop at the first check at or after burn + autocorr_buffer_after_burn
# (default 0) where all four hold for `tau_consecutive` (default 2) checks in a row. The saved chain's
# "converged" attribute records whether the rule was met.

def _acf_1d(x: np.ndarray) -> Optional[np.ndarray]:
    """Normalised autocorrelation function of a 1-D series (FFT), or None if it is constant."""
    x = np.asarray(x, dtype=float)
    n = 1
    while n < len(x):
        n <<= 1
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    if not (acf[0] > 0):
        return None
    return acf / acf[0]


def autocorr_time_mk(chain, c: float = 5.0) -> np.ndarray:
    """
    Integrated autocorrelation time of every parameter of an ensemble chain
    (n_steps, n_walkers, n_dim), with zeus's default estimator ("mk";
    Karamanis, Beutler & Peacock 2021; zeus.autocorr.AutoCorrTime): the walkers' chains
    are joined end to end, the autocorrelation function is taken about the
    overall mean and summed up to Sokal's automated window (M >= c tau).

    Unlike the emcee estimator, which averages the autocorrelation functions of
    the walkers about each walker's own mean, this keeps the differences between
    the walkers' means, so a parameter whose walkers drift slowly is not
    reported as well mixed. Returns NaN for a parameter that does not move.
    """
    x = np.asarray(chain, dtype=float)
    n, w, d = x.shape
    taus = np.full(d, np.nan)
    for k in range(d):
        f = _acf_1d(x[:, :, k].T.reshape(-1))      # walker after walker
        if f is None:
            continue
        t = 2.0 * np.cumsum(f) - 1.0
        m = np.arange(len(t)) < c * t
        taus[k] = t[int(np.argmin(m))] if np.any(m) else t[-1]
    return taus


def split_rhat(chain) -> np.ndarray:
    """
    Split-Rhat of every parameter (Gelman et al. 2013, Bayesian Data Analysis,
    Sec. 11.4): each walker's chain is cut into two halves and the
    2 x n_walkers half-chains are compared (between- and within-chain variance).
    Values above ~1.01 mean that walkers, or the first and second halves of the
    chain, still sample different regions.
    """
    x = np.asarray(chain, dtype=float)
    n, w, d = x.shape
    h = n // 2
    if h < 2:
        return np.full(d, np.nan)
    parts = np.concatenate([x[:h], x[h:2 * h]], axis=1)          # (h, 2w, d)
    means = parts.mean(axis=0)
    W = parts.var(axis=0, ddof=1).mean(axis=0)
    B = h * means.var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(((h - 1) / h * W + B / h) / W)


def convergence_rules(PLOT_SETTINGS: Optional[dict] = None, convergence: Optional[float] = None) -> Dict[str, float]:
    """Thresholds of the stopping rule (constants, overridable through PLOT_SETTINGS)."""
    ps = PLOT_SETTINGS or {}
    return {
        "tau_factor": float(ps.get("conv_tau_factor", K.CONV_TAU_FACTOR)),
        "ess_min": float(ps.get("conv_ess_min", K.CONV_ESS_MIN)),
        "rhat_max": float(ps.get("conv_rhat_max", K.CONV_RHAT_MAX)),
        "rtol": float(convergence if convergence is not None else ps.get("conv_tau_rtol", K.CONV_TAU_RTOL)),
    }


def convergence_status(post_chain, rules: Dict[str, float], tau_prev: Optional[float] = None,
                       param_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """All quantities of the stopping rule for a post-burn-in chain (n_post, n_walkers, n_dim)."""
    x = np.asarray(post_chain, dtype=float)
    n, w, d = x.shape
    names = list(param_names) if param_names else [f"p{i}" for i in range(d)]
    tau = autocorr_time_mk(x)
    rhat = split_rhat(x)
    j = int(np.nanargmax(tau)) if np.isfinite(tau).any() else 0
    jr = int(np.nanargmax(rhat)) if np.isfinite(rhat).any() else 0
    tau_max = float(tau[j]) if np.isfinite(tau[j]) else float("nan")
    rhat_max = float(rhat[jr]) if np.isfinite(rhat[jr]) else float("nan")
    n_over = n / tau_max if tau_max > 0 else float("nan")
    ess = n * w / tau_max if tau_max > 0 else float("nan")
    rel = (abs(tau_max - tau_prev) / tau_max
           if (tau_prev is not None and np.isfinite(tau_prev) and tau_max > 0) else float("nan"))
    crit = {
        "length": bool(np.isfinite(n_over) and n_over >= rules["tau_factor"]),
        "ess": bool(np.isfinite(ess) and ess >= rules["ess_min"]),
        "stable": bool(np.isfinite(rel) and rel < rules["rtol"]),
        "rhat": bool(np.isfinite(rhat_max) and rhat_max < rules["rhat_max"]),
    }
    return {
        "n_post": int(n), "walkers": int(w), "tau": tau, "tau_max": tau_max,
        "tau_param": names[j] if j < len(names) else f"p{j}",
        "n_over_tau": n_over, "ess": ess, "rel_change": rel,
        "rhat": rhat, "rhat_max": rhat_max, "rhat_param": names[jr] if jr < len(names) else f"p{jr}",
        "criteria": crit, "ok": all(crit.values()),
    }


class ConvergenceMonitor:
    """
    Applies the stopping rule during a run (zeus callback or emcee loop) and
    keeps the history that convergence_plot draws.

    update(it, chain) takes the GLOBAL chain up to step `it` (steps, walkers,
    dim), including steps from a resumed file, and returns True when the run
    may stop.
    """

    def __init__(self, burn: int, rules: Dict[str, float], check_every: int = 100,
                 earliest_stop: Optional[int] = None, consecutive: int = 1,
                 param_names: Optional[List[str]] = None):
        self.burn = int(burn)
        self.rules = dict(rules)
        self.check_every = max(1, int(check_every))
        self.earliest_stop = int(earliest_stop if earliest_stop is not None else burn)
        self.consecutive = max(1, int(consecutive))
        self.param_names = list(param_names) if param_names else None
        self.hist: Dict[str, List[Any]] = {k: [] for k in
                                           ("it", "tau_max", "tau_param", "n_post", "ess", "rel", "rhat_max", "ok")}
        # Burn-in phase, for the plot only (never used to stop): the same numbers on the
        # second half of the chain so far, as zeus's own callbacks do (discard = 0.5).
        self.pre: Dict[str, List[Any]] = {k: [] for k in ("it", "tau_max", "rel", "rhat_max")}
        self.walkers = None
        self.streak = 0
        self.stopped_at: Optional[int] = None
        self.last: Optional[Dict[str, Any]] = None

    def _evaluate_burnin(self, it: int, chain) -> None:
        """Burn-in diagnostics for the plot: chain[it//2 : it], once that holds >= check_every steps."""
        if it < 2 * self.check_every:
            return
        try:
            part = np.asarray(chain[it // 2:it], dtype=float)
            prev = self.pre["tau_max"][-1] if self.pre["tau_max"] else None
            st = convergence_status(part, self.rules, prev, self.param_names)
        except Exception:
            return
        for k, v in (("it", it), ("tau_max", st["tau_max"]), ("rel", st["rel_change"]),
                     ("rhat_max", st["rhat_max"])):
            self.pre[k].append(v)

    def _evaluate(self, it: int, chain) -> Optional[Dict[str, Any]]:
        it = int(it)
        n_post = it - self.burn
        if n_post <= 0:
            self._evaluate_burnin(it, chain)
            return None
        if n_post < max(self.check_every, 20):
            return None
        post = np.asarray(chain[self.burn:it], dtype=float)
        prev = self.hist["tau_max"][-1] if self.hist["tau_max"] else None
        st = convergence_status(post, self.rules, prev, self.param_names)
        st["it"] = it
        self.walkers = st["walkers"]
        for k, v in (("it", it), ("tau_max", st["tau_max"]), ("tau_param", st["tau_param"]),
                     ("n_post", st["n_post"]), ("ess", st["ess"]), ("rel", st["rel_change"]),
                     ("rhat_max", st["rhat_max"]), ("ok", st["ok"])):
            self.hist[k].append(v)
        self.last = st
        return st

    def update(self, it: int, chain) -> bool:
        st = self._evaluate(it, chain)
        if st is None:
            return False
        self.streak = self.streak + 1 if st["ok"] else 0
        if self.stopped_at is None and self.streak >= self.consecutive and int(it) >= self.earliest_stop:
            self.stopped_at = int(it)
            return True
        return False

    def final(self, it: int, chain) -> Dict[str, Any]:
        """Status at the end of the run; 'converged' is True if the rule stopped the run or holds now."""
        if self.last is None or self.last.get("it") != int(it):
            st = self._evaluate(it, chain)
            if st is not None:
                self.streak = self.streak + 1 if st["ok"] else 0
        st = dict(self.last or {})
        st["converged"] = bool(self.stopped_at is not None
                               or (st.get("ok", False) and self.streak >= self.consecutive))
        st["stopped_at"] = self.stopped_at
        return st

    def steps_needed(self, st: Dict[str, Any]) -> Optional[int]:
        """Rough number of extra steps for the length and ESS conditions at the current tau_max."""
        tau, w = st.get("tau_max"), st.get("walkers") or self.walkers
        if not (tau and np.isfinite(tau) and w):
            return None
        need = max(self.rules["tau_factor"] * tau, self.rules["ess_min"] * tau / w)
        return int(max(0, np.ceil(need - st.get("n_post", 0))))

    def message(self, st: Dict[str, Any], model_name: str, group: str, engine: str) -> str:
        """One-line verdict with the numbers behind each condition."""
        r = self.rules
        if not st or "tau_max" not in st:
            return (f"[{model_name} | {group}] {engine}: the chain after burn-in is too short to estimate "
                    "the autocorrelation time. Treat these results as preliminary; increase nsteps.")
        c = st["criteria"]

        def mark(ok):
            return "ok" if ok else "NOT MET"
        rel = st["rel_change"]
        parts = [
            f"tau_max = {st['tau_max']:.1f} ({st['tau_param']}), N_post/tau = {st['n_over_tau']:.0f} "
            f"(needs >= {r['tau_factor']:.0f}: {mark(c['length'])})",
            f"ESS = {st['ess']:.0f} (needs >= {r['ess_min']:.0f}: {mark(c['ess'])})",
            ("tau_max change since the previous check = "
             + ("n/a" if not np.isfinite(rel) else f"{100 * rel:.1f}%")
             + f" (needs < {100 * r['rtol']:.0f}%: {mark(c['stable'])})"),
            f"split-Rhat = {st['rhat_max']:.4f} ({st['rhat_param']}) (needs < {r['rhat_max']:.2f}: {mark(c['rhat'])})",
        ]
        if st.get("converged"):
            where = (f"stopped at step {st['stopped_at']}" if st.get("stopped_at")
                     else f"met the rule at the end ({st['it']} steps)")
            return f"[{model_name} | {group}] {engine} converged, {where}: " + "; ".join(parts) + "."
        more = self.steps_needed(st)
        extra = f" Roughly {more} more steps are needed at this tau_max." if more else ""
        if not c["rhat"]:
            extra += " Split-Rhat falls as the walkers mix, so a longer chain is needed."
        return (f"[{model_name} | {group}] {engine} reached {st['it']} steps without meeting the "
                f"convergence rule: " + "; ".join(parts) + "." + extra
                + " Treat these results as preliminary; increase nsteps.")


def emcee_autocorr_stopping(
    pos: np.ndarray,
    sampler: "emcee.EnsembleSampler",
    nsteps: int,
    model_name: str,
    colors: list,
    obs: list,
    PLOT_SETTINGS: dict,
    convergence: float = None,
    last_obs: bool = False,
    resume_offset: int = 0,
    local_burn: Optional[int] = None,
    global_burn: Optional[int] = None,
    buffer_after_burn: Optional[int] = None,
    print_enabled: bool = False,
    print_every: int = 0,
    param_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Run emcee for up to `nsteps` more steps, checking the convergence rule
    (ConvergenceMonitor) every autocorr_check_every steps on the whole chain,
    including steps from a resumed file, and stop once it holds. Sets
    sampler.kosm_converged and sampler.kosm_monitor for the caller and draws the
    convergence plot (auto_corr/<group>.png).
    """
    import Plots.Plots as MP

    if global_burn is None:
        global_burn = local_burn or 0
    check_every = int(PLOT_SETTINGS.get("autocorr_check_every", 100))
    buffer_after_burn = int(buffer_after_burn if buffer_after_burn is not None
                            else PLOT_SETTINGS.get("autocorr_buffer_after_burn", 0))
    try:
        print_every = max(0, int(print_every or 0))
    except Exception:
        print_every = 0

    group = generate_label(obs)
    folder = os.path.join(PLOT_SETTINGS["autocorr_save_path"], model_name, "auto_corr")
    os.makedirs(folder, exist_ok=True)
    plot_path = os.path.join(folder, f"{group.replace('+', '_')}.png")

    monitor = ConvergenceMonitor(
        burn=int(global_burn),
        rules=convergence_rules(PLOT_SETTINGS, convergence),
        check_every=check_every,
        earliest_stop=int(global_burn) + buffer_after_burn,
        consecutive=int(PLOT_SETTINGS.get("tau_consecutive", 2)),
        param_names=param_names,
    )
    sampler.kosm_converged = False
    sampler.kosm_monitor = monitor

    for sample in sampler.sample(pos, iterations=nsteps, progress=True):
        it_global = int(sampler.iteration)          # backend total (includes resumed steps)

        if (print_enabled and print_every > 0 and (it_global % print_every) == 0
                and is_rank0() and is_main_process()):
            lp = getattr(sample, "log_prob", None)
            if lp is not None:
                lp = np.asarray(lp, dtype=float)
                if lp.size:
                    print(f"[EMCEE Step {it_global}] Log-Post: Max={np.nanmax(lp):.4f} | "
                          f"Mean={np.nanmean(lp):.4f}", flush=True)

        if (it_global % check_every) != 0:
            continue
        stop = monitor.update(it_global, sampler.get_chain())
        try:
            MP.convergence_plot(monitor, plot_path, model_name, group, PLOT_SETTINGS)
        except Exception as e:
            logging.getLogger(__name__).debug("convergence plot failed: %s", e)
        if stop:
            break

    chain = sampler.get_chain()
    st = monitor.final(chain.shape[0], chain)
    sampler.kosm_converged = bool(st.get("converged", False))
    try:
        MP.convergence_plot(monitor, plot_path, model_name, group, PLOT_SETTINGS)
    except Exception:
        pass
    msg = monitor.message(st, model_name, group, "emcee")
    log = logging.getLogger(__name__)
    (log.info if sampler.kosm_converged else log.warning)(msg)

    return sampler.get_chain(discard=int(global_burn), flat=True)


class ZeusConvergenceCallback:
    """
    zeus callback: every `ncheck` steps it writes the new samples (append_writer),
    applies the convergence rule to the global chain (earlier steps of a resumed
    run + this run) and redraws the convergence plot. Also handles the one-time
    CMB precision switch and the optional log-posterior printout.
    """

    def __init__(self, monitor: "ConvergenceMonitor", ncheck: int, done: int = 0, prefix_chain=None,
                 plot_func=None, append_writer=None, precision_switch_iter: Optional[int] = None,
                 fine_kwargs: Optional[dict] = None, debug: bool = False):
        self.monitor = monitor
        self.ncheck = max(1, int(ncheck))
        self.done = int(done)
        self.prefix = None if prefix_chain is None else np.asarray(prefix_chain, dtype=float)
        self.plot_func = plot_func
        self.append_writer = append_writer
        self.precision_switch_iter = precision_switch_iter
        self.fine_kwargs = fine_kwargs
        self.debug = debug
        self._switched = False

    def __call__(self, iteration, chain, log_prob):
        if (self.precision_switch_iter is not None and not self._switched
                and iteration >= int(self.precision_switch_iter)):
            try:
                from Kosmulator_main import Statistical_packages as SP
                SP.set_precision("fine", **(self.fine_kwargs or {}))
                self._switched = True
            except Exception as e:
                if self.debug:
                    print(f"[Zeus] precision switch failed: {e}")

        if K.print_loglike and (iteration % K.print_loglike_every == 0):
            valid_lp = log_prob[np.isfinite(log_prob)] if log_prob is not None else np.array([])
            if valid_lp.size > 0:
                print(f"[Zeus Step {iteration}] Log-Post: Max={np.max(valid_lp):.4f} | "
                      f"Mean={np.mean(valid_lp):.4f}", flush=True)

        if iteration % self.ncheck != 0:
            return False
        if callable(self.append_writer):
            try:
                self.append_writer(iteration, chain, log_prob)
            except Exception:
                if self.debug:
                    print("[Zeus] append_writer raised; ignored.")
        full = chain if self.prefix is None else np.concatenate([self.prefix, chain], axis=0)
        stop = self.monitor.update(self.done + int(iteration), full)
        if callable(self.plot_func):
            try:
                self.plot_func(self.monitor)
            except Exception:
                if self.debug:
                    print("[Zeus] plot_func raised; ignored.")
        return stop


def make_zeus_callbacks(monitor: "ConvergenceMonitor", ncheck: int, done: int = 0, prefix_chain=None,
                        plot_func=None, append_writer=None, precision_switch_iter: Optional[int] = None,
                        fine_kwargs: Optional[dict] = None, debug: bool = False):
    """The zeus callback list used by Kosmulator (one ZeusConvergenceCallback)."""
    return [ZeusConvergenceCallback(monitor, ncheck, done, prefix_chain, plot_func, append_writer,
                                    precision_switch_iter, fine_kwargs, debug)]


# ───────────────────────────────────────────────────────────────────────────────
# 10) Stats / tables to disk
# ───────────────────────────────────────────────────────────────────────────────

def _obs_col_width(stats_list, base=38, wmin=30, wmax=68, pad=2):
    """
    Pick a nice width for the Observation column.
    - Start from `base`
    - If a longer name appears, grow up to `wmax`
    - Never shrink below `wmin`
    """
    try:
        maxlen = max(
            len(str(s.get("Observation", ""))) for s in stats_list
        ) + pad
    except ValueError:
        maxlen = base
    return max(wmin, min(wmax, max(base, maxlen)))


def save_stats_to_file(model: str, folder: str, stats_list: List[Dict[str, float]]) -> None:
    file_path = os.path.join(folder, "stats_summary.txt")

    # Dynamic width
    obs_w = _obs_col_width(stats_list, base=38, wmin=30, wmax=68, pad=2)

    header = (
        f"{'Observation':<{obs_w}} | {'Log-Likelihood':>18} | "
        f"{'Chi-Squared':>15} | {'Reduced Chi-Squared':>20} | "
        f"{'AIC':>11} | {'BIC':>11} | {'AICc':>11} | {'DIC':>11} | {'WAIC':>11} | {'dAIC':>11} | {'dBIC':>11} | {'dAICc':>11} | {'dDIC':>11} | {'dChi':>11} | {'Sigma':>11} | {'dWAIC':>11} |"
    )

    import numpy as _np

    def _as_float(x):
        """Coerce numpy scalars/arrays to a Python float for formatting."""
        try:
            arr = _np.asarray(x)
            if arr.ndim == 0:
                return float(arr)
            if arr.size == 1:
                return float(arr.reshape(()))
            # fallbacks for unexpected vectors: finite mean or NaN
            if _np.isfinite(arr).any():
                return float(_np.nanmean(arr))
            return float("nan")
        except Exception:
            try:
                return float(x)
            except Exception:
                return float("nan")

    with open(file_path, "w") as f:
        f.write(f"Statistical Results for Model: {model}\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for s in stats_list:
            obs = str(s.get("Observation", ""))
            ll = _as_float(s.get("Log-Likelihood", _np.nan))
            chi2 = _as_float(s.get("Chi_squared", _np.nan))
            rchi = _as_float(s.get("Reduced_Chi_squared", _np.nan))
            aic = _as_float(s.get("AIC", _np.nan))
            bic = _as_float(s.get("BIC", _np.nan))
            aicc = _as_float(s.get("AICc",_np.nan))
            dic = _as_float(s.get("DIC",_np.nan))
            waic = _as_float(s.get('WAIC',_np.nan))
            dwaic = _as_float(s.get('dWAIC',_np.nan))
            daic = _as_float(s.get("dAIC", _np.nan))
            dbic = _as_float(s.get("dBIC", _np.nan))
            daicc = _as_float(s.get("dAICc",_np.nan))
            ddic = _as_float(s.get("dDIC",_np.nan))
            dchi = _as_float(s.get("dChi", _np.nan))
            sigma = _as_float(s.get('sigma', _np.nan))
            row = (
                f"{obs:<{obs_w}} | {ll:>18.4f} | {chi2:>15.4f} | "
                f"{rchi:>20.4f} | {aic:>11.3f} | {bic:>11.3f} | {aicc:>11.3f} | {dic:>11.3f} | {waic:>11.3f} | "
                f"{daic:>11.3f} | {dbic:>11.3f} | {daicc:>11.3f} | {ddic:>11.3f} | {dchi:>11.3f} |{sigma:>11.3f} | {dwaic:>11.3f} | "
            )
            f.write(row + "\n")
        # Per-group notes (why k exceeds the sampled parameters, H_0 fixed or unconstrained, ...)
        notes = [(str(s.get("Observation", "")), str(s.get("Note") or "")) for s in stats_list]
        notes = [(o, n) for o, n in notes if n]
        if notes:
            f.write("Notes:\n")
            for o, n in notes:
                f.write(f"  {o}: {n}\n")
        f.write("\n")


def _interp_prose(text: str) -> str:
    """Diagnostic text from provide_model_diagnostics as one paragraph with the file's symbols."""
    import re as _re
    t = str(text or "")
    t = t.replace("Statistical Interpretation:", " ")
    t = _re.sub(r"Benchmark Comparison \(Relative to [^)]*\):", " ", t)
    t = _re.sub(r"\s*\n\s*-\s*", " ", t)
    t = _re.sub(r"^\s*-\s*", "", t.strip())
    t = " ".join(t.split())
    for a, b in (("chi^2_nu", "χ²_ν"), ("P(chi^2", "P(χ²"), ("chi^2", "χ²"), ("+/-", "±"),
                 ("dChi", "Δχ²"), ("dAIC", "ΔAIC"), ("dBIC", "ΔBIC"), ("dDIC", "ΔDIC")):
        t = t.replace(a, b)
    # "sigma" as a word only (dataset tags such as f_sigma_8 stay as they are)
    t = _re.sub(r"(?<![\w])sigma(?![\w])", "σ", t)
    return t


def _ic_words(line: str) -> str:
    """'Delta AIC: Slight evidence ... (ΔAIC = -0.67).' -> 'Slight evidence ...' (wording unchanged)."""
    import re as _re
    m = _re.match(r"^\s*(?:Delta \w+|Significance):\s*(.*?)\s*\((?:Δ\w+|Sigma)\s*=\s*[^)]*\)\.?\s*$", str(line))
    return m.group(1) if m else str(line).strip()


def save_interpretations_to_file(
    model: str,
    folder: str,
    interpretations_list: List[Dict[str, Any]],
    reference_model: Optional[str] = None,
) -> None:
    """
    interpretations_summary.txt: one block per observation group.

      Fit     chi^2, dof, chi^2_nu and its reading (provide_model_diagnostics)
      Δχ², ΔAIC, ΔBIC, ΔAICc, ΔDIC, ΔWAIC with the wording of interpret_delta_IC
              (model minus reference; omitted for the reference model itself)
    """
    def num(x, fmt):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not np.isfinite(v) else format(v, fmt)

    width, ind = 96, " " * 12
    is_ref = reference_model is not None and model == reference_model
    file_path = os.path.join(folder, "interpretations_summary.txt")
    out = [f"Interpretations for model {model}"
           + (" (the reference model)" if is_ref else
              (f", compared with the reference model {reference_model}" if reference_model else ""))]
    if not is_ref:
        out.append(f"Δ = {model} minus reference: negative values favour {model}.")
    out.append("=" * width)
    for it in interpretations_list:
        out.append("")
        out.append(str(it.get("Observation", "")))
        out.append("-" * width)
        chi2, dof, rchi = it.get("chi2"), it.get("dof"), it.get("chi2_nu")
        head = f"χ² = {num(chi2, '.2f')} for ν = {num(dof, '.0f')}, χ²_ν = {num(rchi, '.3f')}"
        out.append(f"  {'Fit':<10}" + head)
        for ln in textwrap.wrap(_interp_prose(it.get("Reduced Chi2 Diagnostics", "")), width=width - len(ind)):
            out.append(ind + ln)
        if is_ref:
            out.append(f"  {'Δ':<10}Reference model: every Δ is zero by definition.")
            continue
        dchi, sig, dk = it.get("dChi"), it.get("sigma"), it.get("dk")
        extra = (f" with {int(dk)} extra parameter{'s' if int(dk) != 1 else ''}"
                 if dk is not None and np.isfinite(float(dk)) and int(dk) > 0 else "")
        sig_words = _ic_words(it.get("Significance Interpretation", ""))
        out.append(f"  {'Δχ²':<10}{num(dchi, '+.2f'):>8}{extra}: {num(sig, '.1f')}σ ({sig_words})")
        for lab, key in (("ΔAIC", "AIC"), ("ΔBIC", "BIC"), ("ΔAICc", "AICc"), ("ΔDIC", "DIC"), ("ΔWAIC", "WAIC")):
            out.append(f"  {lab:<10}{num(it.get('d' + key), '+.2f'):>8}  {_ic_words(it.get(key + ' Interpretation', ''))}")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def _obs_col_width_from_names(names, base=30, wmin=28, wmax=72, pad=2):
    try:
        longest = max(len(str(n)) for n in names) + pad
    except ValueError:
        longest = base
    return max(wmin, min(wmax, max(base, longest)))


# ───────────────────────────────────────────────────────────────────────────────
# 11) Zeus I/O callbacks
# ───────────────────────────────────────────────────────────────────────────────

class AppendProgressCallback:
    """
    Zeus callback that appends only new local samples to the 'samples' dataset,
    and the matching log-probabilities to 'log_prob_chain' (steps, walkers).

    Both datasets grow together, so a resumed run keeps samples and
    log-probabilities aligned step by step. Pass force=True for the final
    flush, so steps after the last multiple of ncheck are not lost.
    """
    def __init__(self, filename: str, ncheck: int):
        self.filename = filename
        self.ncheck = int(ncheck)
        self._prev_local = 0

    def __call__(self, iteration: int, chain: np.ndarray, log_prob: np.ndarray, force: bool = False):
        if h5py is None:
            return False
        if (not force) and iteration % self.ncheck != 0:
            return False

        local_n, nwalker, ndim = chain.shape
        new_local = local_n - self._prev_local
        if new_local <= 0:
            return False

        new_block = chain[self._prev_local:local_n]
        lp = None if log_prob is None else np.asarray(log_prob, dtype=float)
        new_lp = None
        if lp is not None and lp.shape[:2] == (local_n, nwalker):
            new_lp = lp[self._prev_local:local_n]
        with h5py.File(self.filename, "a") as f:
            if "samples" not in f:
                old_n = 0
                f.create_dataset(
                    "samples",
                    data=new_block,
                    maxshape=(None, nwalker, ndim),
                    chunks=(1, nwalker, ndim),
                )
            else:
                ds = f["samples"]
                old_n = ds.shape[0]
                new_n = old_n + new_local
                ds.resize((new_n, nwalker, ndim))
                ds[old_n:new_n, :, :] = new_block
            if new_lp is not None:
                if "log_prob_chain" not in f:
                    if old_n == 0:
                        f.create_dataset(
                            "log_prob_chain",
                            data=new_lp,
                            maxshape=(None, nwalker),
                            chunks=(1, nwalker),
                        )
                    # A chain resumed from a file written before log_prob_chain
                    # existed cannot be aligned, so nothing is stored for it.
                else:
                    dl = f["log_prob_chain"]
                    if dl.shape[0] == old_n:
                        dl.resize((old_n + new_local, nwalker))
                        dl[old_n:old_n + new_local, :] = new_lp

        self._prev_local = local_n
        return False


# ───────────────────────────────────────────────────────────────────────────────
# Convergence report (per model and observation group)
# ───────────────────────────────────────────────────────────────────────────────

CONVERGENCE_NOTE = (
    "tau_max: largest integrated autocorrelation time over the sampled parameters, on "
    "the chain after burn-in, with zeus's default estimator (walkers joined end to end, "
    "Sokal window c = 5; used by the stopping rule for both engines). tau_emcee: the "
    "emcee estimator (per-walker autocorrelation averaged), for comparison; it can be much "
    "smaller when the walkers' means drift slowly. N_post: retained steps per walker. "
    "ESS = N_post x walkers / tau_max. Split-Rhat: largest over the parameters, each "
    "walker's chain cut in two halves (Gelman et al. 2013). Acceptance: emcee, mean over "
    "walkers, including burn-in; zeus (slice sampling) has no rejection step. Converged: "
    "the stopping rule was met (N_post >= {tau_factor:g} tau_max, ESS >= {ess_min:g}, "
    "tau_max stable to {rtol_pct:g}% between checks, split-Rhat < {rhat_max:g}). Stuck: walkers "
    "whose post-burn-in median log P lies beyond the chi2(ndim) 1 - 1e-6 point below the ensemble."
)


def chain_convergence_summary(output_dir: str, key: str, burn: int, param_names: List[str],
                              rules: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Convergence numbers for one chain file (zeus '<key>_zeus.h5' or emcee '<key>.h5'
    in output_dir; the more recently written one if both exist), computed with
    convergence_status, i.e. the same quantities as the stopping rule.
    """
    import emcee as _emcee
    rules = rules or convergence_rules()
    row: Dict[str, Any] = dict(
        observation=key, engine="n/a", steps=0, walkers=0, burn=int(burn), retained=0,
        tau_max=float("nan"), tau_param="", tau_emcee=float("nan"), n_over_tau=float("nan"),
        ess_min=float("nan"), rhat=float("nan"), rhat_param="", acceptance=float("nan"),
        converged=None, stuck=None,
    )
    paths = [(p, e) for p, e in ((os.path.join(output_dir, f"{key}_zeus.h5"), "zeus"),
                                  (os.path.join(output_dir, f"{key}.h5"), "emcee")) if os.path.exists(p)]
    if not paths or h5py is None:
        return row
    path, engine = max(paths, key=lambda pe: os.path.getmtime(pe[0]))
    lp = None
    with h5py.File(path, "r") as f:
        if engine == "zeus":
            chain = np.asarray(f["samples"][:], float)
            if "log_prob_chain" in f:
                lp = np.asarray(f["log_prob_chain"][: chain.shape[0]], float)
            stuck_attr = f.attrs.get("stuck_walkers", None)
            acc = float("nan")
        else:
            g = f["mcmc"]
            it = int(g.attrs.get("iteration", g["chain"].shape[0]))
            chain = np.asarray(g["chain"][:it], float)
            lp = np.asarray(g["log_prob"][:it], float) if "log_prob" in g else None
            acc = float(np.mean(np.asarray(g["accepted"], float)) / it) if ("accepted" in g and it > 0) else float("nan")
            stuck_attr = None
        conv = f.attrs.get("converged", None)
    nsteps, nwalk, ndim = chain.shape
    b = int(burn) if int(burn) < nsteps else 0
    post = chain[b:]
    row.update(engine=engine, steps=int(nsteps), walkers=int(nwalk), burn=b, retained=int(post.shape[0] * nwalk),
               acceptance=acc, converged=(None if conv is None else bool(conv)))
    if post.shape[0] >= 4:
        st = convergence_status(post, rules, None, param_names)
        row.update(tau_max=st["tau_max"], tau_param=st["tau_param"], n_over_tau=st["n_over_tau"],
                   ess_min=st["ess"], rhat=st["rhat_max"], rhat_param=st["rhat_param"])
        lg = logging.getLogger("emcee.autocorr")
        old = lg.level
        lg.setLevel(logging.ERROR)
        try:
            row["tau_emcee"] = float(np.nanmax(_emcee.autocorr.integrated_time(post, quiet=True)))
        except Exception:
            pass
        finally:
            lg.setLevel(old)
    if stuck_attr is not None:
        row["stuck"] = int(np.size(stuck_attr))
    elif lp is not None and lp.shape[0] > b + 1:
        row["stuck"] = int(len(find_stuck_walkers(lp[b:], ndim)))
    return row


def _fmt_conv_row(r: Dict[str, Any]) -> List[str]:
    def num(x, fmt):
        return "n/a" if (x is None or not np.isfinite(x)) else format(x, fmt)
    return [
        str(r["observation"]), str(r["engine"]), str(r["steps"]), str(r["burn"]), str(r["retained"]),
        (num(r["tau_max"], ".1f") + (f" ({r['tau_param']})" if r["tau_param"] else "")),
        num(r.get("tau_emcee"), ".1f"),
        num(r["n_over_tau"], ".1f"), num(r["ess_min"], ".0f"),
        (num(r.get("rhat"), ".4f") + (f" ({r['rhat_param']})" if r.get("rhat_param") else "")),
        ("n/a" if r["engine"] == "zeus" else num(r["acceptance"], ".3f")),
        ("n/a" if r["converged"] is None else ("yes" if r["converged"] else "no")),
        ("n/a" if r["stuck"] is None else str(r["stuck"])),
    ]


CONVERGENCE_HEADER = ["Observation", "Engine", "Steps", "Burn-in", "Retained samples", "tau_max (param)",
                      "tau_emcee", "N_post/tau_max", "ESS", "Split-Rhat (param)", "Acceptance", "Converged",
                      "Stuck walkers"]


def format_convergence_table(model: str, rows: List[Dict[str, Any]],
                             rules: Optional[Dict[str, float]] = None) -> str:
    """Aligned plain-text convergence table for one model."""
    rules = rules or convergence_rules()
    body = [_fmt_conv_row(r) for r in rows]
    widths = [max(len(h), *(len(b[i]) for b in body)) if body else len(h) for i, h in enumerate(CONVERGENCE_HEADER)]
    line = lambda cells: " | ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))
    out = [f"Convergence summary for model: {model}", line(CONVERGENCE_HEADER), "-" * len(line(CONVERGENCE_HEADER))]
    out += [line(b) for b in body]
    note = CONVERGENCE_NOTE.format(tau_factor=rules["tau_factor"], ess_min=rules["ess_min"],
                                   rtol_pct=100 * rules["rtol"], rhat_max=rules["rhat_max"])
    out += ["", textwrap.fill(note, width=110)]
    return "\n".join(out)


def write_convergence_reports(model: str, rows: List[Dict[str, Any]], folder: str,
                              rules: Optional[Dict[str, float]] = None) -> List[str]:
    """Write convergence_summary.txt for one model; returns the paths written."""
    os.makedirs(folder, exist_ok=True)
    paths = []
    p = os.path.join(folder, "convergence_summary.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(format_convergence_table(model, rows, rules) + "\n")
    paths.append(p)
    return paths


def find_stuck_walkers(log_prob_chain, ndim: int, p_tail: float = 1e-6) -> np.ndarray:
    """
    Walkers whose median log-posterior over the given (post-burn-in) steps lies
    so far below the ensemble median that 2*(median - walker median) exceeds
    chi2.isf(p_tail, ndim). For a healthy ensemble 2*Delta(log P) follows
    roughly a chi^2 with ndim degrees of freedom, so this flags only walkers
    left behind in a different region (for example behind a prior wall).
    """
    from scipy.stats import chi2 as _chi2
    lp = np.asarray(log_prob_chain, dtype=float)
    if lp.ndim != 2 or lp.shape[0] < 2:
        return np.array([], dtype=int)
    lp = np.where(np.isfinite(lp), lp, -1e300)
    wmed = np.median(lp, axis=0)
    ref = np.median(lp)
    return np.where(2.0 * (ref - wmed) > float(_chi2.isf(p_tail, max(1, int(ndim)))))[0]


def zeus_flat_log_like(h5f, burn: int, n_rows: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Step-major flat log-likelihood for a Kosmulator zeus file, aligned with
    samples[burn:].reshape(-1, ndim).

    Uses 'log_prob_chain' (steps, walkers) when present. Older files only hold
    a flat 'log_like' in zeus's own walker-major order (order='F'); those are
    reordered here. Returns None when no aligned array can be built.
    """
    samples = h5f["samples"]
    n_rows = int(samples.shape[0] if n_rows is None else n_rows)
    nwalker = int(samples.shape[1])
    burn = max(0, min(int(burn), n_rows))
    if "log_prob_chain" in h5f and h5f["log_prob_chain"].shape[0] >= n_rows:
        return np.asarray(h5f["log_prob_chain"][burn:n_rows], dtype=float).reshape(-1)
    if "log_like" in h5f:
        ll = np.asarray(h5f["log_like"], dtype=float).reshape(-1)
        n_post = n_rows - burn
        if ll.size != n_post * nwalker:
            return None
        if h5f.attrs.get("log_like_order", "") == "step":
            return ll
        return ll.reshape(nwalker, n_post).T.reshape(-1)
    return None


# ───────────────────────────────────────────────────────────────────────────────
# 12) Init-time log collapsing (summary handler + filter)
# ───────────────────────────────────────────────────────────────────────────────

class InitSummaryHandler(logging.Handler):
    """
    Collects repetitive init-time INFO/WARNING logs and prints a compact summary.
    Use with init_summary_context() that attaches this handler and (optionally)
    a filter to suppress the raw noisy lines during init.
    """

    # Reuse the module-level patterns
    INIT_COLLAPSE_KEYS = INIT_COLLAPSE_KEYS

    POLICY_MODEL_RE = re.compile(r"\s*\(model [^)]+\)\s*$")  # strip trailing "(model ...)"

    def __init__(self, style: str = "terse"):
        super().__init__(level=logging.INFO)
        self.style = style  # "terse" | "normal" | "verbose"
        self.buffer = []    # original records (used only for verbose replay)
        # structured buckets
        self.added_params = defaultdict(lambda: defaultdict(list))  # model -> group_str -> [paramlist_str]
        self.policies = defaultdict(Counter)                        # model -> Counter(policy_message)
        self.advisories = Counter()                                 # message -> count
        self.bump_nwalker = []                                      # list[str]
        self.preprocess = []                                        # list[str]
        self._already_rendered = False

    # ---------- helpers ----------
    @staticmethod
    def _normalize_added_list(s: str):
        """'['H_0','r_d']' -> ('H_0','r_d') sorted (robust to spacing)."""
        try:
            vals = ast.literal_eval(s)
            return tuple(sorted(str(x).strip() for x in vals))
        except Exception:
            return tuple(
                sorted(
                    p.strip().strip("[]' ")
                    for p in s.split(",")
                    if p.strip()
                )
            )

    @staticmethod
    def _canon_injections_by_group(by_group: dict):
        """
        Input: {group: ["['H_0','r_d']", "['H_0','r_d']", "['H_0']"], ...}
        Output: {group: {('H_0','r_d'): 2, ('H_0',): 1}, ...}
        """
        canon = {}
        for group, entries in by_group.items():
            c = Counter(
                InitSummaryHandler._normalize_added_list(e) for e in entries
            )
            canon[group] = dict(c)
        return canon

    @classmethod
    def _strip_model(cls, msg: str) -> str:
        """Remove trailing '(model XYZ)' from a policy message."""
        return cls.POLICY_MODEL_RE.sub("", msg).strip()

    @staticmethod
    def _is_fs8_solo(msg: str) -> bool:
        low = msg.lower()
        return (
            "run solo" in low
            and ("fσ8" in msg or "fσ₈" in msg or "fsigma8" in low)
        )

    # ---------- logging.Handler API ----------
    def emit(self, record: logging.LogRecord):
        msg = record.getMessage()
        self.buffer.append((record.levelname, msg))

        if not any(
            msg.startswith(k) or k in msg for k in self.INIT_COLLAPSE_KEYS
        ):
            return

        # Route tokens to advisories (not policies)
        if self._is_fs8_solo(msg):
            self.advisories["fσ8 run solo"] += 1
            return
        if msg.strip().lower() == "f solo":
            self.advisories["f solo"] += 1
            return
        # SNe solo: Union3 / JLA / Pantheon / Pantheon+ (uncal) / DESY5
        if (
            "run solo" in msg
            and any(
                tag in msg
                for tag in ("JLA", "Pantheon", "Pantheon+ (uncal)", "Union3", "DESY5")
            )
        ):
            self.advisories["SNe_solo_H0_Mabs"] += 1
            return

        if msg.startswith("[Config] Normalised group "):
            self.preprocess.append(msg)
            return

        if msg.startswith("Added "):
            # "Added ['H_0','r_d'] to parameters for ['BAO'] in model LCDM_v"
            try:
                pre, post = msg.split(" to parameters for ")
                added = pre[len("Added ") :].strip()
                group_part, model_part = post.split(" in model ")
                group_str = group_part.strip()
                model = model_part.strip()
                self.added_params[model][group_str].append(added)
            except Exception:
                self.advisories[msg] += 1
            return

        if "Bumping nwalker from" in msg:
            self.bump_nwalker.append(msg)
            return

        if msg.startswith("Observation f alone") or msg.startswith(
            "Auto-calibrating r_d"
        ):
            self.advisories[msg] += 1
            return

        # policy-style lines (BAO singleton/combo; fσ8 singleton)
        self.policies[self._infer_model(msg)][msg] += 1

    def _infer_model(self, msg: str) -> str:
        # messages end with "(model XYZ)" → extract last token
        if "(model " in msg:
            return msg.rsplit("(model ", 1)[-1].rstrip(")")
        return "ALL"

    # ---------- final summary ----------
    def render(self, logger: logging.Logger):
        # prevent double render
        if self._already_rendered:
            return
        self._already_rendered = True

        if self.style == "verbose":
            # Replay original records as-is
            for lvl, m in self.buffer:
                getattr(logger, lvl.lower())(m)
            return

        divider = "─" * 64
        logger.info(divider)

        # Pre-processing actions (e.g., group normalisation)
        if self.preprocess:
            logger.info("Pre-processing actions")
            for m in self.preprocess:
                logger.warning("  %s", m)
            logger.info(divider)

        # Parameter injections (collapse across models if identical)
        model_inj = {
            m: self._canon_injections_by_group(byg)
            for m, byg in self.added_params.items()
        }
        inj_models = list(model_inj.keys())

        def injections_identical():
            if not inj_models:
                return True
            first = model_inj[inj_models[0]]
            return all(model_inj[m] == first for m in inj_models[1:])

        if model_inj:
            if injections_identical():
                logger.info(
                    "Parameter injections (identical for all models; "
                    "added automatically from dataset requirements)"
                )

                common = model_inj[inj_models[0]]
                labels = [str(g) for g in common.keys()]
                col_width = max(len(lbl) for lbl in labels) if labels else 0

                for group, combo in common.items():
                    compact = "; ".join(
                        (
                            (
                                ",".join(params)
                                if len(params) > 1
                                else params[0]
                            )
                            + (f" ×{cnt}" if cnt > 1 else "")
                        )
                        for params, cnt in combo.items()
                    )
                    # Left-align group label into a fixed-width column
                    logger.warning("  %-*s  → %s", col_width, group, compact)
                logger.info(divider)
            else:
                for model in inj_models:
                    logger.info(
                        "Model %s — parameter injections "
                        "(added automatically from dataset requirements)",
                        model,
                    )
                    groups = model_inj[model]
                    labels = [str(g) for g in groups.keys()]
                    col_width = max(len(lbl) for lbl in labels) if labels else 0

                    for group, combo in groups.items():
                        compact = "; ".join(
                            (
                                (
                                    ",".join(params)
                                    if len(params) > 1
                                    else params[0]
                                )
                                + (f" ×{cnt}" if cnt > 1 else "")
                            )
                            for params, cnt in combo.items()
                        )
                        logger.warning("  %-*s  → %s", col_width, group, compact)
                    logger.info(divider)

        # Policy decisions (collapsed across models)
        per_model_pols = {
            m: Counter({self._strip_model(k): c for k, c in cnt.items()})
            for m, cnt in self.policies.items()
        }
        all_models = sorted(set(per_model_pols.keys()))
        printed_any_policy = False

        def policies_identical():
            if not all_models:
                return True
            first = per_model_pols[all_models[0]]
            return all(per_model_pols[m] == first for m in all_models[1:])

        if any(per_model_pols.values()):
            if policies_identical():
                logger.info("Policy decisions (identical for all models)")
                for pol, c in per_model_pols[all_models[0]].items():
                    logger.warning(
                        "  %s%s", pol, f" ×{c}" if c > 1 else ""
                    )
                printed_any_policy = True
            else:
                for m in all_models:
                    if per_model_pols[m]:
                        logger.info("Model %s — policy decisions", m)
                        for pol, c in per_model_pols[m].items():
                            logger.warning(
                                "  %s%s", pol, f" ×{c}" if c > 1 else ""
                            )
                        printed_any_policy = True
            if printed_any_policy and (self.advisories or self.bump_nwalker):
                logger.info(divider)

        # Advisories (compact wording; unique only)
        if self.advisories:
            logger.info("General advisories / notes")
            for msg in sorted(self.advisories.keys()):
                if msg == "SNe_solo_H0_Mabs":
                    # internal aggregation token; skip printing by itself
                    continue
                if self._is_fs8_solo(msg):
                    msg = (
                        "fσ8 solo: fixed γ="
                        f"{GAMMA_FS8_SINGLETON:.2f} (GR) to avoid "
                        "σ₈–Ωₘ–γ degeneracy; add f+BAO/CC or CMB/S₈."
                    )
                elif msg == "f solo":
                    msg = (
                        "f solo: geometry weak; pair with BAO/CC or "
                        "early-time anchor."
                    )
                logger.warning("  %s", msg)

        # nwalker adjustments
        if self.bump_nwalker:
            if self.advisories:
                logger.info(divider)
            logger.info("nwalker adjustments")
            for m in self.bump_nwalker:
                logger.warning("  %s", m)

        # single final divider
        logger.info(divider)


class InitCollapseFilter(logging.Filter):
    """
    Suppress known noisy init lines unless style is 'verbose'.
    """
    def __init__(self, style: str):
        super().__init__()
        self.style = style

    def filter(self, record: logging.LogRecord) -> bool:
        if self.style == "verbose":
            return True  # pass everything through
        msg = record.getMessage()
        # Block if it matches any of our noisy patterns
        return not any(
            msg.startswith(k) or k in msg for k in INIT_COLLAPSE_KEYS
        )


@contextmanager
def init_summary_context(style: str = "terse"):
    """
    Context manager that:
      • attaches InitSummaryHandler(style)
      • temporarily suppresses init-noise on other handlers
      • on exit, renders a compact init summary.
    """
    root = logging.getLogger()
    summary_handler = InitSummaryHandler(style=style)
    root.addHandler(summary_handler)

    # Attach a temporary filter to existing handlers to suppress spam
    temp_filter = InitCollapseFilter(style=style)
    attached = []
    for h in root.handlers:
        if h is summary_handler:
            continue
        h.addFilter(temp_filter)
        attached.append(h)

    try:
        yield
    finally:
        # Remove temp filter from other handlers
        for h in attached:
            try:
                h.removeFilter(temp_filter)
            except Exception:
                pass
        # Remove our summary handler and print the compact summary
        root.removeHandler(summary_handler)
        summary_handler.render(logging.getLogger(__name__))


# ───────────────────────────────────────────────────────────────────────────────
# 13) Cosmology helpers (background, labels, f/fσ8 integrals)
# ───────────────────────────────────────────────────────────────────────────────

def asarray(z):
    """Ensure input is a 1D float ndarray."""
    return np.atleast_1d(z).astype(float)

def _scalar_or_array(x):
    """
    Return a Python float if x has exactly one element; otherwise return an ndarray.
    Robust to x being np.scalar, (1,), or (1,1), etc.
    """
    x = np.asarray(x)
    if x.size == 1:
        return x.squeeze().item()
    return x

def with_fixed_params(p: dict, model_config: Optional[dict], obs_index: Optional[int]) -> dict:
    """
    Parameter dict of one observation group completed with the values the
    configuration fixes for that group (model_config["fixed_params_by_group"],
    e.g. H_0 for uncalibrated supernovae). Returns `p` itself when nothing is
    fixed, otherwise a new dict; sampled values are never overwritten.
    """
    try:
        fixed = (model_config or {}).get("fixed_params_by_group", {}) or {}
        extra = fixed.get(obs_index) if obs_index is not None else None
        if extra is None and obs_index is not None:
            extra = fixed.get(str(obs_index))
    except AttributeError:
        extra = None
    if not extra:
        return p
    out = dict(p)
    for k, v in extra.items():
        out.setdefault(k, float(v))
    return out


def sn_fitted_offsets(obs_list, data: dict) -> int:
    """
    Number of SN magnitude offsets fitted analytically in a group (counted in k
    for AIC/BIC and the degrees of freedom): 2 for the official JLA (M_B and
    Delta_M), 1 for any other SN set whose offset is marginalised, 0 otherwise.
    """
    n = 0
    for o in obs_list or []:
        d = (data or {}).get(o) or {}
        if not isinstance(d, dict):
            continue
        if d.get("jla_full"):
            n += 2
        elif d.get("marginalise_offset", False):
            n += 1
    return n


def ensure_background_params(p: dict) -> dict:
    """
    Ensure a consistent background parameter set.

    Supports both:
      - 'H_0', 'Omega_bh^2', 'Omega_dh^2'   (CMB/BBN parametrisation)
      - 'H_0', 'Omega_m', 'Omega_b'         (BAO/geometry parametrisation)

    and fills in any missing counterparts:
      - Omega_m    ← (Omega_bh^2 + Omega_dh^2) / h^2
      - Omega_b    ← Omega_bh^2 / h^2
      - Omega_bh^2 ← Omega_b * h^2
      - Omega_dh^2 ← (Omega_m - Omega_b) * h^2
    where h = H_0 / 100.
    """
    tm = dict(p)

    h = None
    if "H_0" in tm:
        try:
            h = float(tm["H_0"]) / 100.0
        except Exception:
            h = None

    # 1) From (Omega_m, Omega_b, H_0) → physical densities
    if h is not None and "Omega_b" in tm:
        try:
            Ob = float(tm["Omega_b"])
            if "Omega_bh^2" not in tm:
                tm["Omega_bh^2"] = Ob * h * h

            if "Omega_m" in tm and "Omega_dh^2" not in tm:
                Om = float(tm["Omega_m"])
                Od = Om - Ob
                tm["Omega_dh^2"] = Od * h * h
        except Exception:
            pass

    # 2) From (Omega_bh^2, Omega_dh^2, H_0) → Omega_m, Omega_b
    if h is not None and all(k in tm for k in ("Omega_bh^2", "Omega_dh^2")):
        try:
            Obh2 = float(tm["Omega_bh^2"])
            Odh2 = float(tm["Omega_dh^2"])

            if "Omega_m" not in tm:
                tm["Omega_m"] = (Obh2 + Odh2) / (h * h)

            if "Omega_b" not in tm:
                tm["Omega_b"] = Obh2 / (h * h)
        except Exception:
            pass

    return tm



def E_of_z(z, model_func, p):
    """
    Safe wrapper returning MODEL E(z) after background reconstruction.
    """
    p = ensure_background_params(p)
    zz = np.atleast_1d(z).astype(float)
    if zz.size == 0:
        return zz
    return model_func(zz, p)


def _canonicalise_group(group):
    """
    Return a tuple of unique obs names sorted by our canonical order, then name.
    """
    seen = set()
    uniq = [g for g in group if not (g in seen or seen.add(g))]

    # unknown names get rank after known ones, but still sorted by their string
    def _key(x):
        return (_OBS_RANK.get(x, 10_000), x)

    return tuple(sorted(uniq, key=_key))


def canonicalise_and_dedup_observations(observations, logger=None):
    """
    Canonicalise order within each group and drop duplicate groups.
    Preserves the first occurrence of a canonical group.
    Optionally logs when a group was normalised or removed.
    """
    canon = []
    seen = set()
    for grp in observations or []:
        c = _canonicalise_group(list(grp))

        # Log if the input group was changed
        if list(grp) != list(c) and logger:
            logger.info("[Config] Normalised group %s → %s", grp, "+".join(c))

        # Skip duplicates after canonicalisation
        if c in seen:
            if logger:
                logger.info(
                    "[Config] Skipping duplicate observation group %s "
                    "(canonical=%s)",
                    grp,
                    "+".join(c),
                )
            continue

        seen.add(c)
        canon.append(list(c))
    return canon


def generate_label(
    obs: Union[str, List[str], tuple],
    *,
    use_types: bool = False,
    config_model: Optional[dict] = None,
    obs_index: Optional[int] = None,
    sep: str = "+",
) -> str:
    """
    Turn an observation group into a compact label (BAO+CC, PantheonP_SH0ES, ...).
    """
    # Normalize to a list of strings
    if isinstance(obs, (list, tuple)):
        names = [str(x) for x in obs]
    else:
        names = [str(obs)]

    # Optional: swap to human-readable types
    if use_types and config_model is not None and obs_index is not None:
        try:
            types = config_model.get("observation_types", [[]])[obs_index]
            names = [str(t) for t in types] if types else names
        except Exception:
            pass

    # Normalise Pantheon+ naming:
    # If the user used the SH0ES-tagged dataset ("PantheonPS"), make the label
    # explicit as "PantheonP_SH0ES". Plain "PantheonP" stays as-is.
    if "PantheonPS" in names:
        i = names.index("PantheonPS")
        names[i] = "PantheonP_SH0ES"

    return sep.join(names)


def _inject_planck_nuisance_defaults(
    reference_values: Dict[str, float],
    prior_limits: Dict[str, Tuple[float, float]],
    names: List[str],
) -> None:
    """
    Ensure priors/initials exist for all requested Planck nuisance names.
    """
    from Kosmulator_main.constants import PLANCK_NUISANCE_FIXED
    for n in names:
        if n in prior_limits and n in reference_values:
            continue
        if n in PLANCK_NUISANCE_FIXED:
            continue   # fixed in the Planck baseline: never sampled
        default = PLANCK_NUISANCE_DEFAULTS.get(n)
        if default is not None:
            tv, (lo, hi) = default
            reference_values.setdefault(n, tv)
            prior_limits.setdefault(n, (lo, hi))
        else:
            # Fallback heuristic if a name isn't in our table
            reference_values.setdefault(n, 0.0)
            prior_limits.setdefault(n, (-5.0, 5.0))


# ---------------------------------------------------------------------------
# Solo-dataset advisory logging
# ---------------------------------------------------------------------------

_FS8_TOKEN_EMITTED: bool = False
_F_TOKEN_EMITTED: bool = False


def issue_observation_warnings(CONFIG, models, *, token_mode: bool = True) -> None:
    """
    Print one-time warnings for solo datasets (SNe, f, f_sigma_8).

    Pure side-effect: logging only. Does not mutate CONFIG.
    Uses generate_label(...) to resolve things like PantheonP_SH0ES.
    """
    global _FS8_TOKEN_EMITTED, _F_TOKEN_EMITTED

    log_cfg = logging.getLogger("MCMC_setup")
    emitted: set[str] = set()

    def _warn_once(key: str, msg: str) -> None:
        if key not in emitted:
            log_cfg.warning(msg)
            emitted.add(key)

    # Normalise model list (dict keys or plain iterable)
    model_list = list(models.keys()) if isinstance(models, dict) else list(models)

    for m in model_list:
        cfg = CONFIG[m]
        obs_groups = cfg.get("observations", [])

        for i, obs_set in enumerate(obs_groups):
            # Solo = exactly one dataset in the group
            if not isinstance(obs_set, (list, tuple)) or len(obs_set) != 1:
                continue

            raw = str(obs_set[0])
            try:
                resolved = generate_label(
                    obs_set, config_model=cfg, obs_index=i
                )
            except Exception:
                resolved = "+".join(obs_set)

            rlow = resolved.lower()
            rraw = raw.lower()

            # ------------------------------------------------------------------
            # SNe solo warnings
            # ------------------------------------------------------------------
            if (rraw == "jla_legacy") or ("jla_legacy" in rlow):
                if token_mode:
                    _warn_once(f"JLA_solo::{resolved}", "JLA run solo")
                else:
                    _warn_once(
                        f"JLA_solo::{resolved}",
                        (
                            "JLA_legacy (SNe) run solo: this is the old 359-SN "
                            "file of undocumented origin, kept to reproduce "
                            "earlier runs. Use 'JLA' for the official 740-SN "
                            "likelihood. Its magnitude offset is marginalised, "
                            "so H0 is not constrained."
                        ),
                    )
                continue

            if (rraw == "jla") or ("jla" in rlow):
                if token_mode:
                    _warn_once(f"JLA_solo::{resolved}", "JLA run solo")
                else:
                    _warn_once(
                        f"JLA_solo::{resolved}",
                        (
                            "JLA (SNe) run solo: official 740-SN likelihood "
                            "with alpha_JLA and beta_JLA sampled and the "
                            "absolute magnitudes (with the host-mass step) "
                            "fitted analytically, so H0 is not constrained. "
                            "Combine with CC, BAO or CMB data for H0."
                        ),
                    )
                continue

            if (rraw == "pantheon") or (
                "pantheon" in rlow and "pantheonp" not in rlow
            ):
                if token_mode:
                    _warn_once(
                        f"Pantheon_solo::{resolved}", "Pantheon run solo"
                    )
                else:
                    _warn_once(
                        f"Pantheon_solo::{resolved}",
                        (
                            "Pantheon (SNe) run solo: not Cepheid-calibrated, "
                            "so H0–M_abs is degenerate. Combine with other "
                            "observations (e.g., CC/BAO/CMB)."
                        ),
                    )
                continue

            if (rraw == "pantheonp") or ("pantheonp" in rlow):
                shoesy = ("pantheonp_sh0es" in rlow) or ("sh0es" in rlow)
                disp = (
                    resolved.replace("PantheonP_SH0ES", "Pantheon+SH0ES")
                    .replace("PantheonP", "Pantheon+")
                )
                if not shoesy:
                    if token_mode:
                        _warn_once(
                            f"PantheonP_solo_uncal::{resolved}",
                            "Pantheon+ (uncal) run solo",
                        )
                    else:
                        _warn_once(
                            f"PantheonP_solo_uncal::{resolved}",
                            (
                                f"{disp} (SNe) run solo: without SH0ES "
                                "calibration these SNe remain uncalibrated; "
                                "H0–M_abs is degenerate. Combine with "
                                "CC/BAO/CMB."
                            ),
                        )
                continue
                
            # ------------------------------------------------------------------
            # Union3 solo warnings (Pantheon-like, no internal H0 calibrator)
            # ------------------------------------------------------------------
            if (rraw == "union3") or ("union3" in rlow):
                if token_mode:
                    _warn_once(
                        f"Union3_solo::{resolved}",
                        "Union3 run solo",
                    )
                else:
                    _warn_once(
                        f"Union3_solo::{resolved}",
                        (
                            "Union3 (SNe) run solo: no internal absolute-distance "
                            "calibrator; H0 and M_abs are strongly degenerate. "
                            "Combine with CC/BAO/CMB or an external H0 prior "
                            "(e.g., SH0ES) to break the distance-ladder "
                            "degeneracy."
                        ),
                    )
                continue

            # ------------------------------------------------------------------
            # DESY5 solo warnings (also SNe-only, H0–M_abs degenerate)
            # ------------------------------------------------------------------
            if (rraw == "desy5") or ("desy5" in rlow):
                if token_mode:
                    _warn_once(
                        f"DESY5_solo::{resolved}",
                        "DESY5 run solo",
                    )
                else:
                    _warn_once(
                        f"DESY5_solo::{resolved}",
                        (
                            "DESY5 (SNe) run solo: without external calibration, "
                            "H0 and M_abs remain strongly degenerate. "
                            "Combine with CC/BAO/BBN/CMB or an external H0 prior "
                            "to obtain meaningful H0 constraints."
                        ),
                    )
                continue

            # ------------------------------------------------------------------
            # CMB_lensing solo warnings (currently RAW-only)
            # ------------------------------------------------------------------
            #if (rraw == "cmb_lensing") or ("cmb_lensing" in rlow):
             #   msg = (
               #     "CMB_lensing run solo: using RAW (non-marginalised) Planck "
                #    "lensing likelihood. The CMB-marginalised lensing mode is "
                 #   "not yet wired into Kosmulator and will be added in a future "
                  #  "update. You can safely combine CMB_lensing with CMB_lowl, "
                   # "CMB_hil, or CMB_hil_TT; those combinations use the standard "
                    #"RAW lensing treatment as in Planck TT/TE/EE+lensing."
                #)
                #_warn_once(f"CMBLensing_solo::{resolved}", msg)
                #continue
                
            # ------------------------------------------------------------------
            # f / f_sigma_8 solo warnings
            # ------------------------------------------------------------------
            if (rraw == "f") or (rlow == "f"):
                if token_mode and not _F_TOKEN_EMITTED:
                    _F_TOKEN_EMITTED = True
                    _warn_once(f"f_solo::{resolved}", "f solo")
                elif not token_mode:
                    _warn_once(
                        f"f_solo::{resolved}",
                        (
                            "Observation f alone is not ideal for cosmology; "
                            "consider combining it with complementary data "
                            "(e.g., BAO/CC/CMB)."
                        ),
                    )
                continue

            if (rraw in ("f_sigma_8", "fσ8")) or (
                "f_sigma_8" in rlow or "fσ8" in rlow
            ):
                if token_mode and not _FS8_TOKEN_EMITTED:
                    _FS8_TOKEN_EMITTED = True
                    _warn_once(f"fs8_solo::{resolved}", "fσ8 run solo")
                elif not token_mode:
                    _warn_once(
                        f"fs8_solo::{resolved}",
                        (
                            "fσ₈ run solo: strong degeneracy between σ₈, Ωₘ, "
                            "and γ. Recommended: add f (growth-only) + BAO or "
                            "CC for geometry; or pair fσ₈ with CMB or "
                            "weak-lensing (S₈). Internally we fix "
                            f"γ≈{GAMMA_FS8_SINGLETON:.2f} (GR) for this solo "
                            "fσ₈ group to keep the run numerically stable."
                        ),
                    )
                continue


def _inject_derived_background(theta_map: dict) -> dict:
    """
    Derive Omega_m from physical densities and optionally solve for H_0
    if 100theta_s is supplied without H_0 (via CMB helper).
    """
    tm = dict(theta_map)  # work on a shallow copy

    # Derive Omega_m from (H_0, Omega_bh^2, Omega_dh^2)
    if all(k in tm for k in ("H_0", "Omega_bh^2", "Omega_dh^2")):
        try:
            h = float(tm["H_0"]) / 100.0
            if h > 0:
                Om = (
                    float(tm["Omega_bh^2"]) + float(tm["Omega_dh^2"])
                ) / (h * h)
                tm.setdefault("Omega_m", Om)
        except Exception:
            pass

    # If 100theta_s is supplied without H_0, run the CMB helper to back-solve H_0
    if "100theta_s" in tm and "H_0" not in tm:
        try:
            # Import lazily to avoid circular imports at module level
            from Kosmulator_main import Statistical_packages as SP

            SP._compute_cls_cached(tm, Lmax=8, mode="lowl")
            if "H_0" in tm and all(
                k in tm for k in ("Omega_bh^2", "Omega_dh^2")
            ):
                h = float(tm["H_0"]) / 100.0
                if h > 0:
                    Om = (
                        float(tm["Omega_bh^2"])
                        + float(tm["Omega_dh^2"])
                    ) / (h * h)
                    tm.setdefault("Omega_m", Om)
        except Exception:
            # If CLASS/clik are unavailable or this fails, just skip this refinement
            pass

    return tm


def _Ez(MODEL_func, z, param_dict):
    param = _inject_derived_background(param_dict)
    return MODEL_func(z, param)


def Comoving_distance_vectorized(MODEL_func, redshifts, param_dict):
    """
    D_c(z) = (c/H0) * ∫_0^z dz'/E(z').
    """
    zs = np.atleast_1d(redshifts).astype(float)

    # Empty-input guard
    if zs.size == 0:
        return zs

    # Non-finite redshifts: return NaN (same behaviour as a failed E(z))
    if not np.isfinite(zs).all():
        return np.full_like(zs, np.nan, dtype=float)

    idx = np.argsort(zs)
    z_sorted = zs[idx]

    # Integration grid: a uniform grid (dz <= 0.01) merged with the data
    # redshifts and z = 0. Integrating on the data redshifts alone is too
    # coarse for sparse sets (DESI: D_M off by ~0.6% at z = 2.33); this grid
    # gives a relative error of ~2e-6. Intended for late-time redshifts;
    # for z ~ 1100 use CLASS rather than this routine.
    z_lo, z_hi = min(0.0, z_sorted[0]), max(0.0, z_sorted[-1])
    if z_hi == z_lo:                       # all redshifts are zero
        return np.zeros_like(zs)
    n_fine = int(np.ceil((z_hi - z_lo) / 0.01)) + 1
    grid = np.union1d(np.linspace(z_lo, z_hi, n_fine), np.append(z_sorted, 0.0))

    Ez = np.asarray(_Ez(MODEL_func, grid, param_dict), dtype=float)
    if (not np.isfinite(Ez).all()) or np.any(Ez <= 0):
        out = np.full_like(z_sorted, np.nan, dtype=float)
        d_c = np.empty_like(out)
        d_c[idx] = out
        return d_c

    invEz = 1.0 / Ez
    I_grid = cumtrapz(invEz, grid, initial=0.0)
    I_grid -= I_grid[np.searchsorted(grid, 0.0)]          # measure from z = 0
    integral = I_grid[np.searchsorted(grid, z_sorted)]    # duplicates map to the same node
    d_c = np.empty_like(integral)
    d_c[idx] = integral
    param = _inject_derived_background(param_dict)
    return d_c * (C_KM_S / float(param["H_0"]))


def sn_luminosity_distance(d_c, z, z_hel=None):
    """
    Luminosity distance for a supernova sample from its comoving distance.

    D_L = (1 + z_fac) * D_c(z)

    z      : redshift used inside the comoving-distance integral
             (for DES-SN5YR this is the Hubble-diagram redshift, zHD).
    z_hel  : optional heliocentric redshift. If given, it replaces z in the
             (1 + z) prefactor, giving D_L = (1 + z_HEL) * D_c(z_HD), the
             convention used by the DES-SN5YR/Pantheon+ likelihoods (e.g. the
             Cobaya implementation). If None, z is used in both places, which
             is the original single-redshift behaviour.

    Datasets that do not carry a `z_hel` array (Union3, JLA, Pantheon,
    Pantheon+) therefore give bit-for-bit the same result as before.
    """
    z_fac = z if z_hel is None else z_hel
    return d_c * (1.0 + z_fac)


def matter_density_z_array(zs, param_dict, MODEL_func):
    """
    Ω_m(z) = Ω_m0 (1+z)^3 / E(z)^2
    """
    Ez = _Ez(MODEL_func, zs, param_dict)
    Ez2 = Ez**2
    param = _inject_derived_background(param_dict)
    return float(param["Omega_m"]) * (1.0 + zs) ** 3 / Ez2


def fs8_ap_factor(zs, param_dict, MODEL_func, om_fid):
    """
    q(z) = H(z) d_A(z) / [H_fid(z) d_A_fid(z)] for each f sigma_8 point, the fiducial
    being flat LCDM with Omega_m = om_fid (one value per point) and the model's
    radiation, so q = 1 for LCDM at the fiducial Omega_m. H_0 and (1 + z) cancel,
    so q = E(z) chi(z) / [E_fid(z) chi_fid(z)] with chi = int_0^z dz'/E.
    """
    import User_defined_modules as _UDM      # radiation_density (UDM imports this module)
    zs = np.atleast_1d(zs).astype(float)
    om_fid = np.broadcast_to(np.asarray(om_fid, dtype=float), zs.shape)
    zg = np.linspace(0.0, float(zs.max()), max(800, int(600 * float(zs.max()))))
    Eg = np.asarray(_Ez(MODEL_func, zg, param_dict), dtype=float)
    if (not np.isfinite(Eg).all()) or np.any(Eg <= 0):
        return np.full_like(zs, np.nan)
    EX = np.interp(zs, zg, Eg) * np.interp(zs, zg, cumtrapz(1.0 / Eg, zg, initial=0.0))
    q = np.ones_like(zs)                     # q -> 1 as z -> 0 (a z = 0 row would give 0/0)
    o_r = float(_UDM.radiation_density(_inject_derived_background(param_dict)))
    for om in np.unique(om_fid):
        m = (om_fid == om) & (zs > 0.0)
        Ef = np.sqrt(om * (1.0 + zg) ** 3 + o_r * (1.0 + zg) ** 4 + 1.0 - om - o_r)
        q[m] = EX[m] / (np.interp(zs[m], zg, Ef) * np.interp(zs[m], zg, cumtrapz(1.0 / Ef, zg, initial=0.0)))
    return q


def growth_prediction(obs_type, obs_data, param_dict, MODEL_func, gamma=None):
    """
    Prediction for a growth dataset, used by the sampler, the statistics and WAIC:
    f = Omega_m(z)^gamma, or f sigma_8 = sigma_8 Omega_m(z)^gamma exp(-int_0^z
    Omega_m^gamma/(1+z') dz'), divided by the AP-type factor when the dataset asks
    for it (constants.FS8_AP_CORRECTION). NaN entries mean an invalid background.
    """
    z = np.asarray(obs_data["redshift"], dtype=float)
    Omz = matter_density_z_array(z, param_dict, MODEL_func)
    if (not np.isfinite(Omz).all()) or np.any(Omz <= 0):
        return np.full_like(z, np.nan)
    if obs_type == "f":
        return Omz ** float(param_dict["gamma"])
    gamma = float(param_dict["gamma"]) if gamma is None else float(gamma)
    model = float(param_dict["sigma_8"]) * Omz ** gamma * np.exp(-integral_term_array(z, param_dict, MODEL_func, gamma))
    if obs_data.get("ap_correction"):
        model = model / fs8_ap_factor(z, param_dict, MODEL_func, obs_data["omega_m_fid"])
    return model


def integral_term_array(zs, param_dict, MODEL_func, gamma):
    """
    ∫^z [(Ω_m(z')^γ)/(1+z')] dz' — vectorised in z (uses monotonic grid).
    """
    zs_arr = np.atleast_1d(zs).astype(float)
    if zs_arr.size == 0:
        return zs_arr

    zmax = float(np.max(zs_arr))
    if zmax <= 0:
        return np.zeros_like(zs_arr)

    N = max(800, int(600 * zmax))
    a_min = 1.0 / (1.0 + zmax)
    a_grid = np.geomspace(a_min, 1.0, N)  # inc. in a
    z_grid = (1.0 / a_grid) - 1.0         # dec. in z

    z_inc = z_grid[::-1].copy()           # 0 → zmax
    Om_inc = matter_density_z_array(z_inc, param_dict, MODEL_func)
    if (not np.isfinite(Om_inc).all()) or np.any(Om_inc <= 0):
        return np.full_like(zs_arr, np.nan, dtype=float)

    f_inc = (Om_inc**gamma) / (1.0 + z_inc)
    cumint = np.concatenate(([0.0], cumtrapz(f_inc, z_inc)))
    out = np.interp(zs_arr, z_inc, cumint)
    return float(out) if out.size == 1 else out


# ---------------------------------------------------------------------
# 14) Shared low-level helpers (CLASS/clik C-level noise control, etc.)
# ---------------------------------------------------------------------

@contextmanager
def quiet_cstdio():
    """
    Temporarily silence all C-level stdout/stderr (used around clik/CLASS calls).

    This redirects file descriptors 1 and 2 to /dev/null inside the context and
    restores them afterwards. Safe to nest, but don't use across MPI barriers.
    """
    dn = os.open(os.devnull, os.O_WRONLY)
    so, se = os.dup(1), os.dup(2)
    try:
        os.dup2(dn, 1)
        os.dup2(dn, 2)
        yield
    finally:
        os.dup2(so, 1)
        os.dup2(se, 2)
        os.close(dn)
        os.close(so)
        os.close(se)


def fast_path_for_clik(path: str) -> str:
    """
    Copy a .clik directory to a fast filesystem (/dev/shm if available, else /tmp)
    and return that path. If `path` is an HDF5 file or copying fails, return the
    original path unchanged.

    This is used to speed up Planck likelihood reads on HPC systems.
    """
    src = Path(path)
    try:
        # If it doesn't exist or is a file (e.g. .hdf5), just return as is.
        if not src.exists() or src.is_file():
            return str(src)

        # Prefer RAM disk if present
        tmp_root = Path("/dev/shm") if Path("/dev/shm").exists() else Path(
            tempfile.gettempdir()
        )
        dst = tmp_root / src.name

        if not dst.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)

        return str(dst)
    except Exception:
        # Fallback: don't crash, just use the original path
        return str(src)
