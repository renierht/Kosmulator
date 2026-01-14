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
    "CMB_lensing",
    "CMB_lowl",
    "DESI_DR1",
    "DESI_DR2",
    "f",
    "f_sigma_8",
    "JLA",
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
        dest="consecutive_required", type=int, default=3,
        help=(
            "For Zeus early-stop: require this many consecutive callback checks "
            "with |Δτ|/τ < target. Default 3."
        ),
    )
    parser.add_argument(
        "--autocorr-buffer", type=int, default=None,
        help=(
            "Extra iterations AFTER burn before convergence checks start. "
            "Default: max(1000, burn/5)."
        ),
    )
    parser.add_argument(
        "--autocorr-check-every", type=int, default=100,
        help="Check autocorrelation every N iterations (default 100)",
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
        parts.append("pool")
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

    # Forward initializer/initargs/maxtasksperchild/etc.
    return ctx.Pool(processes=num_cores, **pool_kwargs)


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

def build_plot_settings(
    observations,
    suffix: str,
    latex_enabled: bool,
    plot_table: bool,
) -> Dict[str, Any]:
    import matplotlib.pyplot as plt  # noqa: F401

    colors = DEFAULT_PLOT_COLORS
    base_plots = DEFAULT_PLOTS_BASE
    plot_table = bool(plot_table)

    n_obs = len(observations)

    # Make sure we have at least one colour per observation group.
    # If there are more obs than base colours, cycle through the list.
    if n_obs <= len(colors):
        color_schemes = list(colors[:n_obs])
    else:
        # tile and trim
        repeats = (n_obs + len(colors) - 1) // len(colors)
        color_schemes = (colors * repeats)[:n_obs]

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
        "autocorr_buffer_after_burn": 1000,
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
                if obs == "BAO":
                    z_vals = np.array(
                        [
                            0.295, 0.510, 0.510, 0.706, 0.706,
                            0.930, 0.930, 1.317, 1.317, 1.491,
                            2.330, 2.330,
                        ],
                        dtype=float,
                    )
                    tested_obs = obs
                    break
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
            tv = config[mod]["true_values"][0]
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
        with h5py.File(zeus_chain, "r") as f:
            all_samples = f["samples"][:]    # (nsteps, nwalker, ndim)

        burn = int(CONFIG_model.get("burn", 0) or 0)
        # Guard against silly values
        if burn >= all_samples.shape[0]:
            burn = 0

        # Flatten to (n_samples, ndim) as usual
        return all_samples[burn:, :, :].reshape(-1, all_samples.shape[-1])
        
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
            return Kosmulator_MCMC.run_mcmc(
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
    flat_samples = Kosmulator_MCMC.run_mcmc(
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

    if (
        flat_samples is None
        or flat_samples.size == 0
        or not np.any(np.isfinite(flat_samples))
    ):
        print(f"[WARNING] Samples for {chain_file} are empty or invalid!")
    return flat_samples


# ───────────────────────────────────────────────────────────────────────────────
# 8) emcee helpers (autocorr stopping)
# ───────────────────────────────────────────────────────────────────────────────

def emcee_autocorr_stopping(
    pos: np.ndarray,
    sampler: "emcee.EnsembleSampler",
    nsteps: int,
    model_name: str,
    colors: list,
    obs: list,
    PLOT_SETTINGS: dict,
    convergence: float = 0.01,
    last_obs: bool = False,
    resume_offset: int = 0,
    local_burn: Optional[int] = None,
    global_burn: Optional[int] = None,
    buffer_after_burn: Optional[int] = None,
    # ---------------- NEW ----------------
    print_enabled: bool = False,
    print_every: int = 0,
) -> np.ndarray:
    """
    emcee with periodic autocorr checks; only start checking once
    global_iter >= global_burn + buffer_after_burn.

    - resume_offset: steps already completed BEFORE this call
    - local_burn:    how many steps in THIS call to still consider burn
    - global_burn:   the absolute burn target for the overall chain
    - buffer_after_burn: extra margin after burn before checking (default 100)

    NEW:
    - print_enabled / print_every:
        Emit "[EMCEE Step N] Log-Post: Max=... | Mean=..." from the MAIN PROCESS
        so it works even when emcee uses a multiprocessing Pool.
    """
    import matplotlib.pyplot as plt  # noqa: F401
    import Plots.Plots as MP

    if h5py is None:
        raise RuntimeError(
            "h5py required for autocorr plotting/checkpointing."
        )

    if local_burn is None:
        local_burn = global_burn or 0
    if global_burn is None:
        global_burn = local_burn

    check_every = int(PLOT_SETTINGS.get("autocorr_check_every", 100))
    buffer_after_burn = int(
        buffer_after_burn
        if buffer_after_burn is not None
        else PLOT_SETTINGS.get("autocorr_buffer_after_burn", 1000)
    )
    min_check_global = int(global_burn + buffer_after_burn)

    # ---------------- NEW ----------------
    # Normalize/validate print cadence
    try:
        print_every = int(print_every or 0)
    except Exception:
        print_every = 0
    if print_every < 0:
        print_every = 0
    # ------------------------------------

    max_checks = int(np.ceil((resume_offset + nsteps) / max(check_every, 1))) + 2
    autocorr = np.empty(max_checks)
    index = 0
    old_tau = np.inf

    obs_label = generate_label(obs).replace("+", "_")
    folder = os.path.join(
        PLOT_SETTINGS["autocorr_save_path"], model_name, "auto_corr"
    )
    os.makedirs(folder, exist_ok=True)
    plot_path = os.path.join(folder, f"{obs_label}.png")
    tau_target = convergence

    # Drive step-by-step to control the check cadence
    for sample in sampler.sample(pos, iterations=nsteps, progress=True):
        it_local = sampler.iteration            # steps in *this* call so far
        it_global = resume_offset + it_local    # absolute progress

        # Pool-safe EMCEE logging (main process only)
        if (
            print_enabled
            and (print_every > 0)
            and (it_global % print_every) == 0
            and is_rank0()
            and is_main_process()
        ):
            lp = getattr(sample, "log_prob", None)

            # Fallback for emcee state variants
            if lp is None:
                try:
                    lp_all = sampler.get_log_prob()
                    lp = lp_all[-1] if getattr(lp_all, "ndim", 0) > 1 else lp_all
                except Exception:
                    lp = None

            if lp is not None:
                lp = np.asarray(lp, dtype=float)
                if lp.size:
                    lp_max = float(np.nanmax(lp))
                    lp_mean = float(np.nanmean(lp))
                    print(
                        f"[EMCEE Step {it_global}] Log-Post: Max={lp_max:.4f} | Mean={lp_mean:.4f}",
                        flush=True,
                    )

        # only do work on our check cadence
        if (it_global % check_every) != 0:
            continue

        # Try to get τ̂; early on this can fail or be noisy
        tau = None
        try:
            tau = sampler.get_autocorr_time(tol=0, quiet=True)
            tau_mean = float(np.mean(tau))
        except Exception:
            tau_mean = np.nan

        # Record for the live plot (even before burn), then draw
        autocorr[index] = tau_mean
        index += 1

        MP.autocorrPlot(
            autocorr,
            index,
            model_name,
            colors[0] if isinstance(colors, (list, tuple)) and colors else "C0",
            obs,
            PLOT_SETTINGS,
            plot_path=plot_path,
            close_plot=False,
            resume_offset=resume_offset,
            check_every=check_every,
            global_burn=global_burn,
            nsteps=nsteps,
            convergence=convergence,
        )

        # Only attempt a STOP decision after burn + buffer, and only with finite τ̂
        if (
            (it_global >= min_check_global)
            and (tau is not None)
            and np.all(np.isfinite(tau))
        ):
            # Use a *global* effective iteration count since the start of sampling
            effective_iter = max(0, it_global - global_burn)

            # Same criterion you used, but vs global steps
            converged = np.all(tau * check_every < effective_iter)

            # Stable τ̂ (relative change tolerance)
            if np.all(np.isfinite(old_tau)):
                stable = np.all(
                    np.abs(old_tau - tau)
                    / np.maximum(tau, 1e-12)
                    < tau_target
                )
            else:
                stable = False

            if converged and stable:
                MP.autocorrPlot(
                    autocorr,
                    index,
                    model_name,
                    colors,
                    obs,
                    PLOT_SETTINGS,
                    plot_path=plot_path,
                    close_plot=True,
                    resume_offset=resume_offset,
                    check_every=check_every,
                    global_burn=global_burn,
                    nsteps=nsteps,
                    convergence=convergence,
                )
                break

        # Update old_tau only if we got a valid estimate
        if (tau is not None) and np.all(np.isfinite(tau)):
            old_tau = tau
    else:
        # Finished all iterations without “break”: close the plot (or leave open if more obs)
        MP.autocorrPlot(
            autocorr,
            index,
            model_name,
            colors,
            obs,
            PLOT_SETTINGS,
            plot_path=plot_path,
            close_plot=last_obs,
            resume_offset=resume_offset,
            check_every=check_every,
            global_burn=global_burn,
            nsteps=nsteps,
            convergence=convergence,
        )

    backend = sampler.backend
    return backend.get_chain(discard=global_burn, flat=True)



# ───────────────────────────────────────────────────────────────────────────────
# 9) Zeus callbacks / monitors
# ───────────────────────────────────────────────────────────────────────────────

def make_zeus_callbacks(
    burn: int,
    nsteps: int,
    target_autocorr: float = 0.01,
    plot_func=None,
    debug: bool = False,
    precision_switch_iter: Optional[int] = None,
    coarse_kwargs: Optional[dict] = None,
    fine_kwargs: Optional[dict] = None,
    ncheck: Optional[int] = None,
    append_writer: Optional[callable] = None,
    consecutive_required: int = 1,
):
    """
    Composite Zeus callback:
      • updates τ every ncheck via AutocorrelationCallback
      • plots (optional) and appends (optional)
      • optional precision switch at precision_switch_iter
      • early-stop AFTER local 'burn' when fractional change |Δτ|/τ < target_autocorr
        for `consecutive_required` consecutive checks
    """
    try:
        from zeus.callbacks import AutocorrelationCallback
    except Exception:
        logging.getLogger(__name__).warning(
            "Zeus not installed: no callbacks will be created."
        )
        return []

    import numpy as _np
    from collections import deque

    ncheck = int(ncheck) if ncheck is not None else 100
    discard_frac = max(0.1, min(0.7, float(max(1, burn)) / float(max(nsteps, 1))))
    consecutive_required = max(1, int(consecutive_required))

    class CompositeAutoCorr(AutocorrelationCallback):
        def __init__(self):
            super().__init__(ncheck=ncheck, dact=1e9, nact=1, discard=discard_frac)
            self._prev_tau = None
            self._fracs = deque(maxlen=8)   # recent fractional |Δτ|/τ
            self._taus = deque(maxlen=8)    # optional τ history (debug)
            self._switched = False

        def __call__(self, iteration, chain, log_prob):
            # Update τ̂ estimates every ncheck via the parent
            super().__call__(iteration, chain, log_prob)

            # Optional precision switch (one-time)
            if (
                precision_switch_iter is not None
                and (not self._switched)
                and iteration >= int(precision_switch_iter)
            ):
                try:
                    from Kosmulator_main import Statistical_packages as SP
                    SP.set_precision("fine", **(fine_kwargs or {}))
                    self._switched = True
                    if debug:
                        print(f"[Zeus] precision -> fine at iter {iteration}")
                except Exception as e:
                    if debug:
                        print(f"[Zeus] precision switch failed: {e}")

            # --- NEW PRINT LOGIC FOR ZEUS ---
            if K.print_loglike and (iteration % K.print_loglike_every == 0):
                # log_prob is the Posterior (shape: nwalkers,)
                valid_lp = log_prob[_np.isfinite(log_prob)]
                if valid_lp.size > 0:
                    max_p = _np.max(valid_lp)
                    mean_p = _np.mean(valid_lp)
                    # We print "Log-Post" because Zeus passes Posterior, not just Likelihood
                    print(f"[Zeus Step {iteration}] Log-Post: Max={max_p:.4f} | Mean={mean_p:.4f}", flush=True)
            # -------------------------------

            # Only act on callback ticks for autocorr/plotting
            if iteration % ncheck != 0:
                return False

            ests = getattr(self, "estimates", None)
            if ests is None:
                return False

            # Current τ̂ (scalar summary from zeus’ estimates)
            tau = float(_np.asarray(ests, dtype=float)[-1])
            stopped = False

            # Compute fractional change and record
            if self._prev_tau is not None:
                delta = abs(tau - self._prev_tau)
                frac = delta / max(tau, 1e-12)
                self._fracs.append(frac)
                self._taus.append(tau)

                # Plot (pass τ̂; plotter can compute its own metric)
                if callable(plot_func):
                    try:
                        plot_func(ests, iteration)
                    except Exception:
                        if debug:
                            print("[Zeus] plot_func raised; ignored.")

                # Optional incremental writer
                if callable(append_writer):
                    try:
                        append_writer(iteration, chain, log_prob)
                    except Exception:
                        if debug:
                            print("[Zeus] append_writer raised; ignored.")

                # ---- Early-stop rule (LOCAL burn gate) ----
                if iteration >= int(burn) and len(self._fracs) >= consecutive_required:
                    window = list(self._fracs)[-consecutive_required:]
                    if all(f < float(target_autocorr) for f in window):
                        if debug:
                            print(
                                f"[Zeus] Early stop at iter={iteration}: "
                                f"(Δτ/τ)={window[-1]:.3g} < target={target_autocorr} "
                                f"(for {consecutive_required} consecutive checks)"
                            )
                        stopped = True

            self._prev_tau = tau
            return stopped

    return [CompositeAutoCorr()]


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
        f"{'AIC':>11} | {'BIC':>11} | {'dAIC':>11} | {'dBIC':>11}"
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
            daic = _as_float(s.get("dAIC", _np.nan))
            dbic = _as_float(s.get("dBIC", _np.nan))
            row = (
                f"{obs:<{obs_w}} | {ll:>18.4f} | {chi2:>15.4f} | "
                f"{rchi:>20.4f} | {aic:>11.3f} | {bic:>11.3f} | "
                f"{daic:>11.3f} | {dbic:>11.3f}"
            )
            f.write(row + "\n")
        f.write("\n")


def save_interpretations_to_file(
    model: str,
    folder: str,
    interpretations_list: List[Dict[str, str]],
) -> None:
    file_path = os.path.join(folder, "interpretations_summary.txt")
    obs_w, diag_w, aic_w, bic_w = 30, 50, 35, 35

    with open(file_path, "w") as f:
        f.write(f"Interpretations for Model: {model}\n\n")
        header = (
            f"{'Observation':<{obs_w}} | "
            f"{'Reduced Chi2 Diagnostics':<{diag_w}} | "
            f"{'AIC Interpretation':<{aic_w}} | "
            f"{'BIC Interpretation':<{bic_w}}"
        )
        f.write(header + "\n")
        total = obs_w + diag_w + aic_w + bic_w + 9
        f.write("-" * total + "\n")

        for it in interpretations_list:
            obs = it["Observation"]
            diag = it["Reduced Chi2 Diagnostics"]
            aic_i = it["AIC Interpretation"]
            bic_i = it["BIC Interpretation"]

            diag_lines = textwrap.wrap(diag, width=diag_w)
            aic_lines = textwrap.wrap(aic_i, width=aic_w)
            bic_lines = textwrap.wrap(bic_i, width=bic_w)
            obs_line = obs.ljust(obs_w)
            max_lines = max(1, len(diag_lines), len(aic_lines), len(bic_lines))

            for i in range(max_lines):
                line = (
                    f"{(obs_line if i == 0 else ' ' * obs_w):<{obs_w}} | "
                    f"{(diag_lines[i] if i < len(diag_lines) else ''):<{diag_w}} | "
                    f"{(aic_lines[i] if i < len(aic_lines) else ''):<{aic_w}} | "
                    f"{(bic_lines[i] if i < len(bic_lines) else ''):<{bic_w}}"
                )
                f.write(line + "\n")


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
    Zeus callback that appends only new local samples to 'samples' dataset.
    """
    def __init__(self, filename: str, ncheck: int):
        self.filename = filename
        self.ncheck = int(ncheck)
        self._prev_local = 0

    def __call__(self, iteration: int, chain: np.ndarray, log_prob: np.ndarray):
        if h5py is None:
            return False
        if iteration % self.ncheck != 0:
            return False

        local_n, nwalker, ndim = chain.shape
        new_local = local_n - self._prev_local
        if new_local <= 0:
            return False

        new_block = chain[self._prev_local:local_n]
        with h5py.File(self.filename, "a") as f:
            if "samples" not in f:
                ds = f.create_dataset(
                    "samples",
                    data=chain,
                    maxshape=(None, nwalker, ndim),
                    chunks=(1, nwalker, ndim),
                )
            else:
                ds = f["samples"]
                old_n = ds.shape[0]
                new_n = old_n + new_local
                ds.resize((new_n, nwalker, ndim))
                ds[old_n:new_n, :, :] = new_block

        self._prev_local = local_n
        return False


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
    true_values: Dict[str, float],
    prior_limits: Dict[str, Tuple[float, float]],
    names: List[str],
) -> None:
    """
    Ensure priors/initials exist for all requested Planck nuisance names.
    """
    for n in names:
        if n in prior_limits and n in true_values:
            continue
        default = PLANCK_NUISANCE_DEFAULTS.get(n)
        if default is not None:
            tv, (lo, hi) = default
            true_values.setdefault(n, tv)
            prior_limits.setdefault(n, (lo, hi))
        else:
            # Fallback heuristic if a name isn't in our table
            true_values.setdefault(n, 0.0)
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
            if (rraw == "jla") or ("jla" in rlow):
                if token_mode:
                    _warn_once(f"JLA_solo::{resolved}", "JLA run solo")
                else:
                    _warn_once(
                        f"JLA_solo::{resolved}",
                        (
                            "JLA (SNe) run solo: these SNe have M_abs values "
                            "calibrated to Cepheids and are treated as fixed "
                            "per object. Running solo is acceptable, but "
                            "pairing with other observations (e.g., CC) is "
                            "recommended."
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
            if (rraw == "cmb_lensing") or ("cmb_lensing" in rlow):
                msg = (
                    "CMB_lensing run solo: using RAW (non-marginalised) Planck "
                    "lensing likelihood. The CMB-marginalised lensing mode is "
                    "not yet wired into Kosmulator and will be added in a future "
                    "update. You can safely combine CMB_lensing with CMB_lowl, "
                    "CMB_hil, or CMB_hil_TT; those combinations use the standard "
                    "RAW lensing treatment as in Planck TT/TE/EE+lensing."
                )
                _warn_once(f"CMBLensing_solo::{resolved}", msg)
                continue
                
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

    idx = np.argsort(zs)
    z_sorted = zs[idx]
    grid = np.concatenate(([0.0], z_sorted))

    Ez = _Ez(MODEL_func, grid, param_dict)
    if (not np.isfinite(Ez).all()) or np.any(Ez <= 0):
        out = np.full_like(z_sorted, np.nan, dtype=float)
        d_c = np.empty_like(out)
        d_c[idx] = out
        return d_c

    invEz = 1.0 / Ez
    integral = cumtrapz(invEz, grid, initial=0.0)[1:]  # len == len(z_sorted)
    d_c = np.empty_like(integral)
    d_c[idx] = integral
    param = _inject_derived_background(param_dict)
    return d_c * (C_KM_S / float(param["H_0"]))


def matter_density_z_array(zs, param_dict, MODEL_func):
    """
    Ω_m(z) = Ω_m0 (1+z)^3 / E(z)^2
    """
    Ez = _Ez(MODEL_func, zs, param_dict)
    Ez2 = Ez**2
    param = _inject_derived_background(param_dict)
    return float(param["Omega_m"]) * (1.0 + zs) ** 3 / Ez2


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
