"""Plot_functions.py — central plotting helpers for Kosmulator

This module contains helpers only (no figure construction). Plot-producing
code lives in `Plots.py`. Keeping responsibilities separate makes both files
simpler and safer to change.

Exports:
  • Console/UI helpers: phase_banner, section_banner, print_rule
  • Label helpers: greek_Symbols, format_for_latex, texify_label, pretty_obs_name
  • Table helpers: align_table_to_parameters, add_corner_table,
                   print_aligned_latex_table, print_stats_table
  • Data helpers: extract_observation_data, fetch_best_fit_values,
                  partition_by_compatibility, residual_unit,
                  OBS_COLOR_ORDER, MODEL_COLOR
  • Model evaluators: compute_E, compute_Dc, compute_DM, compute_DV,
                      compute_rd, compute_f, compute_sigma8z
  • Plot evaluators: model_curve_for_type, evaluator_for_points
  • CMB helpers: read_bandpowers, cmb_lowl_curve
  • Path helpers: normalize_save_roots, base_dir
  • IO: save_figure

Directory layout (unified across all savers):
  ./Plots/Saved_Plots/<output_suffix>/<model>/{auto_corr, corner_plots, best_fits}/...
"""
from __future__ import annotations
import logging, time as _time
logger = logging.getLogger(__name__)

# stdlib
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# third-party
import numpy as np
import math

# project-local
import User_defined_modules as UDM  # user's cosmology/model functions
from Kosmulator_main.constants import (
    GREEK_SYMBOLS, COMBINE_GROUPS, OBS_COLOR_ORDER, MODEL_COLOR, R_D_SINGLETON, DEFAULT_PLOTS_BASE, GAMMA_FS8_SINGLETON,
    OBS_PRETTY_MAP, TABLE_ANCHORS_OBS, TABLE_ANCHORS_PARM, C_KM_S,T_CMB_DEFAULT, N_EFF_DEFAULT, TAU_N_DEFAULT, DEFAULT_CMB_FILES
)
from Kosmulator_main.rd_helpers import compute_rd as compute_rd
#from Kosmulator_main.Kosmulator_MCMC import _inject_derived_background
__all__ = [
    # console
    "phase_banner", "section_banner", "print_rule",
    # labels
    "greek_Symbols", "format_for_latex", "texify_label", "pretty_obs_name",
    # tables
    "align_table_to_parameters", "add_corner_table",
    "print_aligned_latex_table", "print_parameter_list_table", "print_cmb_summary_matrix", "print_stats_table",
    # data helpers
    "extract_observation_data", "fetch_best_fit_values",
    "partition_by_compatibility", "residual_unit",
    "OBS_COLOR_ORDER", "MODEL_COLOR",
    # model evaluators
    "compute_E", "compute_Dc", "compute_DM", "compute_DV",
    "compute_rd", "compute_f", "compute_sigma8z",
    # plot evaluators
    "model_curve_for_type", "evaluator_for_points",
    # paths / io
    "normalize_save_roots", "base_dir", "save_figure",
    # cmb
    "read_bandpowers", "cmb_lowl_curve",
]

# =============================================================================
# Console helpers (match utils.py banner aesthetic)
# =============================================================================

_BAR = "#" * 48
_DASH = "-" * 47
_RD_WARN_LAST = 0.0
DEFAULT_GROWTH_INDEX = GAMMA_FS8_SINGLETON


def phase_banner(title: str) -> None:
    print(f"\n\033[33m{_BAR}\033[0m", flush=True)
    print(f"\033[33m####\033[0m {title}", flush=True)
    print(f"\033[33m{_BAR}\033[0m", flush=True)


def section_banner(title: str) -> None:
    print(f"{_DASH}\n {title}\n{_DASH}", flush=True)


def print_rule() -> None:
    print(_DASH, flush=True)


# =============================================================================
# Label helpers
# =============================================================================
PARAM_LATEX_OVERRIDES = {
    # Core cosmological / CMB parameters with non-trivial syntax
    "Omega_bh^2":  r"\Omega_b h^2",
    "Omega_dh^2":  r"\Omega_d h^2",
    "ln10^10_As":  r"\ln 10^{10} A_s",
    "tau_reio":    r"\tau_\mathrm{reio}",
    "A_planck":    r"A_\mathrm{Planck}",
    # You can extend this dict with any other special cases you care about.
    # e.g.
    # "A_cib_217":   r"A_\mathrm{CIB}^{217}",
    # "A_sz":        r"A_\mathrm{SZ}",
}

def greek_Symbols(parameters: Sequence[str] | Sequence[Sequence[str]] | None = None):
    """
    Map parameter names to TeX, handling subscripts.

    Rules:
      * Simple names like ``Omega_m`` or ``H_0`` become ``\\Omega_{m}``, ``H_{0}``.
      * Known special cases (Planck nuisance etc.) are taken from
        ``PARAM_LATEX_OVERRIDES``.
      * Names with more than one ``_`` (e.g. ``ps_A_100_100``) are treated as
        plain text and wrapped in ``\\mathrm{...}`` so we don't guess a
        subscript structure and lose information.
    """
    def fmt_one(name: str) -> str:
        # 1) Exact override first
        if name in PARAM_LATEX_OVERRIDES:
            return PARAM_LATEX_OVERRIDES[name]

        # 2) No underscore → try full-name Greek replacement, otherwise leave
        if "_" not in name:
            return GREEK_SYMBOLS.get(name, name)

        # 3) Very complex names (multiple underscores) → keep as roman text
        #    e.g. ps_A_100_100, A_sbpx_100_100_TT, ...
        if name.count("_") > 1:
            safe = name.replace("_", r"\_")
            return rf"\mathrm{{{safe}}}"

        # 4) Simple base_sub pattern (Omega_m, H_0, etc.)
        base, sub = name.split("_", 1)
        base_label = GREEK_SYMBOLS.get(base, base)
        sub_label  = GREEK_SYMBOLS.get(sub, sub)
        return rf"{base_label}_{{{sub_label}}}"

    if not parameters:
        return []
    if isinstance(parameters[0], (list, tuple)):
        # list-of-lists case (one row per observation)
        return [[fmt_one(p) for p in row] for row in parameters]  # type: ignore[index]
    # flat list
    return [fmt_one(p) for p in parameters]  # type: ignore[arg-type]


def format_for_latex(items: Sequence[str]) -> List[str]:
    """Wrap strings with ``$...$`` for LaTeX rendering (strip any existing math $)."""
    out = []
    for s in items:
        s = str(s).strip()
        # remove leading/trailing unescaped $ to avoid $$...$$
        while s.startswith("$") and not s.startswith("\\$"):
            s = s[1:]
        while s.endswith("$") and not s.endswith("\\$"):
            s = s[:-1]
        out.append(f"${s}$")
    return out


def texify_label(text: Optional[str], PLOT_SETTINGS: Mapping[str, Any]) -> Optional[str]:
    """Convert friendly text/Unicode to LaTeX when enabled via ``PLOT_SETTINGS``."""
    if text is None:
        return None
    latex_on = bool(PLOT_SETTINGS.get("latex_enabled", False))
    out = str(text)

    # Script ell → \ell
    if "ℓ" in out or "ell" in out:
        if latex_on:
            out = out.replace("ℓ", r"$\ell$").replace("low-ell", r"low-$\ell$").replace("high-ell", r"high-$\ell$")
        else:
            out = out.replace("ℓ", "l").replace("low-ell", "low-l").replace("high-ell", "high-l")

    if latex_on:
        for key, cmd in GREEK_SYMBOLS.items():
            out = out.replace(key, rf"${cmd}$")
    else:
        for key, cmd in GREEK_SYMBOLS.items():
            out = out.replace(cmd, key)

    return out


def pretty_obs_name(obs_key: str, latex_on: bool = True) -> str:
    parts = obs_key.split("+")
    def one(p: str) -> str:
        pair = OBS_PRETTY_MAP.get(p)
        if pair is None: return p
        return pair[0] if latex_on else pair[1]
    return " + ".join(one(p) for p in parts)


# =============================================================================
# Table helpers (autoscaling + top-band layout)
# =============================================================================

def _interp_piecewise(x, xs, ys):
    """Linear interpolate across sorted xs with values ys; clamp outside."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] * (1 - t) + ys[i + 1] * t
    return ys[-1]


def autoscale_table_settings(n_params: int, n_obs: int, position: str = "top", overrides: dict | None = None) -> dict:
    """
    Compute PLOT_SETTINGS for the corner table by interpolating between
    your hand-tuned anchor points.
    """
    # Vertical scaling vs number of observation combinations (6 parameters)
    OBS_X = [2, 5, 8]
    OBS_ANCH = {
        "corner_top":           [0.94, 0.90, 0.86],
        "table_pad":            [0.11, 0.12, 0.23],
        "min_sidebar_width":    [0.30, 0.20, 0.16],
        "table_width_per_col":  [0.04, 0.04, 0.03],
        "table_height_per_row": [0.95, 0.55, 0.18],
        "cell_height_factor":   [8.5,  5.5,  2.8],
    }

    # n_obs == 1 special: anchors vs number of params (4,3,2)
    PARM_X = [2, 3, 4]
    PARM_ANCH = {
        "corner_top":          [0.90, 0.93, 0.94],
        "table_width_per_col": [0.06, 0.04, 0.04],
        "table_height_per_row":[0.95, 0.95, 0.95],
        "cell_height_factor":  [8.5,  8.5,  8.5],
    }

    # Interpolate
    S = {}
    if n_obs == 1:
        for k, ys in PARM_ANCH.items():
            S[k] = _interp_piecewise(n_params, PARM_X, ys)
        for k, ys in OBS_ANCH.items():
            if k not in S:
                S[k] = ys[0]   # borrow the "2 obs" line for the rest
    else:
        x = max(min(n_obs, OBS_X[-1]), OBS_X[0])
        for k, ys in OBS_ANCH.items():
            S[k] = _interp_piecewise(x, OBS_X, ys)

    # Baselines
    S.setdefault("table_width_base", 0.030)
    S.setdefault("table_width_cap",  0.22)
    S.setdefault("table_height_base", 0.012)
    S.setdefault("table_fill_x", 0.98)
    S.setdefault("table_fill_y", 0.55)
    S.setdefault("table_font_min", 9)
    S.setdefault("table_font_max", 14)

    # gentle n_params dependence
    S["table_width_per_col"] = float(S["table_width_per_col"] * (1.0 + 0.02 * max(0, n_params - 6)))

    # clamp
    S["corner_top"] = float(max(0.82, min(0.95, S["corner_top"])))
    S["table_height_per_row"] = float(max(0.10, min(1.20, S["table_height_per_row"])))
    S["cell_height_factor"]   = float(max(1.6,  min(9.5,  S["cell_height_factor"])))

    S["table_position"] = position  # top preferred

    if overrides:
        S.update({k: overrides[k] for k in overrides if overrides[k] is not None})
    return S


def align_table_to_parameters(
    latex_table: List[List[str]],
    parameters: List[List[str]]
) -> List[List[str]]:
    """Align each row of `latex_table` to the union of parameters (first-seen order)."""
    # union (order preserved)
    full: List[str] = []
    seen = set()
    for plist in parameters:
        for p in plist:
            if p not in seen:
                seen.add(p)
                full.append(p)
    pos = {p: i for i, p in enumerate(full)}

    # align each observation row into full width
    out: List[List[str]] = []
    for row, obs_params in zip(latex_table, parameters):
        aligned = [""] * len(full)
        for val, pname in zip(row, obs_params):
            j = pos.get(pname)
            if j is not None:
                aligned[j] = val
        out.append(aligned)
    return out


def add_corner_table(
    g,
    latex_table,
    labels,
    PLOT_SETTINGS,
    parameter_labels,
    flat_parameters,
    num_params,
):
    """
    Draw the best–fit summary table as a band above the corner plot.

    Behaviour:
      * Band height mainly depends on the number of TABLE ROWS (obs combos),
        with extra room when there is 1 row but many parameters.
      * The table hugs the top of the corner grid with a small vertical pad.
      * Font size and cell size adapt to BOTH rows and parameters.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get figure from GetDist plotter
    fig = getattr(g, "fig", None)
    if fig is None:
        fig = plt.gcf()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _renderer(f):
        try:
            return f.canvas.get_renderer()
        except Exception:
            f.canvas.draw()
            return f.canvas.get_renderer()

    # ------------------------------------------------------------------
    # Determine footprint of the main corner grid
    # ------------------------------------------------------------------
    axes = [
        ax for ax in fig.axes
        if ax.get_visible() and ax.xaxis.get_visible() and ax.yaxis.get_visible()
    ]
    if not axes:
        axes = [ax for ax in fig.axes if ax.get_visible()]
    if not axes:
        return None

    fig.canvas.draw()
    pos = [ax.get_position().bounds for ax in axes]  # (left, bottom, width, height)
    grid_left  = min(l for (l, b, w, h) in pos)
    grid_right = max(l + w for (l, b, w, h) in pos)
    grid_top   = max(b + h for (l, b, w, h) in pos)

    # ------------------------------------------------------------------
    # Row/column counts
    # ------------------------------------------------------------------
    n_rows = max(1, len(latex_table))        # table rows (obs combos)
    n_cols = max(1, len(parameter_labels))   # parameters (columns)
    n_obs  = max(1, len(labels))             # legend labels (for completeness)

    # ------------------------------------------------------------------
    # Band height: primarily rows, with extra for wide 1-row tables
    # ------------------------------------------------------------------
    base_band     = float(PLOT_SETTINGS.get("table_band_base", 0.14))
    per_row       = float(PLOT_SETTINGS.get("table_band_per_row", 0.018))
    per_col_wide  = float(PLOT_SETTINGS.get("table_band_per_col_wide", 0.012))
    max_band_frac = float(PLOT_SETTINGS.get("table_max_band_fraction", 0.28))

    # Start from row-based budget
    band_frac = base_band + per_row * (n_rows - 1)

    # Extra room if we have 1 row but many parameters (your CMB_hi case)
    if n_rows == 1 and n_cols > 6:
        band_frac += per_col_wide * (n_cols - 6)

    # Clamp to sensible range
    band_frac = max(0.12, min(max_band_frac, band_frac))

    # Base vertical padding between grid and table
    vpad = float(PLOT_SETTINGS.get("table_vpad", 0.018))
    # For very shallow tables, use a smaller gap so they sit closer
    if n_rows == 1:
        vpad = min(vpad, 0.010)

    # Push the corner grid down so `band_frac` of the figure height
    # is available above it for the table + padding.
    new_top = 1.0 - band_frac
    fig.subplots_adjust(top=new_top)
    fig.canvas.draw()

    # Recompute grid after margin change
    pos = [ax.get_position().bounds for ax in axes]
    grid_left  = min(l for (l, b, w, h) in pos)
    grid_right = max(l + w for (l, b, w, h) in pos)
    grid_top   = max(b + h for (l, b, w, h) in pos)
    headroom   = max(0.0, 1.0 - grid_top)

    # ------------------------------------------------------------------
    # Geometry of the table band
    # ------------------------------------------------------------------
    width_grid = grid_right - grid_left
    width      = 0.94 * width_grid
    height     = max(0.01, headroom - 2.0 * vpad)

    # Horizontal centre, with optional user offset
    xoffset = float(PLOT_SETTINGS.get("table_xoffset", 0.0))
    xoffset = max(-0.25, min(0.25, xoffset))
    left    = grid_left + (width_grid - width) * (0.5 + xoffset)

    # Table band sits directly above the grid
    bottom = grid_top + vpad

    # ------------------------------------------------------------------
    # Font size based on physical cell size
    # ------------------------------------------------------------------
    fmin = float(PLOT_SETTINGS.get("table_font_min", 9))
    fmax = float(PLOT_SETTINGS.get("table_font_max", 14))

    # For shallow but wide tables we allow a bit more fontsize
    if n_rows <= 2 and n_cols >= 5:
        fmax = min(fmax, 13.0)

    fig_w_in, fig_h_in = fig.get_size_inches()

    # Height constraint
    cell_h_in = max(1e-3, (height * fig_h_in) / (n_rows + 1))
    fontsize_h = 0.85 * 72.0 * cell_h_in

    # Width constraint: ~10 characters per cell
    approx_cols = float(n_cols + 1)  # +1 for Observation column
    cell_w_in   = max(1e-3, (width * fig_w_in) / approx_cols)
    fontsize_w  = 72.0 * cell_w_in / (10.0 * 0.6)

    raw_fs   = min(fontsize_h, fontsize_w)
    fontsize = float(np.clip(raw_fs, fmin, fmax))

    # Row height factor; boost for single-row, many-parameter tables
    cell_k = float(
        PLOT_SETTINGS.get(
            "table_cell_height_factor",
            PLOT_SETTINGS.get("cell_height_factor", 6.0)
        )
    )
    if n_rows == 1 and n_cols >= 5:
        # make rows noticeably taller in the 1-obs / many-param case
        cell_k *= 2.1

    # ------------------------------------------------------------------
    # Draw the table
    # ------------------------------------------------------------------
    cols_tex = (
        format_for_latex(greek_Symbols(list(parameter_labels)))
        if PLOT_SETTINGS.get("latex_enabled", False)
        else list(parameter_labels)
    )

    # Pad rows to full width
    rows = [row + [""] * (n_cols - len(row)) for row in latex_table]

    ax = fig.add_axes([left, bottom, width, height], facecolor="none", zorder=3)
    ax.axis("off")
    ax.patch.set_alpha(0)

    row_labels = list(labels)
    if len(row_labels) < n_rows:
        row_labels += [""] * (n_rows - len(row_labels))
    elif len(row_labels) > n_rows:
        row_labels = row_labels[:n_rows]

    tbl = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=cols_tex,
        cellLoc="center",
        loc="upper left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)

    # 1) scale Y to hit the desired physical cell height
    target_cell_h_in = cell_k * fontsize / 72.0

    # Allow a bit more vertical stretch for the 1-row, many-parameter case
    max_y = 1.8
    if n_rows == 1 and n_cols >= 7:
        max_y = 3.5

    yscale = float(np.clip(target_cell_h_in / cell_h_in, 0.65, max_y))
    tbl.scale(1.0, yscale)

    # 2) scale X so the table fills ~98% of the available width
    fig.canvas.draw()
    bb0 = tbl.get_window_extent(_renderer(fig)).transformed(
        fig.transFigure.inverted()
    )
    cur_w = max(bb0.width, 1e-6)
    xscale = float(np.clip((width * 0.98) / cur_w, 0.70, 2.2))
    tbl.scale(xscale, 1.0)

    # For very large tables thin the borders slightly
    if n_rows > 6 or n_cols > 6:
        for (row, col), cell in tbl.get_celld().items():
            cell.set_linewidth(0.4)

    # ------------------------------------------------------------------
    # Re-position the legend into the gap between grid and table
    # ------------------------------------------------------------------
    legend = None
    for ax0 in fig.axes:
        lg = ax0.get_legend()
        if lg is not None:
            legend = lg
            break

    if legend is not None:
        gap = max(0.0, bottom - grid_top)
        y_legend = max(
            grid_top + 0.01,
            min(bottom - 0.01, grid_top + 0.5 * gap),
        )
        x_center = 0.5 * (grid_left + grid_right)

        base_leg_fs = PLOT_SETTINGS.get("legend_font_size", fontsize)
        n_leg_labels = len(legend.get_texts())
        leg_fs = base_leg_fs
        if n_leg_labels >= 8:
            leg_fs = max(7, min(base_leg_fs, fontsize - 2))
        for txt in legend.get_texts():
            txt.set_fontsize(leg_fs)

        legend.set_bbox_to_anchor(
            (x_center, y_legend),
            transform=fig.transFigure,
            loc="center",
        )
        legend.set_frame_on(True)
        legend.get_frame().set_alpha(0.9)

    return legend



def _obs_col_width_from_names(names, base=30, wmin=28, wmax=72, pad=2):
    """Pick a width for the Observation column that fits the longest name."""
    try:
        longest = max(len(str(n)) for n in names) + pad
    except ValueError:
        longest = base
    return max(wmin, min(wmax, max(base, longest)))
    
def print_aligned_latex_table(
    latex_table: List[List[str]],
    parameter_labels: Sequence[str],
    observation_names: Sequence[str],
    out=None,
) -> None:
    """
    Pretty-print an aligned LaTeX-style table.

    If `out` is None, print to stdout with colour accents.
    If `out` is a file-like object, write plain text to that file.
    """
    blue, red, reset = "\033[34m", "\033[31m", "\033[0m"

    # Dynamic width for the Observation column
    obs_w = _obs_col_width_from_names(observation_names, base=30, wmin=28, wmax=72, pad=2)

    header = ["Observation"] + list(parameter_labels)
    head_str = f"{header[0]:<{obs_w}} | " + " | ".join(f"{c:<32}" for c in header[1:])

    if out is None:
        print(blue + head_str + reset)
        print("-" * len(head_str))
    else:
        print(head_str, file=out)
        print("-" * len(head_str), file=out)

    for name, row in zip(observation_names, latex_table):
        row_str = " | ".join(f"{c:<32}" for c in row)
        if out is None:
            obs_str = red + f"{name:<{obs_w}}" + reset
            print(f"{obs_str} | {row_str}")
        else:
            obs_str = f"{name:<{obs_w}}"
            print(f"{obs_str} | {row_str}", file=out)

def print_parameter_list_table(
    row: Sequence[str],
    parameter_labels: Sequence[str],
    observation_name: str,
    title: str | None = None,
    out=None,
) -> None:
    """
    Print a vertical 'Parameter | Value' table for a single observation row.

    Only non-empty cells are shown. If `out` is provided, write to that file.
    """
    blue, red, reset = "\033[34m", "\033[31m", "\033[0m"

    if title is None:
        title = f"Detailed parameter table for {observation_name}"

    header = f"{'Parameter':<30} | {'Value':>30}"

    if out is None:
        print(title)
        print(blue + header + reset)
        print("-" * len(header))
    else:
        print(title, file=out)
        print(header, file=out)
        print("-" * len(header), file=out)

    for pname, val in zip(parameter_labels, row):
        v = str(val).strip()
        if not v:
            continue
        line = f"{pname:<30} | {v:>30}"
        if out is None:
            print(line)
        else:
            print(line, file=out)

def print_cmb_summary_matrix(
    aligned_table: List[List[str]],
    parameter_labels: Sequence[str],
    observation_names: Sequence[str],
    cmb_param_names: Sequence[str],
    title: str = "CMB parameter summary (rows = params, columns = CMB obs):",
    out=None,
) -> None:
    """
    Print a matrix:

        CMB parameter | Obs1 | Obs2 | ...

    where columns are all observation rows whose name contains 'CMB'.
    Only parameters in `cmb_param_names` and present in `parameter_labels` are shown.
    """
    # Select CMB rows by name (simple and robust)
    cmb_row_indices = [
        i for i, name in enumerate(observation_names) if "CMB" in str(name)
    ]
    if not cmb_row_indices:
        return

    # Which parameters will form the rows?
    params = [p for p in cmb_param_names if p in parameter_labels]
    if not params:
        return

    # Column indices of those parameters
    col_index = {p: parameter_labels.index(p) for p in params}

    # Build header: 'CMB parameter' + CMB obs names
    header = ["CMB parameter"] + [observation_names[i] for i in cmb_row_indices]
    col_w = 26
    val_w = 32

    head_str = f"{header[0]:<{col_w}} | " + " | ".join(
        f"{name:<{val_w}}" for name in header[1:]
    )

    if out is None:
        print(title)
        print(head_str)
        print("-" * len(head_str))
    else:
        print(title, file=out)
        print(head_str, file=out)
        print("-" * len(head_str), file=out)

    # Each row: one parameter
    for p in params:
        values = []
        for ridx in cmb_row_indices:
            row = aligned_table[ridx]
            values.append(str(row[col_index[p]]).strip() or " ")
        row_str = f"{p:<{col_w}} | " + " | ".join(f"{v:<{val_w}}" for v in values)
        if out is None:
            print(row_str)
        else:
            print(row_str, file=out)

        
def _obs_col_width(stats_list, base=38, wmin=30, wmax=68, pad=2):
    """Pick a nice width for the Observation column.
    - Start from `base`
    - If a longer name appears, grow up to `wmax`
    - Never shrink below `wmin`"""
    try:
        maxlen = max(len(str(s.get("Observation", ""))) for s in stats_list) + pad
    except ValueError:
        maxlen = base
    return max(wmin, min(wmax, max(base, maxlen)))
    
def print_stats_table(model: str, stats_list):
    red, blue, reset = "\033[31m", "\033[34m", "\033[0m"

    # --- coerce numpy scalars / 0-D arrays / len-1 arrays to float
    import numpy as _np
    def _as_float(x):
        try:
            arr = _np.asarray(x)
            if arr.ndim == 0:
                return float(arr)
            if arr.size == 1:
                return float(arr.reshape(()))
            if _np.isfinite(arr).any():
                return float(_np.nanmean(arr))
            return float('nan')
        except Exception:
            try:
                return float(x)
            except Exception:
                return float('nan')

    # NEW: dynamic width for the Observation column
    obs_w = _obs_col_width(stats_list, base=38, wmin=30, wmax=68, pad=2)

    header = (
        f"{'Observation':<{obs_w}} | {'Log-Likelihood':>18} | {'Chi-Squared':>15} | "
        f"{'Reduced Chi-Squared':>20} | {'AIC':>11} | {'BIC':>11} | {'dAIC':>11} | {'dBIC':>11}"
    )
    print(f"Statistical Results for Model: {model}")
    print(blue + header + reset)
    print("-" * len(header))  # ruler matches header width

    for stats in stats_list:
        # pad BEFORE coloring to keep alignment
        plain_obs = f"{stats['Observation']:<{obs_w}}"
        obs  = str(stats.get('Observation', ''))
        ll   = _as_float(stats.get('Log-Likelihood', _np.nan))
        chi2 = _as_float(stats.get('Chi_squared', _np.nan))
        rchi = _as_float(stats.get('Reduced_Chi_squared', _np.nan))
        aic  = _as_float(stats.get('AIC', _np.nan))
        bic  = _as_float(stats.get('BIC', _np.nan))
        daic = _as_float(stats.get('dAIC', _np.nan))
        dbic = _as_float(stats.get('dBIC', _np.nan))

        obs_str = f"{obs:<{obs_w}}"
        print(
            f"{obs_str} | {ll:>18.4f} | {chi2:>15.4f} | "
            f"{rchi:>20.4f} | {aic:>11.3f} | {bic:>11.3f} | "
            f"{daic:>11.3f} | {dbic:>11.3f}"
        )
        #print(row)

# =============================================================================
# Data helpers
# =============================================================================

def _compat_group_of(name: str) -> set[str]:
    for g in COMBINE_GROUPS:
        if name in g:
            return g
    return {name}


def partition_by_compatibility(obs_list: Sequence[str]) -> List[List[str]]:
    """Group compatible observable types into columns for plotting."""
    parts: List[List[str]] = []
    remaining = list(obs_list)
    while remaining:
        this = [remaining.pop(0)]
        g = _compat_group_of(this[0])
        i = 0
        while i < len(remaining):
            if remaining[i] in g:
                this.append(remaining.pop(i))
            else:
                i += 1
        parts.append(this)
    return parts


def residual_unit(obs_type: str) -> str:
    if obs_type in ("OHD", "CC"):
        return r"km s$^{-1}$ Mpc$^{-1}$"
    if obs_type in ("PantheonP", "Pantheon", "JLA", "DESY5", "Union3"):
        return "mag"
    if obs_type in ("BBN_DH", "BBN_DH_AlterBBN"):
        return "D/H (absolute)"
    return ""  # BAO/DESI/f/f_sigma_8 dimensionless

def rd_policy_label(obs_list, model_config: dict, obs_index: int | None = None) -> str:
    """
    Return concise policy tokens for an observation group, e.g.:
      BAO                -> "rd: fixed 147.5 (RD_SINGLETON) Mpc" / "rd: free"
      BAO+CC             -> "rd: free"
      BAO+CC+f_sigma_8   -> "rd: free, gamma: free"
      BAO+CMB / BAO+BBN  -> "rd: calibrated"
      f_sigma_8 (solo)   -> "gamma: fixed 0.55" / "gamma: free"
    """
    if not isinstance(obs_list, (list, tuple)) or not obs_list:
        return ""

    tokens = []
    tags   = [str(t) for t in obs_list]
    lower  = [t.lower() for t in tags]

    # -------- Part 1: r_d token (BAO/DESI) --------
    has_bao_desi = any(("bao" in L) or ("desi" in L) for L in lower)
    has_cc       = any(L == "cc" for L in lower)
    has_cal      = any(L.startswith("bbn") or L.startswith("cmb") for L in lower)

    if has_bao_desi:
        # Early-time calibrator present -> calibrated
        if has_cal:
            tokens.append("rd: calibrated")
        # If BAO is combined with CC (or any non-calibrator second dataset) -> FREE
        elif has_cc or len(obs_list) > 1:
            tokens.append("rd: free")
        else:
            # Singleton BAO policy (typical DESI rule): FIX to configured value if present
            rdpol = model_config.get("rd_policy", {})
            mode  = str(rdpol.get("mode", "")).lower()
            if mode.startswith("fixed") or ("fixed_value" in rdpol):
                val = rdpol.get("fixed_value")
                try:
                    val = float(val)
                    tokens.append(f"rd: fixed {val:g} Mpc")
                except Exception:
                    tokens.append("rd: fixed")
            else:
                # Fallback: if no explicit fixed policy, treat singleton as free
                tokens.append("rd: free")

    # -------- Part 2: gamma token for f_sigma_8 --------
    has_fs8 = any(("f_sigma_8" in L) or ("fσ8" in tags[i]) for i, L in enumerate(lower))
    if has_fs8:
        fixed_map = model_config.get("fs8_gamma_fixed_by_group", {}) or {}

        matched_index = obs_index
        if matched_index is None:
            for i, grp in enumerate(model_config.get("observations", [])):
                if list(grp) == list(obs_list):
                    matched_index = i
                    break

        if (matched_index is not None) and (matched_index in fixed_map):
            val = fixed_map.get(matched_index)
            try:
                val = float(val)
                tokens.append(f"gamma: fixed {val:.2f}")
            except Exception:
                tokens.append("gamma: fixed")
        else:
            tokens.append("gamma: free")

    return ", ".join(tokens)


    
def extract_observation_data(
    data: Mapping[str, Mapping[str, Any]],
    obs_type: str,
    params_median: Optional[Mapping[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return (z, y, yerr, meta) for a given observation type.

    For DESI/BAO, `meta` encodes per-point type codes (3,5,6,7,8).
    For PantheonP / PantheonPS, we plot `m_b_corr - M_abs` if `M_abs` is provided.
    """

    # --- Resolve Pantheon aliases between internal labels and data keys ---
    key = obs_type
    if obs_type.startswith("PantheonP") and obs_type not in data:
        # Prefer an exact PantheonP dataset if present; otherwise fall back to PantheonPS
        if "PantheonP" in data:
            key = "PantheonP"
        elif "PantheonPS" in data:
            key = "PantheonPS"

    obs_data = data[key]

    # --- Pantheon+ (with or without SH0ES anchor) ---
    if key in ("PantheonP", "PantheonPS"):
        z = np.asarray(obs_data["zHD"])  # redshift
        M0 = 0.0
        if params_median is not None and "M_abs" in params_median:
            M0 = float(params_median["M_abs"])
        y = np.asarray(obs_data["m_b_corr"]) - M0
        yerr = np.asarray(obs_data.get("type_data_error", np.zeros_like(y)))
        return z, y, yerr, None

    # --- CMB HANDLING ---
    if obs_type.startswith("CMB_"):
        # Map tag to file key
        mode = "TT"
        if "lowl" in obs_type: mode = "EE"
        
        # Define bounds (Adjust these based on your specific requirements)
        if "lowl" in obs_type:
            l_min, l_max = 2, 30
        else:
            l_min, l_max = 30, 2500
        
        file_path = DEFAULT_CMB_FILES.get(mode) 
        
        if file_path and os.path.exists(file_path):
            ell, Dl, yerr = read_bandpowers(file_path)
            
            # Now l_min and l_max are defined for the mask
            mask = (ell >= l_min) & (ell <= l_max)
            
            if yerr is not None:
                if yerr.ndim == 2:
                    yerr_masked = yerr[:, mask]
                else:
                    yerr_masked = yerr[mask]
            else:
                yerr_masked = None

            return ell[mask], Dl[mask], yerr_masked, None

    if obs_type in ("BBN_PryMordial", "BBN_prior"):
        return np.array([]), np.array([]), np.array([]), {"is_prior": True}
    
    # --- BBN (D/H) ---
    if obs_type in ("BBN_DH", "BBN_DH_AlterBBN"):
        # Convert to absolute number ratio if input is in 10^6 × (D/H)
        units = obs_data.get("units", "absolute")
        scale = 1e-6 if units == "scaled1e6" else 1.0
        mode  = obs_data.get("mode", "mean")
        # meta passes backend + constants (and grid if present) to the evaluator
        meta = {
            "bbn_backend": obs_data.get("bbn_backend", "approx"),
            # Fall back to global defaults when the dataset itself does not specify Neff / tau_n
            "Neff":  float(obs_data.get("Neff",  N_EFF_DEFAULT)),
            "tau_n": float(obs_data.get("tau_n", TAU_N_DEFAULT)),
            "bbn_grid":   obs_data.get("bbn_grid", None),
        }
        if mode == "mean" and "weighted_mean" in obs_data:
            wm   = obs_data["weighted_mean"]
            z    = np.array([0.0], dtype=float)  # plot at z=0
            y    = np.array([float(wm.get("DH", wm.get("value"))) * scale], dtype=float)
            yerr = np.array([float(wm.get("sigma", 0.0)) * scale], dtype=float)
            return z, y, yerr, meta
        # per-system mode
        systems = obs_data.get("systems", [])
        if not systems:
            return np.array([]), np.array([]), np.array([]), meta
        y    = np.array([float(s.get("DH", s.get("value"))) * scale for s in systems], dtype=float)
        yerr = np.array([float(s.get("sigma", 0.0))          * scale for s in systems], dtype=float)
        # Prefer absorber redshift if available; else plot vs index
        if all(("z" in s) for s in systems):
            z = np.array([float(s["z"]) for s in systems], dtype=float)
        else:
            z = np.arange(len(systems), dtype=float)
        return z, y, yerr, meta
        
    if obs_type == "BAO":
        z = np.array([0.295, 0.510, 0.510, 0.706, 0.706, 0.930, 0.930, 1.317, 1.317, 1.491, 2.330, 2.330], dtype=float)
        if "obs_vec" in obs_data and obs_data["obs_vec"] is not None:
            y = np.asarray(obs_data["obs_vec"], dtype=float)
        else:
            y = np.array([
                7.92512927,
                13.6200308, 20.98334647,
                16.84645313, 20.07872919,
                21.70841761, 17.87612922,
                27.78720817, 13.82372285,
                26.07217182,
                39.70838281, 8.52256583,
            ], dtype=float)
        cov  = np.asarray(obs_data["covd1"])
        yerr = np.sqrt(np.diag(cov)) if cov.ndim == 2 else np.zeros_like(y)
        meta = np.array([3, 8, 6, 8, 6, 8, 6, 8, 6, 3, 8, 6], dtype=int)
        return z, y, yerr, meta
        
    if obs_type in ("DESI", "DESI_DR1", "DESI_DR2"):
        # z
        if   "redshift" in obs_data: z = np.asarray(obs_data["redshift"])
        elif "z_eff"    in obs_data: z = np.asarray(obs_data["z_eff"])
        elif "z"        in obs_data: z = np.asarray(obs_data["z"])
        else:                        z = np.asarray(obs_data.get("z_data", []))

        # y (measurement vector)
        if   "measurement" in obs_data: y = np.asarray(obs_data["measurement"])
        elif "y"           in obs_data: y = np.asarray(obs_data["y"])
        elif "y_data"      in obs_data: y = np.asarray(obs_data["y_data"])
        else:                           y = np.asarray(obs_data.get("type_data", []))  # last resort

        # 1σ errors: prefer (inv_cov)^(-1/2) diag, then cov diag; else explicit field(s)
        if "cov" in obs_data and obs_data["cov"] is not None:
            cov = np.asarray(obs_data["cov"])
            yerr = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        elif "inv_cov" in obs_data and obs_data["inv_cov"] is not None:
            inv = np.asarray(obs_data["inv_cov"])
            diag_inv = np.diag(inv)
            with np.errstate(divide="ignore", invalid="ignore"):
                yerr = np.sqrt(np.where(diag_inv > 0, 1.0 / diag_inv, np.nan))
        elif "measurement_error" in obs_data:
            yerr = np.asarray(obs_data["measurement_error"])
        elif "sigma" in obs_data:
            yerr = np.asarray(obs_data["sigma"])
        else:
            yerr = np.zeros_like(y)

        # meta/type codes for multi-curve selection (3,5,6,7,8)
        if   "type_data" in obs_data: meta = np.asarray(obs_data["type_data"])
        elif "types"     in obs_data: meta = np.asarray(obs_data["types"])
        elif "type"      in obs_data: meta = np.asarray(obs_data["type"])
        elif "code"      in obs_data: meta = np.asarray(obs_data["code"])
        elif "codes"     in obs_data: meta = np.asarray(obs_data["codes"])
        else:                         meta = None
        
        if meta is not None and (meta == 4).any():
            try:
                # Prefer explicit r_d if present; otherwise compute from params_median
                if params_median is not None and "r_d" in params_median:
                    rs = float(params_median["r_d"])
                elif params_median is not None:
                    rs = float(compute_rd(params_median))
                else:
                    rs = R_D_SINGLETON
            except Exception:
                rs = R_D_SINGLETON

            idx4 = (meta == 4)
            vals = y[idx4]
            med  = float(np.nanmedian(np.abs(vals))) if vals.size else np.nan

            if np.isfinite(med) and med < 100.0:
                # Already DV/rd → just relabel
                meta = meta.copy()
                meta[idx4] = 3
            else:
                # Convert raw DV → DV/rd (and scale errors), relabel
                y    = y.copy()
                yerr = yerr.copy()
                y[idx4]    = vals / rs
                yerr[idx4] = yerr[idx4] / rs
                meta       = meta.copy()
                meta[idx4] = 3

        return z, y, yerr, meta

        return z, y, yerr, meta

    # Generic (OHD, CC, JLA, Pantheon, f, f_sigma_8, ...)
    for zkey in ("redshift", "z", "z_eff", "zeff"):
        if zkey in obs_data:
            z = np.asarray(obs_data[zkey])
            break
    else:
        raise KeyError(f"No redshift key found for {obs_type}")

    if   "type_data" in obs_data:      y = np.asarray(obs_data["type_data"])
    elif "measurement" in obs_data:    y = np.asarray(obs_data["measurement"])
    elif "y" in obs_data:              y = np.asarray(obs_data["y"])
    elif "y_data" in obs_data:         y = np.asarray(obs_data["y_data"])
    else:
        raise KeyError(f"No measurement key found for {obs_type}")

    if   "type_data_error" in obs_data: yerr = np.asarray(obs_data["type_data_error"])
    elif "measurement_error" in obs_data: yerr = np.asarray(obs_data["measurement_error"])
    elif "sigma" in obs_data:             yerr = np.asarray(obs_data["sigma"])
    else:                                 yerr = np.zeros_like(y)

    return z, y, yerr, None


def fetch_best_fit_values(
    combined_best_fit: Mapping[str, Sequence[float]]
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Return (median, upper, lower) dicts from the combined best-fit mapping."""
    med = {k: float(v[0]) for k, v in combined_best_fit.items()}
    up  = {k: float(v[1]) for k, v in combined_best_fit.items()}
    lo  = {k: float(v[2]) for k, v in combined_best_fit.items()}
    return med, up, lo


# =============================================================================
# Model evaluators (cosmology/model math)
# =============================================================================
def _with_derived_background(p: Mapping[str, float]) -> Dict[str, float]:
    tm = dict(p)
    if ("Omega_m" not in tm) and all(k in tm for k in ("H_0", "Omega_bh^2", "Omega_dh^2")):
        try:
            h = float(tm["H_0"]) / 100.0
            if h > 0:
                tm["Omega_m"] = (float(tm["Omega_bh^2"]) + float(tm["Omega_dh^2"])) / (h*h)
        except Exception:
            pass
    return tm

def _model_func_for(model_name: str):
    return UDM.Get_model_function(model_name)


def compute_E(z: np.ndarray | float, param_dict: Mapping[str, float], model_name: str) -> np.ndarray:
    z = np.atleast_1d(z)
    if z.size == 0:                       # ← NEW: empty guard
        return z.astype(float)
    p = _with_derived_background(param_dict)
    return _model_func_for(model_name)(z, p)  # user model returns E(z)

def compute_Dc(z: np.ndarray | float, param_dict: Mapping[str, float], model_name: str) -> np.ndarray:
    z = np.atleast_1d(z)
    if z.size == 0:                       # ← NEW
        return z.astype(float)
    p = _with_derived_background(param_dict)
    return UDM.Comoving_distance_vectorized(_model_func_for(model_name), z, p)

def compute_DM(z: np.ndarray | float, param_dict: Mapping[str, float], model_name: str) -> np.ndarray:
    return compute_Dc(z, param_dict, model_name)

def compute_DV(z: np.ndarray | float, param_dict: Mapping[str, float], model_name: str) -> np.ndarray:
    z = np.atleast_1d(z)
    if z.size == 0:                       # ← NEW
        return z.astype(float)
    p  = _with_derived_background(param_dict)
    Ez = compute_E(z, p, model_name)
    DM = compute_DM(z, p, model_name)
    H0 = float(p["H_0"])
    DH = C_KM_S / (H0 * Ez)
    return (DM*DM * DH * z)**(1.0/3.0)    # (cz/H)*DM^2, with c absorbed in units via C_KM_S

def _compact_params(p):
    keep = ("Omega_m", "H_0", "Omega_bh^2", "N_eff")
    out  = {k: float(p[k]) for k in keep if k in p}
    return out or {k: float(v) for k, v in list(p.items())[:4]}


def _rate_limited_warn(msg, _payload):
    global _RD_WARN_LAST
    now = _time.time()
    if now - _RD_WARN_LAST > 60.0:
        logger.warning(msg)
        _RD_WARN_LAST = now
    else:
        logger.debug("%s (suppressed)", msg)


def compute_f(z: np.ndarray | float, param_dict: Mapping[str, float], model_name: str) -> np.ndarray:
    z = np.atleast_1d(z)
    if z.size == 0:                       # ← NEW
        return z.astype(float)
    p  = _with_derived_background(param_dict)
    Ez = compute_E(z, p, model_name)
    Omz = (p["Omega_m"] * (1.0 + z)**3) / (Ez**2)
    gamma = float(p.get("gamma", DEFAULT_GROWTH_INDEX))
    return Omz**gamma

def compute_sigma8z(
    z: np.ndarray | float,
    param_dict: Mapping[str, float],
    model_name: str,
) -> np.ndarray:
    """
    Compute sigma_8(z) for plotting.

    Physically:
        σ8(z) = σ8(0) * D(z)
              = σ8(0) * exp(-I(z))

    where:
      • I(z)   – growth integral, provided by User_defined_modules.integral_term_array
    """
    z = np.atleast_1d(z)
    if z.size == 0:
        return z.astype(float)

    p = _with_derived_background(param_dict)
    gamma = float(p.get("gamma", DEFAULT_GROWTH_INDEX))

    MODEL = _model_func_for(model_name)

    integ = UDM.integral_term_array(z, p, MODEL, gamma)
    sigma8_0 = float(p["sigma_8"])

    # CORRECT: σ8(z) = σ8 * exp(-I(z))
    return sigma8_0 * np.exp(-integ)



# =============================================================================
# Plot evaluators (turn params → observable curves)
# =============================================================================

def model_curve_for_type(
    obs_type: str,
    zgrid: np.ndarray,
    param_dict: Mapping[str, float],
    model_name: str,
    MODEL_funcs: Mapping[str, Any],
) -> Tuple[np.ndarray, str]:
    """Return model curve y(z) and a y-axis label for the given observable."""
    # Local default: GR-like growth index, used only if gamma is absent
    DEFAULT_GROWTH_INDEX = GAMMA_FS8_SINGLETON

    # Work with a copy so we can safely inject gamma if needed
    p = dict(param_dict)
    if "gamma" not in p:
        p["gamma"] = DEFAULT_GROWTH_INDEX
        gamma_fixed = True
    else:
        gamma_fixed = False

    if obs_type in ("OHD", "CC"):
        Ez = MODEL_funcs["E"](zgrid, p, model_name)
        return p["H_0"] * Ez, r"$H(z)$ (km s$^{-1}$ Mpc$^{-1}$)"

    if obs_type in ("PantheonP", "Pantheon", "JLA", "DESY5", "Union3"):
        Dc = MODEL_funcs["Dc"](zgrid, p, model_name)
        mu = 25.0 + 5.0 * np.log10(np.clip((1.0 + zgrid) * Dc, 1e-12, None))
        return mu, r"Distance modulus $\mu$ (mag)"

    if obs_type in ("f", "f_sigma_8"):
        fz = MODEL_funcs["f"](zgrid, p, model_name)
        if obs_type == "f":
            ylab = r"$f(z)$" + (r" ($\gamma$ fixed)" if gamma_fixed else "")
            return fz, ylab
        # fσ8:
        sig = MODEL_funcs["sigma8z"](zgrid, p, model_name)
        ylab = r"$f\sigma_8(z)$" + (r" ($\gamma$ fixed)" if gamma_fixed else "")
        return fz * sig, ylab

    if obs_type == "BAO":
        DV = MODEL_funcs["DV"](zgrid, p, model_name)
        rd = MODEL_funcs["rd"](p)
        return DV / rd, r"$D_V(z)/r_d$ (dimensionless)"

    if obs_type == "DESI":
        DM = MODEL_funcs["DM"](zgrid, p, model_name)
        rs = MODEL_funcs["rd"](p)
        return DM / rs, r"$D_M(z)/r_s$ (DESI DR1, dimensionless)"
        
    if obs_type in ("BBN_DH", "BBN_DH_AlterBBN"):
        # For plotting, D/H is effectively constant in z: draw a flat line.
        try:
            from Statistical_packages import (
                bbn_predict_alterbbn,
                bbn_predict_approx,
                bbn_predict_grid,
            )
        except Exception:
            bbn_predict_alterbbn = bbn_predict_approx = bbn_predict_grid = None

        backend = str(param_dict.get("bbn_backend", "approx"))

        try:
            if backend == "alterbbn" and bbn_predict_alterbbn is not None:
                DH_th = float(bbn_predict_alterbbn(param_dict))
            elif backend == "alterbbn_grid" and bbn_predict_grid is not None:
                DH_th = float(bbn_predict_grid(param_dict))
            else:
                DH_th = float(bbn_predict_approx(param_dict))
        except Exception:
            DH_th = float("nan")

        return np.full_like(zgrid, DH_th, dtype=float), r"D/H (absolute)"
        
    if obs_type.startswith("CMB_"):
        from Kosmulator_main import Statistical_packages as SP

        if zgrid.size == 0:
            return np.array([]), r"$D_{\ell}$ [$\mu K^2$]"

        # Pick likelihood mode
        mode = "lowl" if "lowl" in obs_type else "hil"

        # Pick spectrum
        spec_key = "tt"
        if ("TE" in obs_type) or ("_TE" in obs_type):
            spec_key = "te"
        if ("EE" in obs_type) or ("_EE" in obs_type) or ("lowl" in obs_type):
            spec_key = "ee"

        # Compute up to what we need for the dense curve
        l_req = int(np.nanmax(zgrid))
        l_req = max(30, l_req + 5)

        cls = SP._compute_cls_cached(param_dict, l_req, mode=mode)
        arr = np.asarray(cls.get(spec_key, []), dtype=float)

        if arr.size < 3:
            return np.full_like(zgrid, np.nan), r"$D_{\ell}$ [$\mu K^2$]"

        l_max = min(l_req, arr.size - 1)
        if l_max < 2:
            return np.full_like(zgrid, np.nan), r"$D_{\ell}$ [$\mu K^2$]"

        ell = np.arange(2, l_max + 1, dtype=float)
        prefac = ell * (ell + 1) / (2.0 * np.pi) * SP.TCMB2
        Dl_model = prefac * arr[2 : l_max + 1]

        y_model = np.interp(zgrid, ell, Dl_model)
        label = r"$D_{\ell}^{" + spec_key.upper() + r"}$ [$\mu K^2$]"
        return y_model, label

    return np.full_like(zgrid, np.nan), obs_type


def evaluator_for_points(
    obs_type: str,
    z: np.ndarray,
    meta_type: Optional[np.ndarray | int],
    param_dict: Mapping[str, float],
    model_name: str,
    MODEL_funcs: Mapping[str, Any],
) -> np.ndarray:
    """Evaluate a model at data-point locations. Handles DESI/BAO meta codes.

    Codes: 3→DV/rs, 5→DA/rs, 6→DH/rs, 7→rs/DV, 8→DM/rs
    """
    z = np.atleast_1d(z)
    if z.size == 0:                       
        return z.astype(float)

    if obs_type in ("OHD", "CC"):
        Ez = MODEL_funcs["E"](z, param_dict, model_name)
        return param_dict["H_0"] * Ez

    if obs_type in ("PantheonP", "Pantheon", "JLA", "DESY5", "Union3"):
        Dc = MODEL_funcs["Dc"](z, param_dict, model_name)
        return 25.0 + 5.0 * np.log10(np.clip((1.0 + z) * Dc, 1e-12, None))

    if obs_type == "f":
        return MODEL_funcs["f"](z, param_dict, model_name)

    if obs_type == "f_sigma_8":
        return MODEL_funcs["f"](z, param_dict, model_name) * MODEL_funcs["sigma8z"](z, param_dict, model_name)
        
    if obs_type.startswith("CMB_"):
        from Kosmulator_main import Statistical_packages as SP
        
        # 1. Determine l_max safely
        # We need the theory to cover at least the max ell in the data (z)
        data_max_l = int(np.max(z))
        l_max_request = max(2500, data_max_l + 10) # Ensure at least 2500 or data max
        
        mode = "lowl" if "lowl" in obs_type else "hil"
        
        # 2. Compute Cls
        cls = SP._compute_cls_cached(param_dict, l_max_request, mode=mode)
        
        spec_key = "tt"
        if "lowl" in obs_type: spec_key = "ee"
        
        # 3. Check actual size of returned theory
        cls_array = cls[spec_key]
        actual_max_l = len(cls_array) - 1
        
        # Create theory ell grid only up to what we actually have
        ell_theory = np.arange(2, actual_max_l + 1)
        
        # 4. Calculate Dl
        prefac = ell_theory * (ell_theory + 1) / (2.0 * np.pi) * SP.TCMB2
        Dl_theory = prefac * cls_array[2 : actual_max_l + 1]
        
        # 5. Interpolate
        # np.interp will handle values outside the range by pinning them to the boundaries
        return np.interp(z, ell_theory, Dl_theory)
        
    # --- BBN (D/H): constant prediction (independent of z) ---
    if obs_type in ("BBN_DH", "BBN_DH_AlterBBN"):
        try:
            from Statistical_packages import bbn_predict_alterbbn, bbn_predict_approx, bbn_predict_grid
        except Exception:
            bbn_predict_alterbbn = bbn_predict_approx = bbn_predict_grid = None
        backend = None
        data_arg = None
        if isinstance(meta_type, dict):
            backend  = str(meta_type.get("bbn_backend", "approx"))
            # pass through constants and prebuilt grid if available
            data_arg = {k: meta_type[k] for k in ("Neff", "tau_n") if k in meta_type}
            if "bbn_grid" in meta_type and meta_type["bbn_grid"] is not None:
                data_arg["bbn_grid"] = meta_type["bbn_grid"]
        try:
            if backend == "alterbbn" and bbn_predict_alterbbn and data_arg is not None:
                DH_th = bbn_predict_alterbbn(param_dict, data_arg)
            elif backend == "alterbbn_grid" and bbn_predict_grid and data_arg is not None:
                DH_th = bbn_predict_grid(param_dict, data_arg)
            else:
                DH_th = bbn_predict_approx(param_dict)
        except Exception:
            # robust fallback
            try:
                DH_th = bbn_predict_approx(param_dict)
            except Exception:
                DH_th = np.nan
        return np.full_like(np.atleast_1d(z), float(DH_th), dtype=float)

    # BAO / DESI
    try:
        rs = MODEL_funcs["rd"](param_dict)
    except Exception:
        # Fallback for plotting in singleton BAO/DESI sets where r_d is fixed
        rs = R_D_SINGLETON
    DM = MODEL_funcs["DM"](z, param_dict, model_name)
    DA = DM / (1.0 + z)
    Ez = MODEL_funcs["E"](z, param_dict, model_name)
    H0 = float(param_dict["H_0"])
    DH = C_KM_S / (H0 * Ez)
    DV = ((1.0 + z) ** 2 * DA ** 2 * (C_KM_S * z / (H0 * Ez))) ** (1.0 / 3.0)

    if meta_type is None:
        return DV / rs

    meta = np.asarray(meta_type)
    if meta.ndim == 0:
        code = int(meta)
        if code == 3: return DV / rs
        if code == 5: return DA / rs
        if code == 6: return DH / rs
        if code == 7: return 1.0 / (DV / rs)
        if code == 8: return DM / rs
        return DM / rs

    out = np.empty_like(z, dtype=float)
    def put(mask, arr):
        if np.any(mask):
            out[mask] = (arr if arr.shape == z.shape else np.asarray(arr))[mask]

    put(meta == 3, DV / rs)
    put(meta == 5, DA / rs)
    put(meta == 6, DH / rs)
    put(meta == 7, 1.0 / (DV / rs))
    put(meta == 8, DM / rs)

    unknown = ~((meta == 3) | (meta == 5) | (meta == 6) | (meta == 7) | (meta == 8))
    put(unknown, DM / rs)
    return out


# =============================================================================
# CMB helpers
# =============================================================================

def read_bandpowers(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load bandpower file with columns: ell, Dl, [err] or ell, Dl, lo, hi."""
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[0:1, :]
    ell = arr[:, 0]
    Dl  = arr[:, 1]
    if arr.shape[1] == 3:
        yerr: Optional[np.ndarray] = np.abs(arr[:, 2])
    elif arr.shape[1] >= 4:
        lo = np.abs(arr[:, 2])
        hi = np.abs(arr[:, 3])
        yerr = np.vstack([lo, hi])
    else:
        yerr = None
    return ell, Dl, yerr


def cmb_lowl_curve(param_dict: Mapping[str, float], Lmin: int = 2, Lmax: int = 29) -> Tuple[np.ndarray, np.ndarray]:
    from Kosmulator_main import Statistical_packages as SP  # local import to avoid cycles
    cl = SP._compute_cls_cached(param_dict, int(Lmax), mode="lowl")
    ee = np.asarray(cl["ee"])  # type: ignore[index]
    ell = np.arange(max(2, Lmin), Lmax + 1, dtype=int)
    D_ell = ell * (ell + 1.0) / (2.0 * np.pi) * ee[ell] * SP.TCMB2
    return ell, D_ell


# =============================================================================
# Paths & IO helpers
# =============================================================================

def _strip_trailing_subdir(path: str) -> str:
    """Remove trailing known subdir (corner, corner_plots, best_fits, auto_corr)."""
    tails = {"corner", "corner_plots", "best_fits", "auto_corr"}
    norm = os.path.normpath(path)
    tail = os.path.basename(norm)
    return os.path.dirname(norm) if tail in tails else path


def normalize_save_roots(PLOT_SETTINGS: Dict[str, Any]) -> None:
    """Force a single base directory for all plot savers (no subfolders)."""
    base = PLOT_SETTINGS.get("save_root")
    if not base:
        for key in ("cornerplot_save_path", "bestfit_save_path", "autocorr_save_path"):
            if key in PLOT_SETTINGS and PLOT_SETTINGS[key]:
                base = _strip_trailing_subdir(str(PLOT_SETTINGS[key]))
                break
        if not base:
            base = DEFAULT_PLOTS_BASE
    else:
        base = _strip_trailing_subdir(str(base))

    PLOT_SETTINGS["save_root"]            = base
    PLOT_SETTINGS["cornerplot_save_path"] = base
    PLOT_SETTINGS["bestfit_save_path"]    = base
    PLOT_SETTINGS["autocorr_save_path"]   = base


def base_dir(PLOT_SETTINGS: Mapping[str, Any]) -> str:
    """Return unified base dir INCLUDING exactly one `<output_suffix>`."""
    root = os.path.normpath(str(PLOT_SETTINGS.get("save_root", DEFAULT_PLOTS_BASE)))
    suffix = str(PLOT_SETTINGS.get("output_suffix", "default_run"))
    return root if os.path.basename(root) == suffix else os.path.join(root, suffix)


def save_figure(fig, model_name: str, obs_key: Optional[str], fname_suffix: str, PLOT_SETTINGS: Mapping[str, Any]) -> str:
    """Unified saver for best-fit style figures.

    Saves to: ./Plots/Saved_Plots/<suffix>/<model>/best_fits/<obs_key>/<fname_suffix>.png
    """
    root = base_dir(PLOT_SETTINGS)
    folder = os.path.join(root, model_name, "best_fits")
    if obs_key:
        folder = os.path.join(folder, obs_key.replace("+", "_"))
    os.makedirs(folder, exist_ok=True)

    out_path = os.path.join(folder, f"{fname_suffix}.png")
    fig.savefig(
        out_path,
        dpi=int(PLOT_SETTINGS.get("dpi", 300)),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    return out_path
