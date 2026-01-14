"""Plots.py — plot-producing functions only

Saves:
  ./Plots/Saved_Plots/<output_suffix>/<model>/{corner_plots,auto_corr,best_fits}/...
Stat tables:
  ./Statistical_analysis_tables/<output_suffix>/<model>/...
"""
from __future__ import annotations

import io
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
from getdist import plots as gd_plots, MCSamples

from Kosmulator_main import Statistical_packages as SP
from Kosmulator_main.constants import CODE_STYLE, C_KM_S, R_D_SINGLETON, GAMMA_FS8_SINGLETON
from Kosmulator_main import Post_processing as PP
from Kosmulator_main.utils import generate_label as cfg_generate_label
from Kosmulator_main.utils import (
    save_stats_to_file,
    save_interpretations_to_file,
)
from Kosmulator_main.rd_helpers import compute_rd as compute_rd
from Plots.Plot_functions import (
    # console banners / rules
    phase_banner, section_banner, print_rule,

    # table + label helpers
    greek_Symbols, format_for_latex, add_corner_table, pretty_obs_name,
    align_table_to_parameters, print_aligned_latex_table,
    print_parameter_list_table, print_cmb_summary_matrix,

    # observation/model helpers
    partition_by_compatibility, residual_unit,
    OBS_COLOR_ORDER, MODEL_COLOR,
    extract_observation_data, fetch_best_fit_values, save_figure,
    model_curve_for_type, evaluator_for_points, pretty_obs_name,

    # model evaluators
    compute_E, compute_Dc, compute_DM, compute_DV, compute_f, compute_sigma8z, 

    # file/dir helpers
    normalize_save_roots, base_dir, rd_policy_label,

    # stats table printer
    print_stats_table,
    read_bandpowers,
)


__all__ = [
    "generate_plots",
    "autocorrPlot",
    "make_CornerPlot",
    "best_fit_plots",
]

# =============================================================================
# Module-level constants
# =============================================================================

MODEL_FUNCS = {
    "E":       compute_E,
    "Dc":      compute_Dc,
    "DM":      compute_DM,
    "DV":      compute_DV,
    "rd":      compute_rd,
    "f":       compute_f,
    "sigma8z": compute_sigma8z,
}

PRETTY_MAP = {
    "PantheonP":        (r"Pantheon+",       "Pantheon+"),
    "PantheonP_SH0ES":  (r"Pantheon+SH0ES",  "Pantheon+SH0ES"),
    "DESI"             : "DESI DR1",
    "DESI_DR1"         : "DESI DR1",
    "DESI_DR2"         : "DESI DR2",
}

QUALITY_ORDER = {
    # BAO / DESI
    "DESI_DR2": 0, "DESI_DR1": 1, "BAO": 2,
    # Supernovae: overlay order (top → bottom)
    "PantheonP": 0,   # includes PantheonP_SH0ES (we map that to PantheonP earlier)
    "DESY5":     1,
    "Pantheon":  2,
    "Union3":    3,
    "JLA":       4,
    # H(z)
    "CC": 0, "OHD": 1,
    # Growth rate
    "f_sigma_8": 0, "f": 1,
}

# =============================================================================
# Utilities
# =============================================================================

def _latex_model_name(name: str, latex_enabled: bool, settings: dict) -> str:
    """Return a LaTeX-safe model label (consistent everywhere)."""
    if not latex_enabled:
        return name
    # user override wins
    custom = (settings or {}).get("model_latex_names", {})
    if name in custom:
        s = custom[name]
        return s if s.startswith("$") else f"${s}$"
    # already LaTeX-like
    if "$" in name:
        return name
    # simple rule: first '_' starts a subscript with the remainder
    if "_" in name:
        head, tail = name.split("_", 1)
        return rf"${head}_{{{tail}}}$"
    return f"${name}$"
    
def _displayize_key(key: str) -> str:
    """Turn internal keys into human-friendly obs labels."""
    parts = str(key).split("+")
    mapped = []
    for p in parts:
        if p == "PantheonP_SH0ES":
            mapped.append("Pantheon+SH0ES")
        elif p == "PantheonP":
            mapped.append("Pantheon+")
        else:
            mapped.append(p)
    return "+".join(mapped)

@contextmanager
def _filter_stdout(only_first_substring: str | None = None):
    """
    Capture prints and re-emit them, but:
      • drop ALL lines containing "Removed no burn in"
      • if only_first_substring is a non-empty string, keep only the FIRST line containing it

    Usage:
        with _filter_stdout(only_first_substring=""):
            ...  # empty string means: no "first-line" special-casing
    """
    import io, sys

    old_out = sys.stdout
    old_err = sys.stderr
    buf = io.StringIO()
    sys.stdout = buf
    sys.stderr = buf
    try:
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        text = buf.getvalue()
        if not text:
            return

        DROP = "removed no burn in"
        out_lines: list[str] = []
        seen_first = False
        for ln in text.splitlines():
            if DROP in ln.lower():      # case-insensitive match
                continue
            if only_first_substring:
                if only_first_substring in ln:
                    if not seen_first:
                        out_lines.append(ln)
                        seen_first = True
                    continue
            out_lines.append(ln)

        if out_lines:
            print("\n".join(out_lines))



# =============================================================================
# Orchestration
# =============================================================================

def _is_bao_like(obs_list):
    s = [str(o) for o in obs_list]
    return any(("BAO" in o) or ("DESI" in o) for o in s)

def _has_bbn(obs_list):
    return any(str(o).startswith("BBN") for o in obs_list)
        
def _has_cmb(obs_list):
    return any(str(o).startswith("CMB") for o in obs_list)

def _S_pm_from_sample(sample: np.ndarray, names: list, CONFIG: dict):
    import numpy as np
    if "H_0" not in names:
        return None
    iH = names.index("H_0")
    if "r_d" in names:
        iR = names.index("r_d")
        S_vals = C_KM_S / (sample[:, iH] * sample[:, iR])
    else:
        rd_fix = float(CONFIG.get("rd_policy", {}).get("fixed_value", R_D_SINGLETON))
        S_vals = C_KM_S / (sample[:, iH] * rd_fix)
    p16, p50, p84 = np.percentile(S_vals, [16, 50, 84])
    # Pretty “x ± y” string (uses your Stat pkg formatter)
    return PP._format_pm(p50, p50 - p16, p84 - p50), float(p50)
    
def _is_uncalibrated(obs_list):
    """BAO/DESI without calibrators (no BBN/CMB) → uncalibrated r_d."""
    tags = [str(o) for o in obs_list]
    has_bao = any(("BAO" in t) or ("DESI" in t) for t in tags)
    has_cal = any(t.startswith("BBN") or t.startswith("CMB") for t in tags)
    return has_bao and (not has_cal)
    
def _medval(d, k):
    if k not in d:
        return None
    v = d[k]
    try:
        return float(v[0])  # [median, -err, +err]
    except Exception:
        try:
            return float(v)  # scalar
        except Exception:
            return None
            
def _header_params_for_model(config_model: dict, PLOT_SETTINGS: dict) -> list[str]:
    """
    Decide which parameters appear in the 'core summary' header for tables.

    Logic:
      * Start from the union of all parameters in first–seen order.
      * If CMB tracks are present and corner_show_all_cmb_params=False,
        drop the 'extra' CMB parameters and keep only the core CMB set
        (H_0, Omega_bh^2, Omega_dh^2).
    """
    # union of parameters in first–seen order
    full_order, seen = [], set()
    for plist in config_model.get("parameters", []):
        for p in plist:
            if p not in seen:
                seen.add(p)
                full_order.append(p)

    header = list(full_order)

    # Detect CMB tracks
    has_cmb = any(
        any(str(tag).startswith("CMB_") or str(tag) == "CMB" for tag in obs_set)
        for obs_set in config_model.get("observations", [])
    )
    show_all_cmb = bool(PLOT_SETTINGS.get("corner_show_all_cmb_params", False))

    # CMB "core" set kept in the main summary
    _cmb_core = {"H_0", "Omega_bh^2", "Omega_dh^2"}

    if has_cmb and not show_all_cmb:
        # union of all parameters that appear in CMB observation groups
        cmb_param_union = set()
        for obs_set, plist in zip(
            config_model.get("observations", []),
            config_model.get("parameters", []),
        ):
            if any(str(tag).startswith("CMB_") or str(tag) == "CMB" for tag in obs_set):
                cmb_param_union.update(plist)

        keep_from_cmb = _cmb_core & cmb_param_union
        cmb_to_drop   = cmb_param_union - keep_from_cmb
        header = [p for p in header if p not in cmb_to_drop]

    return header

def _is_cmb_obs_name(name: str) -> bool:
    """Heuristic: treat any obs whose name contains 'CMB' as a CMB dataset."""
    return "CMB" in str(name)


def _main_table_parameters(config_model: dict, param_labels: list[str]) -> list[str]:
    """
    Decide which parameters should appear in the main aligned table.

    Rules:
      - Keep ALL parameters that are used with ANY non-CMB observation.
      - Always keep the CMB core set:
            H_0, Omega_bh^2, Omega_dh^2, Omega_m (derived)
      - Drop parameters that are used *only* in purely-CMB groups
        (i.e., groups whose obs list contains only CMB-tagged entries).
    """
    obs_groups   = config_model.get("observations", [])
    param_groups = config_model.get("parameters", [])

    core_cmb = {"H_0", "Omega_bh^2", "Omega_dh^2", "Omega_m (derived)"}

    # Track where each parameter is used
    usage = {p: {"cmb": False, "non_cmb": False} for p in param_labels}

    for obs_set, plist in zip(obs_groups, param_groups):
        has_cmb     = any(_is_cmb_obs_name(tag) for tag in obs_set)
        has_non_cmb = any(not _is_cmb_obs_name(tag) for tag in obs_set)

        for p in plist:
            if p not in usage:
                # parameter appears in CONFIG but not in the aligned table — ignore
                continue
            if has_cmb:
                usage[p]["cmb"] = True
            if has_non_cmb:
                usage[p]["non_cmb"] = True

    main_params: list[str] = []

    for p in param_labels:
        # 1) Always keep the CMB core, regardless
        if p in core_cmb:
            main_params.append(p)
            continue

        info = usage.get(p)
        if info is None:
            # Not seen in CONFIG at all (e.g. some derived/user param) → keep
            main_params.append(p)
            continue

        # CMB-only nuisance: appears in CMB groups, never with non-CMB
        cmb_only = info["cmb"] and not info["non_cmb"]

        # Drop only true CMB-only nuisances; keep everything else
        if not cmb_only:
            main_params.append(p)

    return main_params
    
def generate_plots(All_Samples, CONFIG, PLOT_SETTINGS, data, true_model):
    """Run the full plotting pipeline in a clean, deterministic order.

    Order:
      1) Corner plots (and collect LaTeX tables / best-fit summaries)
      2) Best-fit plots (per observation group)
      3) Print aligned LaTeX table(s)  --> immediately print S lines (if any)
      4) Statistical analysis tables
    """
    import os
    from collections import defaultdict

    # unify roots
    normalize_save_roots(PLOT_SETTINGS)

    # Collectors
    all_best_fit: dict = {}
    all_tables: dict   = {}
    stats_dict: dict   = {}
    interp_dict: dict  = {}

    # ----- 1) Corner plots ---------------------------------------------------
    phase_banner("Creating Plots")

    for model_name, Samples in All_Samples.items():
        section_banner(f"Creating corner plot for model: {model_name}...")
        best_struct, aligned_table, param_labels, obs_names = make_CornerPlot(
            Samples, CONFIG[model_name], model_name, model_name, PLOT_SETTINGS
        )
        obs_names = [_displayize_key(k) for k in obs_names]
        all_best_fit[model_name] = best_struct         # medians etc. per obs_key
        all_tables[model_name]   = (aligned_table, param_labels, obs_names)
        print_rule()
        print()

    # ----- 2) Best-fit plots -------------------------------------------------
    section_banner("Creating best-fit plots for all models/observations...")
    best_fit_plots(all_best_fit, CONFIG, data, PLOT_SETTINGS)
    print()

    # ----- 3) Tables banner + aligned LaTeX tables ---------------------------
    phase_banner("Performing Statistical analysis")

    # Collect “S lines” per model for BAO-like outputs
    bao_S_lines_by_model = defaultdict(list)

    # Threshold for "high parameter" rows (CMB high-l etc.)
    wide_row_threshold = int(PLOT_SETTINGS.get("wide_table_param_threshold", 10))

    # CMB primary list for the matrix
    CMB_PRIMARY_FOR_MATRIX = [
        "H_0", "Omega_bh^2", "Omega_dh^2", "ln10^10_As", "n_s", "tau_reio", "A_planck"
    ]

    # 3a) Pretty-print main aligned table + detailed CMB parameter table
    for model_name, (aligned_table, param_labels, obs_names) in all_tables.items():
        print_rule()
        print(f"Model: {model_name} Aligned LaTeX Table:")

        config_model = CONFIG[model_name]

        # ---------- Main table: all non-CMB + CMB core params ----------
        main_params = _main_table_parameters(config_model, list(param_labels))
        main_indices = [param_labels.index(p) for p in main_params]
        main_table   = [[row[i] for i in main_indices] for row in aligned_table]

        # This is the table you showed (JLA, f_sigma_8, combos, CMB_* rows, etc.)
        print_aligned_latex_table(main_table, main_params, obs_names)
        print()

        # ---------- Detailed CMB table: ONLY parameters used in CMB obs ----------
        # Find which rows correspond to CMB datasets
        cmb_row_indices = [
            i for i, name in enumerate(obs_names) if _is_cmb_obs_name(name)
        ]

        if cmb_row_indices:
            # Take ONLY parameters that have a non-empty value in at least one CMB row
            cmb_param_names: list[str] = []
            for j, pname in enumerate(param_labels):
                if any(str(aligned_table[i][j]).strip() for i in cmb_row_indices):
                    cmb_param_names.append(pname)

            if cmb_param_names:
                print_cmb_summary_matrix(
                    aligned_table=aligned_table,
                    parameter_labels=param_labels,
                    observation_names=obs_names,
                    cmb_param_names=cmb_param_names,
                    title="Detailed CMB parameter table (rows = params, columns = CMB obs):",
                )

        print_rule()
        print()


        # The rest of this loop: build S-lines for BAO-like obs as you already do
        best_struct = all_best_fit.get(model_name, {})
        c_km_s = C_KM_S
        explanatory_note_printed = False

        for i, obs_list in enumerate(CONFIG[model_name]["observations"]):
            # Only BAO/DESI without calibrators
            if not (_is_bao_like(obs_list) and not (_has_bbn(obs_list) or _has_cmb(obs_list))):
                continue

            key_plus   = "+".join(obs_list)
            key_us     = "_".join(obs_list)
            resolved   = cfg_generate_label(obs_list, config_model=CONFIG[model_name], obs_index=i)
            # Prefer the form that actually exists in best_struct; use exactly one
            if resolved in best_struct:
                obs_key = resolved
            elif key_plus in best_struct:
                obs_key = key_plus
            elif key_us in best_struct:
                obs_key = key_us
            else:
                continue

            med = best_struct[obs_key]
            H0  = _medval(med, "H_0")

            # r_d: prefer sampled; else policy-fixed; else EH98 from medians
            rd = _medval(med, "r_d")
            if rd is None:
                rdpol = CONFIG[model_name].get("rd_policy", {})
                mode  = str(rdpol.get("mode", "")).lower()
                if mode.startswith("fixed") or ("fixed_value" in rdpol):
                    try:
                        rd = float(rdpol.get("fixed_value", R_D_SINGLETON))
                    except Exception:
                        rd = float(R_D_SINGLETON)
                if rd is None:
                    from Kosmulator_main.rd_helpers import compute_rd as _rd
                    p = {}
                    for k in ("H_0", "Omega_m", "Omega_bh^2", "N_eff", "T_CMB"):
                        v = _medval(med, k)
                        if v is not None:
                            p[k] = v
                    rd = _rd(p) if all(k in p for k in ("H_0","Omega_m","Omega_bh^2")) else None

            if (H0 is not None) and (rd is not None) and (H0 > 0) and (rd > 0):
                if not explanatory_note_printed:
                    bao_S_lines_by_model[model_name].append(
                        "Note: For BAO-like results without an early-time calibrator, we assume a fiducial rd = 147.5 Mpc. The truly "
                        "data-driven combination is S ≡ c/(H0·r_d), which we also report below"
                    )
                    explanatory_note_printed = True

                S_val = c_km_s / (H0 * rd)
                bao_S_lines_by_model[model_name].append(
                    f"S (c/(H0·r_d)) for {obs_key}: {S_val:.5f}"
                )

        # Immediately print the S lines to console (after the table, before stats)
        if bao_S_lines_by_model[model_name]:
            for line in bao_S_lines_by_model[model_name]:
                print(line)
            print()  # spacer

    # ----- 4) Statistical analysis ------------------------------------------
    print()
    statistical_results = PP.statistical_analysis(all_best_fit, data, CONFIG, true_model)

    # Build reference chi^2 map (from true model) for diagnostics
    ref_chi2 = {}
    if true_model in statistical_results:
        for obs_key, stats in statistical_results[true_model].items():
            ref_chi2[obs_key] = stats["Reduced_Chi_squared"]

    # Collect per-model stats and interpretations
    for model, obs_results in statistical_results.items():
        stats_dict[model]  = []
        interp_dict[model] = []
        for obs_key, stats in obs_results.items():
            # Diagnostics vs reference
            reference_chi2 = None if model == true_model else ref_chi2.get(obs_key)
            diagnostics = PP.provide_model_diagnostics(
                reduced_chi_squared=stats["Reduced_Chi_squared"],
                model_name=model,
                reference_chi_squared=reference_chi2,
            )
            aic_bic_lines = PP.interpret_delta_aic_bic(stats["dAIC"], stats["dBIC"]).splitlines()
            aic_text = aic_bic_lines[0].strip() if len(aic_bic_lines) > 0 else "No AIC interpretation available."
            bic_text = aic_bic_lines[1].strip() if len(aic_bic_lines) > 1 else "No BIC interpretation available."

            # Resolve the raw obs list for this obs_key  (FIX: use CONFIG[model], not CONFIG[model_name])
            obs_list, obs_idx = None, None
            for i, o in enumerate(CONFIG[model]["observations"]):
                if "+".join(o) == obs_key or "_".join(o) == obs_key:
                    obs_list, obs_idx = o, i
                    break

            label = rd_policy_label(obs_list, CONFIG[model], obs_index=obs_idx)
            disp_key = _displayize_key(obs_key)
            obs_text = disp_key if not label else f"{disp_key} [{label}]"

            row = {
                "Observation": obs_text,
                # remove r_d_policy entirely if you don't want a second line anywhere
                # "r_d_policy": label,
                "Log-Likelihood": stats["Log-Likelihood"],
                "Chi_squared": stats["Chi_squared"],
                "Reduced_Chi_squared": stats["Reduced_Chi_squared"],
                "AIC": stats["AIC"],
                "BIC": stats["BIC"],
                "dAIC": stats["dAIC"],
                "dBIC": stats["dBIC"],
            }

            # IMPORTANT: Do NOT include S in the stats row — it will be printed as a stand-alone line
            stats_dict[model].append(row)

            interp_dict[model].append({
                "Observation": _displayize_key(obs_key),
                "Reduced Chi2 Diagnostics": diagnostics.strip(),
                "AIC Interpretation": aic_text,
                "BIC Interpretation": bic_text,
            })

        # Persist to disk
    suffix      = PLOT_SETTINGS.get("output_suffix", "default_run")
    main_folder = os.path.join("Statistical_analysis_tables", suffix)
    os.makedirs(main_folder, exist_ok=True)

    for model in stats_dict:
        model_folder = os.path.join(main_folder, model)
        os.makedirs(model_folder, exist_ok=True)

        # 1) Save ONLY the pretty aligned table (no aligned_table.txt anymore)
        pretty_path = os.path.join(model_folder, "aligned_table_pretty.txt")
        aligned_table, param_labels, obs_names = all_tables[model]

        with open(pretty_path, "w", encoding="utf-8") as out:
            print(f"Model: {model} Aligned LaTeX Table:", file=out)
            print(file=out)

            # 1) Main aligned table: same core set as console
            config_model = CONFIG[model]
            main_params  = _main_table_parameters(config_model, list(param_labels))
            idx          = [param_labels.index(p) for p in main_params]
            main_table   = [[row[i] for i in idx] for row in aligned_table]

            print_aligned_latex_table(main_table, main_params, obs_names, out=out)
            print(file=out)

            # 2) Detailed CMB parameter table (only if there ARE CMB observations)
            cmb_row_indices = [i for i, name in enumerate(obs_names) if "CMB" in str(name)]
            if cmb_row_indices:
                cmb_param_names: list[str] = []
                for j, pname in enumerate(param_labels):
                    if any(str(aligned_table[i][j]).strip() for i in cmb_row_indices):
                        cmb_param_names.append(pname)

                if cmb_param_names:
                    print_cmb_summary_matrix(
                        aligned_table=aligned_table,
                        parameter_labels=param_labels,
                        observation_names=obs_names,
                        cmb_param_names=cmb_param_names,
                        title="Detailed CMB parameter table (rows = params, columns = CMB obs):",
                        out=out,
                    )

            # 3) Append S-lines directly into the pretty file (no S_scale_summary.txt anymore)
            s_lines = bao_S_lines_by_model.get(model, [])
            if s_lines:
                print(file=out)
                print("-" * 80, file=out)
                print("BAO/DESI uncalibrated summary:", file=out)
                for line in s_lines:
                    print(line, file=out)

        # 2) Then save stats & interpretations (unchanged)
        save_stats_to_file(model, model_folder, stats_dict[model])
        save_interpretations_to_file(model, model_folder, interp_dict[model])

    # Pretty-print to console (no duplicate headers)
    for model, rows in stats_dict.items():
        print_rule()
        print_stats_table(model, rows)
        print_rule()
        print()

    return all_best_fit, all_tables, statistical_results



# =============================================================================
# Diagnostics plot: Autocorrelation (live)
# =============================================================================

def autocorrPlot(
    autocorr: np.ndarray,
    index: int,
    model_name: str,
    color: str,
    obs: list,
    PLOT_SETTINGS: dict,
    plot_path: str | None = None,
    close_plot: bool = False,
    nsteps: int = 100,
    resume_offset: int = 0,
    check_every: int = 100,
    global_burn: int | None = None,
    convergence: float | None = None,
):
    """Live autocorrelation plot shown/saved during sampling."""
    if close_plot:
        plt.close()
        return
    if index <= 0 or autocorr.size == 0:
        return  # nothing to draw yet

    plt.clf()

    # 1) Diagonal slope line
    iterations = resume_offset + check_every * np.arange(0, int(nsteps / check_every) + 1)
    diag_vals  = (iterations - resume_offset) / check_every
    plt.plot(iterations, diag_vals, linestyle="--", color="k", label="slope = 1/check")

    # 2) Autocorr points up to current index
    its = resume_offset + check_every * np.arange(1, index + 1)
    tau_hat = np.asarray(autocorr[:index], dtype=float)
    # Keep non-finite as NaN so they don't collapse the y-scale
    tau_hat[~np.isfinite(tau_hat)] = np.nan
    obs_key = cfg_generate_label(obs)                  # e.g. "BAO+CC"
    obs_label = obs_key.replace("+", "_")              # filename-safe
    lbl = obs_key                                     # legend label (or pretty_obs_name(obs_key, latex_on))

    plt.plot(its, tau_hat, label=lbl, color=color)

    latex_on = bool(PLOT_SETTINGS.get("latex_enabled", False))

    # 3) Convergence target
    if convergence is not None and np.isfinite(convergence):
        if latex_on:
            label_target = rf"target $\tau$ = {convergence:.3f}"
        else:
            label_target = f"target τ = {convergence:.3f}"
        plt.axhline(convergence, linestyle=":", color="gray", label=label_target)

    # 4) Global burn
    if global_burn is not None:
        plt.axvline(global_burn, linestyle=":", color="gray", label=f"burn = {global_burn}")

    # 5) Axes/labels/legend
    left  = resume_offset + check_every
    right = resume_offset + nsteps
    if not np.isfinite(left) or not np.isfinite(right) or left >= right:
        plt.autoscale()              # or: right = left + 1; plt.xlim(left, right)
    else:
        plt.xlim(left, right)
    # Use finite tau for y-scaling (ignore NaNs/Infs)
    tau_finite = tau_hat[np.isfinite(tau_hat)]
    tau_max = float(np.max(tau_finite)) if tau_finite.size else 0.0
    diag_max = float(np.max(diag_vals)) if diag_vals.size else 1.0
    ymax_candidates = [diag_max, tau_max + 1.0]

    if convergence is not None and np.isfinite(convergence):
        ymax_candidates.append(1.1 * float(convergence))

    ymax = max(ymax_candidates)
    ymax = max(1.0, 1.05 * ymax)    # small headroom; keep non-trivial minimum
    plt.ylim(0, ymax)

    plt.title(f"Auto-Correlator: {model_name}")
    plt.xlabel("Iteration", fontsize=PLOT_SETTINGS.get("label_font_size", 12))
    plt.ylabel(r"Mean $\hat{\tau}$", fontsize=PLOT_SETTINGS.get("label_font_size", 12))
    plt.legend(fontsize=PLOT_SETTINGS.get("legend_font_size", 10))

    # 6) Save
    if plot_path is None:
        folder = os.path.join(base_dir(PLOT_SETTINGS), model_name, "auto_corr")
        os.makedirs(folder, exist_ok=True)
        obs_label = cfg_generate_label(obs).replace("+", "_")
        plot_path = os.path.join(folder, f"{obs_label}.png")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=PLOT_SETTINGS.get("dpi", 300))


# =============================================================================
# Corner plot
# =============================================================================

def make_CornerPlot(Samples, CONFIG, model_name, save_file_name, PLOT_SETTINGS):
    """
    Corner plot generator with optional reduction of CMB nuisance parameters on the PLOT ONLY.

    Switches (thread via CLI into PLOT_SETTINGS beforehand):
      - PLOT_SETTINGS["corner_show_all_cmb_params"] : bool, default False
          If False and any CMB_* obs present, only keep the commonly-reported CMB parameters
          on the corner axes, dropping the rest of the CMB nuisance parameters.
      - PLOT_SETTINGS["corner_table_full_params"]   : bool, default True
          If True, the printed/top table shows ALL parameters (recommended).
          If False, the printed/top table mirrors whatever appears on the corner axes.
    """
    # 1) union of parameters across all observation groups (first-seen order)
    full_param_order, _seen = [], set()
    for plist in CONFIG["parameters"]:
        for p in plist:
            if p not in _seen:
                _seen.add(p)
                full_param_order.append(p)
    num_params = len(full_param_order)

    # 2) GetDist plotter + typography
    g = gd_plots.get_subplot_plotter(subplot_size=4, subplot_size_ratio=0.8)
    g.settings.figure_legend_frame = True
    plot_table_flag = bool(PLOT_SETTINGS.get("plot_table", False))
    g.settings.tight_layout = not plot_table_flag
    g.settings.alpha_filled_add = 0.4
    g.settings.solid_colors = list(reversed(PLOT_SETTINGS.get("color_schemes", ["r","b","g","c","m"])))
    base_leg = float(PLOT_SETTINGS.get("legend_font_size", 12))
    leg_size = int(np.clip(base_leg + 0.5 * (num_params - 2), base_leg, base_leg + 3))
    g.settings.legend_fontsize = leg_size

    base = PLOT_SETTINGS.get("label_font_size", 12)
    tick = PLOT_SETTINGS.get("tick_font_size", 10)
    t = np.clip((num_params - 2) / 4.0, 0.0, 1.0)
    g.settings.fontsize = base + t * 6
    g.settings.axes_labelsize = base + t * 6
    g.settings.axes_fontsize = tick + t * 4
    g.settings.title_limit_fontsize = PLOT_SETTINGS.get("legend_font_size", 12)
    g.settings.scaling = False

    # 3) build distributions
    # NOTE: Here CONFIG is the model-specific CONFIG dict for this model
    resolved_keys = [
        cfg_generate_label(obs_set, config_model=CONFIG, obs_index=i)
        for i, obs_set in enumerate(CONFIG["observations"])
    ]

    distributions, obs_raw_keys = [], []
    for i, (obs_key, sample) in enumerate(Samples.items()):
        if obs_key not in resolved_keys:
            raise ValueError(
                f"Observation '{obs_key}' not in resolved observation keys: {resolved_keys}"
            )
        obs_index = resolved_keys.index(obs_key)

        names  = CONFIG["parameters"][obs_index]
        labels = greek_Symbols(names) if PLOT_SETTINGS.get("latex_enabled", False) else names

        ms = MCSamples(samples=sample, names=names, labels=labels)
        palette = PLOT_SETTINGS.get("color_schemes", ["r","b","g","c","m"])
        ms.plotColor = palette[i % len(palette)]

        distributions.append(ms)
        obs_raw_keys.append(obs_key)

    # Parameters used for the CORNER (union order). Omega_m will not appear here if Config removed it.
    header_params = list(full_param_order)

    # ---------- Corner-only CMB parameter reduction policy ----------
    # Detect presence of CMB tracks in this CONFIG
    has_cmb = any(
        any(str(tag).startswith("CMB_") or str(tag) == "CMB" for tag in obs_set)
        for obs_set in CONFIG.get("observations", [])
    )
    show_all_cmb = bool(PLOT_SETTINGS.get("corner_show_all_cmb_params", False))

    # Commonly reported CMB primary parameter set seen in the literature
    _cmb_primary = {
        "H_0", "Omega_bh^2", "Omega_dh^2", "ln10^10_As", "n_s", "tau_reio", "A_planck"
    }

    if has_cmb and not show_all_cmb:
        # Build union of parameters that belong to any CMB_* observation group
        cmb_param_union = set()
        for obs_set, plist in zip(CONFIG.get("observations", []), CONFIG.get("parameters", [])):
            if any(str(tag).startswith("CMB_") or str(tag) == "CMB" for tag in obs_set):
                cmb_param_union.update(plist)

        # Keep only the paper-standard subset from the CMB union; drop the rest from the *corner axes*
        keep_from_cmb = _cmb_primary & cmb_param_union
        cmb_to_drop = cmb_param_union - keep_from_cmb

        # Rebuild header with CMB nuisances removed; non-CMB params remain intact
        header_params = [p for p in header_params if p not in cmb_to_drop]
    # ----------------------------------------------------------------

    # 4) pretty legend labels (LaTeX or plain depending on settings)
    use_latex = bool(PLOT_SETTINGS.get("latex_enabled", False))
    legend_labels = [pretty_obs_name(k, latex_on=use_latex) for k in obs_raw_keys]

    # 5) stats + aligned latex rows (for base params)
    resolved_keys = [
        cfg_generate_label(obs_set, config_model=CONFIG, obs_index=i)
        for i, obs_set in enumerate(CONFIG["observations"])
    ]

    # Helper to map resolved → raw key for stats
    def _to_raw_key(k: str) -> str:
        return k

    # Helper to map resolved → raw for later lookups
    def _raw_of(res_key: str) -> str:
        return res_key

    # Preserve the CONFIG observation order for the stats call
    samples_for_stats = {}
    for rk in resolved_keys:
        if rk in Samples:
            samples_for_stats[_to_raw_key(rk)] = Samples[rk]
        elif _to_raw_key(rk) in Samples:
            samples_for_stats[_to_raw_key(rk)] = Samples[_to_raw_key(rk)]
        # else silently skip

    # ---- compute stats ONCE (not inside the loop)
    with _filter_stdout(only_first_substring="Removed"):
        results, latex_table, structured_values = PP.calculate_asymmetric_from_samples(
            samples_for_stats if samples_for_stats else Samples,
            CONFIG["parameters"],
            CONFIG["observations"],
        )

    # Align the table (rows match CONFIG["parameters"][i] order)
    aligned_latex_table = align_table_to_parameters(latex_table, CONFIG["parameters"])

    # Labels shown *above the printed table* (start as the aligned parameter list union per row)
    # Use the union order that the aligned table already follows:
    table_param_labels = CONFIG["parameters"][0][:]
    # Build union across groups in first-seen order:
    seen = set(table_param_labels)
    for plist in CONFIG["parameters"][1:]:
        for p in plist:
            if p not in seen:
                table_param_labels.append(p)
                seen.add(p)

    # ---- Prepare derived Omega_m column (printed table only) ----
    derived_col: list[str] = []
    for i, res_key in enumerate(resolved_keys):
        raw_key = _raw_of(res_key)
        # prefer mapped samples_for_stats, else fall back to Samples
        obs_samples = samples_for_stats.get(raw_key, Samples.get(raw_key, Samples.get(res_key)))
        if obs_samples is None:
            derived_col.append("—")
            continue

        names = CONFIG["parameters"][i]
        try:
            jH  = names.index("H_0")
            jbh = names.index("Omega_bh^2")
            jdh = names.index("Omega_dh^2")
        except ValueError:
            # Missing ingredients → mark as unavailable (e.g., runs without CMB)
            derived_col.append("—")
            continue

        H0  = np.asarray(obs_samples[:, jH],  dtype=float)
        obh = np.asarray(obs_samples[:, jbh], dtype=float)
        odh = np.asarray(obs_samples[:, jdh], dtype=float)
        h   = np.maximum(H0, 1e-12) / 100.0
        Om  = (obh + odh) / (h * h)

        p16, p50, p84 = np.percentile(Om, [16, 50, 84])
        lo, hi = p50 - p16, p84 - p50
        derived_col.append(PP._format_pm(p50, lo, hi))

    # Only append the column if at least one row has a real value
    if any(c != "—" for c in derived_col):
        table_param_labels.append("Omega_m (derived)")
        for row, cell in zip(aligned_latex_table, derived_col):
            row.append(cell)

    # --- Remap 'structured_values' keys from RAW → RESOLVED (downstream consumables use resolved)
    best_struct_resolved = {}
    for rk in resolved_keys:
        if rk in structured_values:
            best_struct_resolved[rk] = structured_values[rk]
        elif _raw_of(rk) in structured_values:
            best_struct_resolved[rk] = structured_values[_raw_of(rk)]
    structured_values = best_struct_resolved

    # --- Ensure line_args exists (needed by triangle_plot)
    line_styles = PLOT_SETTINGS.get("line_styles", ["-","--",":","-."])
    line_widths = [1.2, 1.5]
    palette = PLOT_SETTINGS.get("color_schemes", ["r","b","g","c","m"])
    line_args = [
        {"ls": line_styles[i % len(line_styles)],
         "lw": line_widths[i % len(line_widths)],
         "color": palette[i % len(palette)]}
        for i in range(len(distributions))
    ]

    # 6) triangle using the (possibly reduced) CORNER param order
   # Choose legend layout: small → 1 row, medium → ~2 rows, large → ~3 rows
    _n_leg = len(legend_labels)
    if _n_leg <= 6:
        legend_ncol = _n_leg            # single row
    elif _n_leg <= 10:
        legend_ncol = max(2, (_n_leg + 1) // 2)   # ~2 rows
    elif _n_leg <=14:
        legend_ncol = max(2, (_n_leg + 2) // 3)   # ~3 rows
    elif _n_leg <=18:
        legend_ncol = max(2, (_n_leg + 2) // 4)   # ~4 rows
    else:
        legend_ncol = max(2, (_n_leg + 2) // 4)   # ~5 rows
        
    with _filter_stdout(only_first_substring=""):
        g.triangle_plot(
            distributions,
            params=header_params,
            filled=True,
            legend_labels=legend_labels,
            legend_loc="upper right",
            legend_ncol=legend_ncol,
            line_args=line_args,
        )

    fig = getattr(g, "fig", None) or plt.gcf()

    # Decide which labels to show in the TOP TABLE
    table_full = bool(PLOT_SETTINGS.get("corner_table_full_params", True))  # default: FULL table
    if not table_full:
        # Mirror corner axes selection (i.e., drop hidden CMB nuisances from the table too)
        trimmed_labels = [p for p in table_param_labels if p in header_params or p == "Omega_m (derived)"]
    else:
        trimmed_labels = list(table_param_labels)

    # --- NEW: project the aligned table columns to exactly match `trimmed_labels`
    def _project_table(table, from_labels, to_labels):
        pos = {p: i for i, p in enumerate(from_labels)}
        out = []
        for row in table:
            out.append([row[pos[p]] if (p in pos and pos[p] < len(row)) else "" for p in to_labels])
        return out

    projected_latex_table = _project_table(aligned_latex_table, table_param_labels, trimmed_labels)

    # Table on TOP – only when requested
    if plot_table_flag:
        _ = add_corner_table(
            g,
            projected_latex_table,   # <-- use projected rows
            legend_labels,
            PLOT_SETTINGS,
            trimmed_labels,          # <-- header equals row width
            trimmed_labels,
            len(trimmed_labels),
        )
    else:
        try:
            fig.tight_layout()
        except Exception:
            pass

    # Save figure
    root   = base_dir(PLOT_SETTINGS)
    folder = os.path.join(root, model_name, "corner_plots")
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, f"corner_{save_file_name}.png")
    fig.savefig(out_path, dpi=int(PLOT_SETTINGS.get("dpi", 300)), bbox_inches="tight")
    plt.close(fig)

    # Return for later printing
    obs_names_out = [_displayize_key(k) for k in resolved_keys]
    return structured_values, aligned_latex_table, table_param_labels, obs_names_out



# =============================================================================
# Best-fit plots per observation group
# =============================================================================

def _model_first_legend(ax, model_name: str):
    """Return handles/labels with the model's entry first and de-duplicated."""
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: 0 if labels[i].startswith(f"{model_name} ") else 1)
    new_h, new_l, seen = [], [], set()
    for i in order:
        if labels[i] not in seen:
            new_h.append(handles[i])
            new_l.append(labels[i])
            seen.add(labels[i])
    return new_h, new_l

def _adjust_bestfit_margins(fig, ncols: int, PLOT_SETTINGS: dict) -> None:
    """
    Shrink the left margin as columns increase; keep right fixed.
    Also cap the top whitespace (no suptitle needed).
    Tunables live in PLOT_SETTINGS with sensible defaults.
    """
    base_left = float(PLOT_SETTINGS.get("bestfit_left_base", 0.15))   # for 1 column
    left_step = float(PLOT_SETTINGS.get("bestfit_left_step", 0.045))   # reduce per extra column
    left_min  = float(PLOT_SETTINGS.get("bestfit_left_min", 0.06))    # don't go past this
    right     = float(PLOT_SETTINGS.get("bestfit_right", 0.95))
    top       = float(PLOT_SETTINGS.get("bestfit_top", 0.985))        # trims the constant top gap
    bottom    = float(PLOT_SETTINGS.get("bestfit_bottom", 0.08))

    left = max(left_min, base_left - left_step * (max(1, ncols) - 1))
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    
def best_fit_plots(All_best_fit_values, CONFIG, data, PLOT_SETTINGS):
    """Create multi-panel best-fit plots per model and observation group."""
    use_latex = bool(PLOT_SETTINGS.get("latex_enabled", False))

    for model_name, obs_dict in All_best_fit_values.items():
        model_disp = _latex_model_name(model_name, use_latex, PLOT_SETTINGS)

        for obs_key, _stats in obs_dict.items():
            obs_list = obs_key.split("+")

            # Map resolved labels → raw dataset keys for data lookups / plotting logic
            def _raw_token(t: str) -> str:
                return t.replace("PantheonP_SH0ES", "PantheonP")

            obs_list_raw = [_raw_token(t) for t in obs_list]

            combined = All_best_fit_values[model_name][obs_key]
            params_med, params_hi, params_lo = fetch_best_fit_values(combined)

            # (Optional) diagnostics
            try:
                if any(tag in obs_key for tag in ("BAO", "DESI", "DESI_DR1", "DESI_DR2")):
                    rd_med = compute_rd(params_med)
                    print(f"[diag] {model_name} {obs_key}: median r_d ≈ {rd_med:.2f} Mpc")
            except Exception:
                pass

            try:
                h0_freeRD = 69.0  # or pull from your DESI-singleton fit
                est = h0_freeRD * (R_D_SINGLETON / rd_med)
                print(f"[diag] rough H0~ {est:.2f} km/s/Mpc from r_d scaling")
            except Exception:
                pass

            # Store band for the dedicated CMB plotter (if used elsewhere)
            PLOT_SETTINGS["cmb_params_band"] = (params_lo, params_hi)

            parts = partition_by_compatibility([t for t in obs_list_raw])

            PRIOR_ONLY = ("BBN_PryMordial", "BBN_prior")
            BBN_ONLY = ("BBN_DH", "BBN_DH_AlterBBN") + PRIOR_ONLY

            # Strip priors from each part
            parts = [[ot for ot in p if ot not in PRIOR_ONLY] for p in parts]

            # Drop empty/pure-BBN columns
            parts = [p for p in parts if p and not all(t in BBN_ONLY for t in p)]
            if not parts:
                continue

            ncols = len(parts)
            hspace = float(PLOT_SETTINGS.get("bestfit_hspace", 0.20))
            height_ratios = PLOT_SETTINGS.get("bestfit_height_ratio", (2.4, 1.0))

            fig, axes = plt.subplots(
                2, ncols,
                figsize=(PLOT_SETTINGS.get("bestfit_figwidth_percol", 6.1) * ncols, 6.5),
                sharex="col",
                gridspec_kw={
                    "wspace": PLOT_SETTINGS.get("bestfit_wspace", 0.28),
                    "hspace": hspace,
                    "height_ratios": height_ratios,
                },
                constrained_layout=False,
                squeeze=False,
            )

            _adjust_bestfit_margins(fig, ncols, PLOT_SETTINGS)
            _ = rd_policy_label(obs_list_raw, CONFIG.get(model_name, {}))  # kept for side-effects/consistency

            # Z-order policy (guarantee visibility)
            Z_OBS_BASE = 20
            Z_BAND = 150
            Z_MODEL = 200
            BAND_ALPHA = float(PLOT_SETTINGS.get("model_band_alpha", 0.15))

            for ci, part in enumerate(parts):
                part = [ot for ot in part if ot not in PRIOR_ONLY]

                ax, axr = axes[0, ci], axes[1, ci]

                if not part:
                    ax.set_visible(False)
                    axr.set_visible(False)
                    continue

                # IMPORTANT: define per-column CMB flag (your pasted version was missing this)
                is_cmb_col = any(str(t).startswith("CMB_") for t in part)

                obs_cache, all_xmins = {}, []

                # Cache all observation data in this column and collect per-column min-x
                for obs_type in part:
                    res = extract_observation_data(data, obs_type, params_median=params_med)

                    # Back-compat: normalize to (x, y, yerr, meta)
                    if isinstance(res, tuple) and len(res) == 3:
                        x_dat, y_dat, y_err = res
                        meta = {}
                    else:
                        x_dat, y_dat, y_err, meta = res

                    if isinstance(meta, dict) and meta.get("is_prior"):
                        continue

                    obs_cache[obs_type] = (x_dat, y_dat, y_err, meta)

                    if x_dat is not None and np.size(x_dat):
                        x_arr = np.asarray(x_dat, dtype=float)
                        x_valid = x_arr[np.isfinite(x_arr) & (x_arr > 0.0)]
                        if x_valid.size:
                            all_xmins.append(float(x_valid.min()))

                if not obs_cache:
                    ax.set_visible(False)
                    axr.set_visible(False)
                    continue

                # Per-column max-x from data
                xs_max = []
                for t in part:
                    try:
                        x_here = np.asarray(obs_cache[t][0], dtype=float)
                        x_valid = x_here[np.isfinite(x_here) & (x_here > 0.0)]
                        if x_valid.size:
                            xs_max.append(float(x_valid.max()))
                    except Exception:
                        pass

                _default_xmax = float(CONFIG.get(model_name, {}).get("z_max_plot", 2.5))
                xmax_data = float(np.max(xs_max)) if xs_max else _default_xmax

                # Min-x from your earlier logic; for log axes this MUST be > 0
                xmin_data = 1e-4 if not all_xmins else max(1e-4, min(all_xmins))

                xpad_axis = float(PLOT_SETTINGS.get("xpad_frac", 0.02))
                xpad_model = float(PLOT_SETTINGS.get("model_xpad_frac", 0.03))

                xmax_axis = xmax_data * (1.0 + xpad_axis)
                xmax_model = min(xmax_data * (1.0 + xpad_model), xmax_axis)

                # ---- Axis scaling/limits + x-label (fixes #2 and the label overwrite bug) ----
                if is_cmb_col:
                    ax.set_xscale("log")
                    axr.set_xscale("log")

                    # Use actual ell coverage (no forced [2,2500] here)
                    l_min_data = max(2.0, xmin_data)
                    l_max_view = max(l_min_data * 1.01, xmax_axis)

                    # Multiplicative left padding for log
                    l_min_view = max(2.0, l_min_data / (1.0 + xpad_axis))

                    ax.set_xlim(l_min_view, l_max_view)
                    axr.set_xlim(l_min_view, l_max_view)

                    ax.set_xlabel("")
                    axr.set_xlabel(r"Multipole $\ell$")
                else:
                    # Linear z-axis style
                    ax.set_xscale("linear")
                    axr.set_xscale("linear")
                    ax.set_xlim(0.0, xmax_axis)
                    axr.set_xlim(0.0, xmax_axis)
                    axr.set_xlabel(r"$z$")

                # Dense grid for theory
                N_dense = int(PLOT_SETTINGS.get("z_dense_points", 800))
                if is_cmb_col:
                    x0 = max(2.0, xmin_data)
                    x1 = max(x0 * 1.001, xmax_model)
                    z_dense = np.logspace(np.log10(x0), np.log10(x1), N_dense)
                else:
                    z_dense = np.linspace(xmin_data, max(xmin_data + 1e-6, xmax_model), N_dense)

                # ---- Model curves (drawn with high zorder so they remain visible) ----
                is_bao_desi = any(t in ("BAO", "DESI_DR1", "DESI_DR2") for t in part)

                if is_bao_desi:
                    codes_present = set()
                    for t in part:
                        if t in ("BAO", "DESI_DR1", "DESI_DR2"):
                            _, _, _, meta_here = obs_cache[t]
                            if meta_here is not None:
                                try:
                                    codes_present.update(np.unique(np.asarray(meta_here)).tolist())
                                except Exception:
                                    pass
                    if not codes_present:
                        codes_present = {8}

                    singleton = (len(obs_list_raw) == 1 and obs_list_raw[0] in ("BAO", "DESI_DR1", "DESI_DR2"))
                    rdpol = CONFIG.get(model_name, {}).get("rd_policy", {})
                    fixed_rd = float(rdpol.get("fixed_value", R_D_SINGLETON))

                    def _can_compute_eh98(params: dict) -> bool:
                        return all(k in params for k in ("H_0", "Omega_m", "Omega_bh^2"))

                    def _rd_from_params(params: dict | None) -> float:
                        if params is None:
                            return fixed_rd
                        if "r_d" in params:
                            return float(params["r_d"])
                        if _can_compute_eh98(params):
                            return MODEL_FUNCS["rd"](params)
                        return fixed_rd

                    has_cal = any(x in {
                        "BBN_DH", "BBN_DH_AlterBBN", "BBN_PryMordial",
                        "CMB_hil", "CMB_lowl", "CMB_lensing"
                    } for x in obs_list_raw)
                    has_bao = any(x in {"BAO", "DESI_DR1", "DESI_DR2"} for x in obs_list_raw)
                    has_unanch = any(x in {"JLA", "Pantheon", "f", "f_sigma_8"} for x in obs_list_raw)

                    if has_cal:
                        rs_med = _rd_from_params(params_med)
                        rs_lo = _rd_from_params(params_lo) if params_lo is not None else rs_med
                        rs_hi = _rd_from_params(params_hi) if params_hi is not None else rs_med
                    elif has_bao and has_unanch:
                        rs_med = _rd_from_params(params_med)
                        rs_lo = _rd_from_params(params_lo) if params_lo is not None else rs_med
                        rs_hi = _rd_from_params(params_hi) if params_hi is not None else rs_med
                    else:
                        rs_med = rs_lo = rs_hi = fixed_rd

                    # Compact in-panel rd badge
                    if has_cal:
                        _policy, rd_show = "calibrated", float(rs_med)
                    elif has_bao and has_unanch:
                        _policy, rd_show = "free", float(rs_med)
                    else:
                        _policy, rd_show = "fixed", float(fixed_rd)

                    fs = int(PLOT_SETTINGS.get("label_font_size", 12))
                    rd_text = (rf"$r_d$ ({_policy}): ${rd_show:.2f}\,\mathrm{{Mpc}}$"
                               if use_latex else f"r_d ({_policy}): {rd_show:.2f} Mpc")

                    ax.text(
                        0.98, 0.98, rd_text,
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=fs, zorder=Z_OBS_BASE + 5,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.65, edgecolor="0.6"),
                    )

                    DM = MODEL_FUNCS["DM"](z_dense, params_med, model_name)
                    DA = DM / (1.0 + z_dense)
                    Ez = MODEL_FUNCS["E"](z_dense, params_med, model_name)
                    H0 = float(params_med["H_0"])
                    DH = C_KM_S / (H0 * Ez)
                    DV = MODEL_FUNCS["DV"](z_dense, params_med, model_name)

                    for code in sorted(codes_present):
                        if code == 8:
                            y_curve = DM / rs_med
                        elif code == 6:
                            y_curve = DH / rs_med
                        elif code == 5:
                            y_curve = DA / rs_med
                        elif code == 3:
                            y_curve = DV / rs_med
                        elif code == 7:
                            y_curve = 1.0 / (DV / rs_med)
                        else:
                            y_curve = DM / rs_med

                        label_txt, ls = CODE_STYLE.get(code, ("", "-"))
                        ax.plot(
                            z_dense, y_curve,
                            color=MODEL_COLOR, lw=2.0, ls=ls, zorder=Z_MODEL,
                            label=(f"{model_disp} (median) {label_txt}".strip()),
                        )

                    # Optional 1σ band
                    if params_lo is not None and params_hi is not None:
                        y_low = MODEL_FUNCS["DM"](z_dense, params_lo, model_name) / rs_lo
                        y_high = MODEL_FUNCS["DM"](z_dense, params_hi, model_name) / rs_hi
                        ax.fill_between(z_dense, y_low, y_high, color="k", alpha=BAND_ALPHA,
                                        linewidth=0, zorder=Z_BAND)

                    y_label = r"BAO / DESI (dimensionless)"

                else:
                    rep_type = part[0]

                    if rep_type in ("BBN_DH", "BBN_DH_AlterBBN"):
                        backend = str(data[rep_type].get("bbn_backend", "approx"))
                        if backend == "alterbbn":
                            y0 = float(SP.bbn_predict_alterbbn(params_med, data[rep_type]))
                        elif backend == "alterbbn_grid":
                            y0 = float(SP.bbn_predict_grid(params_med, data[rep_type]))
                        else:
                            y0 = float(SP.bbn_predict_approx(params_med))

                        y_model = np.full_like(z_dense, y0, dtype=float)
                        y_label = r"D/H (absolute)"

                        ax.plot(
                            z_dense, y_model, color=MODEL_COLOR, lw=2.0, zorder=Z_MODEL,
                            label=f"{model_disp} (median)"
                        )

                        if params_lo is not None and params_hi is not None:
                            if backend == "alterbbn":
                                y_low = float(SP.bbn_predict_alterbbn(params_lo, data[rep_type]))
                                y_high = float(SP.bbn_predict_alterbbn(params_hi, data[rep_type]))
                            elif backend == "alterbbn_grid":
                                y_low = float(SP.bbn_predict_grid(params_lo, data[rep_type]))
                                y_high = float(SP.bbn_predict_grid(params_hi, data[rep_type]))
                            else:
                                y_low = float(SP.bbn_predict_approx(params_lo))
                                y_high = float(SP.bbn_predict_approx(params_hi))

                            ax.fill_between(z_dense, y_low, y_high, color="k", alpha=BAND_ALPHA,
                                            linewidth=0, zorder=Z_BAND)

                    else:
                        if rep_type == "f_sigma_8":
                            gamma_fixed = ("gamma" not in params_med)
                            if gamma_fixed:
                                txt = (
                                    rf"$\gamma$ fixed to {GAMMA_FS8_SINGLETON:.2f} (GR)"
                                    if use_latex
                                    else f"γ fixed to {GAMMA_FS8_SINGLETON:.2f} (GR)"
                                )
                                ax.text(
                                    0.98, 0.98, txt,
                                    transform=ax.transAxes, ha="right", va="top",
                                    fontsize=PLOT_SETTINGS.get("label_font_size", 12),
                                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.65, edgecolor="0.6"),
                                    zorder=Z_OBS_BASE + 5,
                                )

                        y_model, y_label = model_curve_for_type(rep_type, z_dense, params_med, model_name, MODEL_FUNCS)
                        ax.plot(
                            z_dense, y_model, color=MODEL_COLOR, lw=2.0, zorder=Z_MODEL,
                            label=f"{model_disp} (median)"
                        )
                        if params_lo is not None and params_hi is not None:
                            y_low, _ = model_curve_for_type(rep_type, z_dense, params_lo, model_name, MODEL_FUNCS)
                            y_high, _ = model_curve_for_type(rep_type, z_dense, params_hi, model_name, MODEL_FUNCS)
                            ax.fill_between(z_dense, y_low, y_high, color="k", alpha=BAND_ALPHA,
                                            linewidth=0, zorder=Z_BAND)

                ax.set_ylabel(y_label)

                # Residual reference
                axr.axhline(0.0, color="0.6", lw=1.0, zorder=1)

                # Quality-aware ordering: best first so it gets higher zorder
                def _qkey(t):
                    return (QUALITY_ORDER.get(t, 99), part.index(t))

                part_sorted = sorted(part, key=_qkey)

                # Plot observations (kept below band/model via zorder)
                for oi, obs_type in enumerate(part_sorted):
                    color = OBS_COLOR_ORDER[oi % len(OBS_COLOR_ORDER)]
                    x_dat, y_dat, y_err, meta = obs_cache[obs_type]

                    disp_obs = pretty_obs_name(obs_type, use_latex)
                    if obs_type == "PantheonP":
                        disp_obs = "Pantheon+SH0ES" if "PantheonP_SH0ES" in obs_key else "Pantheon+"

                    y_mod_pts = evaluator_for_points(
                        obs_type, x_dat, None if meta is None else meta,
                        params_med, model_name, MODEL_FUNCS
                    )

                    # Best dataset highest among observations, but still below band/model
                    z_obs = Z_OBS_BASE + (len(part_sorted) - 1 - oi)

                    ax.errorbar(
                        x_dat, y_dat, yerr=y_err, fmt="o", ms=3.5, lw=1.0,
                        color=color, ecolor=color, capsize=0, alpha=0.95, zorder=z_obs,
                        label=disp_obs,
                    )

                    overlay_flag = bool(PLOT_SETTINGS.get("overlay_model_for_bao_desi", False))
                    SNE_TYPES = ("JLA", "Pantheon", "PantheonP", "PantheonP_SH0ES", "DESY5", "Union3")

                    if obs_type in ("BAO", "DESI_DR1", "DESI_DR2"):
                        if overlay_flag:
                            order = np.argsort(np.asarray(x_dat))
                            ax.plot(
                                np.asarray(x_dat)[order],
                                np.asarray(y_mod_pts)[order],
                                color=MODEL_COLOR, linestyle="-", linewidth=1.2, alpha=0.8,
                                zorder=min(z_obs + 1, Z_BAND - 1),
                            )
                    elif obs_type not in SNE_TYPES:
                        order = np.argsort(np.asarray(x_dat))
                        ax.plot(
                            np.asarray(x_dat)[order],
                            np.asarray(y_mod_pts)[order],
                            color=MODEL_COLOR, linestyle="-", linewidth=1.2, alpha=0.8,
                            zorder=min(z_obs + 1, Z_BAND - 1),
                        )

                    # Residuals
                    res = np.asarray(y_dat) - np.asarray(y_mod_pts)
                    axr.errorbar(
                        x_dat, res, yerr=y_err, fmt="o", ms=3.0, lw=1.0,
                        color=color, ecolor=color, capsize=0, alpha=0.95, zorder=z_obs,
                    )

                unit = residual_unit(rep_type if not is_bao_desi else "BAO")
                axr.set_ylabel("Residual: Data − Model" + (f"\n{unit}" if unit else ""))

                # Legend: model first, then datasets (deduped)
                handles, labels_here = _model_first_legend(ax, model_disp)

                SNE_TYPES = ("JLA", "Pantheon", "PantheonP", "PantheonP_SH0ES", "DESY5", "Union3")
                if all(o in SNE_TYPES for o in part_sorted):
                    legend_loc = "lower right"
                    legend_anchor = (0.98, 0.02)
                else:
                    legend_loc = PLOT_SETTINGS.get("legend_loc", "upper left")
                    legend_anchor = PLOT_SETTINGS.get("legend_bbox_anchor", (0.02, 0.98))

                leg = ax.legend(
                    handles, labels_here,
                    loc=legend_loc,
                    bbox_to_anchor=legend_anchor,
                    fontsize=PLOT_SETTINGS.get("legend_font_size", 10),
                    frameon=True, framealpha=0.8,
                )
                ax.add_artist(leg)

                ax.tick_params(labelbottom=False)

            save_figure(fig, model_name, obs_key, "bestfit", PLOT_SETTINGS)
            plt.close(fig)



# =============================================================================
# CMB spectra plot (best-fit), stacked TT/TE/EE with residuals
# =============================================================================

def _plot_cmb_bestfit_spectra(model_name, obs_key, param_med, PLOT_SETTINGS):
    """CMB TT/TE/EE spectra + residuals (stacked), for best-fit parameters.

    Saves to: .../best_fits/<model_name>/<obs_key>/spectra.png
    """
    # Load bandpowers
    files = PLOT_SETTINGS.get("cmb_bandpower_files") or {}
    if not files:
        files = {k: PLOT_SETTINGS.get(k) for k in ("TT", "TE", "EE") if PLOT_SETTINGS.get(k)}
    have = {}
    for name in ("TT", "TE", "EE"):
        p = files.get(name)
        if p and os.path.exists(p):
            have[name] = read_bandpowers(p)
    if not have:
        print("[CMB plot] No TT/TE/EE bandpowers; skipping.")
        return

    # ℓ window from obs_key
    key = (obs_key or "").lower()
    if   "lowl" in key: lo, hi = 2, 30
    elif "hil"  in key:
        cap = int(PLOT_SETTINGS.get("cmb_ell_max_plot", 2500))
        lo, hi = 30, min(2500, cap)
    else:
        cap = int(PLOT_SETTINGS.get("cmb_ell_max_plot", 2500))
        ell_max_data = max(int(ells.max()) for (ells, _, _) in have.values())
        lo, hi = 2, min(ell_max_data, cap)

    # Theory (Dl) on common ℓ grid
    Lmax = int(hi)
    ell  = np.arange(2, Lmax + 1)
    fac  = ell * (ell + 1) / (2.0 * np.pi) * SP.TCMB2

    def _Dl(cl_dict, k):
        if cl_dict is None:
            return None
        arr = cl_dict.get(k)
        if arr is None:
            return None
        arr = np.asarray(arr)
        return fac * arr[ell] if arr.size > ell.max() else None

    if "lowl" in key:
        cls  = SP._compute_cls_cached(param_med, Lmax, mode="lowl")
        theo = {"TT": None, "TE": None, "EE": _Dl(cls, "ee")}
    else:
        cls  = SP._compute_cls_cached(param_med, Lmax, mode="hil")
        theo = {"TT": _Dl(cls, "tt"), "TE": _Dl(cls, "te"), "EE": _Dl(cls, "ee")}

    # Optional ±1σ band from chain endpoints
    band_low = band_high = None
    if "cmb_params_band" in PLOT_SETTINGS:
        p_lo, p_hi = PLOT_SETTINGS["cmb_params_band"]
        # Match the band computation mode (lowl vs hil) to the current key
        _band_mode = "lowl" if ("lowl" in key) else "hil"
        cl_lo = SP._compute_cls_cached(p_lo, Lmax, mode=_band_mode)
        cl_hi = SP._compute_cls_cached(p_hi, Lmax, mode=_band_mode)
        band_low  = {"TT": _Dl(cl_lo, "tt"), "TE": _Dl(cl_lo, "te"), "EE": _Dl(cl_lo, "ee")}
        band_high = {"TT": _Dl(cl_hi, "tt"), "TE": _Dl(cl_hi, "te"), "EE": _Dl(cl_hi, "ee")}

    # Decide which spectra components to plot based on the observation key.
    # Default: all TT/TE/EE bandpowers that are available.
    key = (obs_key or "").lower()

    # Decide which spectra components to plot based on the observation key.
    allowed = None
    if "lowl" in key and "hil" not in key:
        # Planck SimAll EE: EE-only likelihood
        allowed = ("EE",)
    elif "hil_tt" in key:
        # High-ℓ TT-only
        allowed = ("TT",)
    elif "hil" in key:
        # Full high-ℓ TTTEEE
        allowed = ("TT", "TE", "EE")

    if allowed is None:
        comps = [c for c in ("TT", "TE", "EE") if c in have]
    else:
        comps = [c for c in allowed if c in have]

    if not comps:
        print(f"[CMB plot] Nothing to draw for {obs_key} (no matching TT/TE/EE bandpowers present).")
        return

    fig = plt.figure(figsize=(10.4, 8.8))
    outer = fig.add_gridspec(nrows=len(comps), ncols=1, hspace=0.28)

    main_axes, res_axes, share_ref = [], [], None

    for i, comp in enumerate(comps):
        sub = outer[i].subgridspec(
            nrows=2,
            ncols=1,
            height_ratios=(3.0, 1.2),  # main panel : residuals
            hspace=0.0,
        )
        ax  = fig.add_subplot(sub[0, 0], sharex=share_ref)
        axr = fig.add_subplot(sub[1, 0], sharex=ax)
        if share_ref is None:
            share_ref = ax
        main_axes.append(ax)
        res_axes.append(axr)

        # Model band
        mm = (ell >= lo) & (ell <= hi)
        if band_low and band_high and (band_low.get(comp) is not None) and (band_high.get(comp) is not None):
            ax.fill_between(ell[mm], band_low[comp][mm], band_high[comp][mm], color="0.85", alpha=0.2, zorder=1, linewidth=0, label="Model Uncertainty")

        # Model line
        y_model = theo[comp]
        if y_model is not None:
            ax.plot(ell[mm], y_model[mm], color="#d62728", lw=1.9, label=f"{model_name} (median)", zorder=10)

        # Data
        l_dat, D_dat, yerr = have[comp]
        m = (l_dat >= lo) & (l_dat <= hi)
        if np.any(m):
            ax.errorbar(
                l_dat[m], D_dat[m],
                yerr=None if yerr is None else (yerr[:, m] if (np.ndim(yerr) == 2 and yerr.shape[0] == 2) else yerr[m]),
                fmt=".", ms=2.8, lw=0.7, color="#1f77b4", ecolor="#1f77b4", label=f"{comp} data", zorder = 6
            )

        ax.set_ylabel(rf"$D_\ell^{{{comp}}}$ [$\mu K^2$]")
        ax.grid(False)

        # Residuals
        if np.any(m) and y_model is not None:
            if yerr is None:
                err_resid = None
            elif np.ndim(yerr) == 2 and yerr.shape[0] == 2:
                err_resid = 0.5 * (np.abs(yerr[0, m]) + np.abs(yerr[1, m]))
            else:
                err_resid = yerr[m]
            model_on_data = np.interp(l_dat[m], ell, y_model)
            resid = D_dat[m] - model_on_data
            axr.axhline(0.0, color="0.25", lw=0.9)
            axr.errorbar(l_dat[m], resid, yerr=err_resid, fmt=".", ms=2.6, lw=0.7, color="#1f77b4", ecolor="#1f77b4")

        axr.set_ylabel(r"$\Delta D_\ell$ [$\mu K^2$]")
        axr.grid(False)
        if i < len(comps) - 1:
            axr.tick_params(labelbottom=False)

        # Legend to the right (outside axes)
        handles, labels_here = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels_here,
                loc="upper right",
                frameon=True,
                facecolor="white",
                edgecolor="0.3",
                framealpha=0.9,
                fontsize=9,
            )

    res_axes[-1].set_xlabel(r"$\ell$")
    #fig.suptitle(f"CMB spectra best-fit -- {model_name} [{obs_key}]  ($\\ell\\in[{lo},{hi}]$)" if PLOT_SETTINGS.get("latex_enabled", False)
     #            else f"CMB spectra best-fit -- {model_name} [{obs_key}]  (l in [{lo},{hi}])")

    save_figure(fig, model_name, obs_key, "spectra", PLOT_SETTINGS)
    plt.close(fig)
