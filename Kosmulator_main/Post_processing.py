#!/usr/bin/env python3
"""
High-level statistical tools for Kosmulator.

This module depends on:
  - User_defined_modules (UDM) for cosmology E(z), distances, etc.
  - Statistical_packages for low-level χ² / likelihood routines.
  - utils for Pantheon+ covariance helper.
  - Class_run for CMB (Planck) likelihood evaluation.

It provides:
  - calculate_asymmetric_from_samples
  - statistical_analysis
  - provide_model_diagnostics
  - interpret_delta_aic_bic
(plus small helpers like _scalarize, _format_pm, _resolve_gamma_for_obs).
"""

from __future__ import annotations

from math import isfinite
from typing import Any  # Dict unused; drop if you like

import logging

import numpy as np
from scipy.optimize import minimize

import User_defined_modules as UDM
from Kosmulator_main import utils as U
from Kosmulator_main.constants import GAMMA_FS8_SINGLETON
from Kosmulator_main import Class_run as CR

from Kosmulator_main.Statistical_packages import (
    Calc_PantP_chi,
    Calc_BAO_chi,
    Calc_DESI_chi,
    Calc_chi,
    Calc_BBN_DH_chi,
    Calc_Generic_SNe_chi,
)
from Kosmulator_main import Statistical_packages as SP

logger = logging.getLogger("Kosmulator.Post_processing")


def _scalarize(x: Any) -> float:
    """
    Robustly convert a scalar / array / sequence into a single float.

    - If x is None or empty → NaN.
    - If x is an array → mean of finite entries, or NaN if none finite.
    """
    if x is None:
        return float("nan")
    try:
        arr = np.asarray(x, dtype=float)
    except Exception:
        return float("nan")
    if arr.size == 0:
        return float("nan")
    if arr.ndim == 0:
        return float(arr)
    finite = arr[np.isfinite(arr)]
    return float("nan") if finite.size == 0 else float(np.mean(finite))


def _format_pm(value, minus, plus):
    # ... whatever you already have before computing errors ...
    lo = abs(minus)
    hi = abs(plus)
    err = max(lo, hi)

    # Force FOUR decimal places everywhere
    prec = 4

    fmt = f"{{:.{prec}f}}"
    v_str  = fmt.format(value)
    lo_str = fmt.format(lo)
    hi_str = fmt.format(hi)
    return rf"${v_str}^{{+{hi_str}}}_{{-{lo_str}}}$"


def calculate_asymmetric_from_samples(samples, parameters, observations):
    """
    Calculate median, upper, and lower uncertainties from MCMC samples.

    Returns:
      results: dict[obs][param] → {"median", "lower_error", "upper_error"}
      latex_table: list of rows with LaTeX-formatted strings
      structured_values: dict[obs][param] → [median, median+upper, median-lower]
    """
    results, latex_table, structured_values = {}, [], {}

    for obs, obs_samples in samples.items():
        results[obs], structured_values[obs], row = {}, {}, []

        # Match the observation key to its corresponding parameter list.
        # Accept the raw joined name ("PantheonP"), underscore join ("PantheonP"),
        # and resolved variants like "PantheonP_SH0ES" (which now map to PantheonPS).
        def _matches(entry, key: str) -> bool:
            """
            Decide whether a sample key corresponds to a given observation entry.

            Accept:
              - "A+B" or "A_B"
              - Pantheon aliasing:
                  PantheonP_SH0ES  <-> PantheonPS
                  Pantheon+SH0ES   -> PantheonPS   (defensive; should be display-only)
              - Works inside combined keys too (e.g., "CC+PantheonP_SH0ES").
            """
            joined_plus = "+".join(entry)
            joined_ud   = "_".join(entry)

            def _norm(s: str) -> str:
                return (
                    str(s)
                    .replace("PantheonP_SH0ES", "PantheonPS")
                    .replace("Pantheon+SH0ES", "PantheonPS")
                )

            k = _norm(key)

            # Compare against both join styles, with normalization on both sides
            if k == _norm(joined_plus) or k == _norm(joined_ud):
                return True

            # Also allow +/- separator swaps after normalization
            if k.replace("+", "_") == _norm(joined_ud):
                return True
            if k.replace("_", "+") == _norm(joined_plus):
                return True

            return False

        obs_list = next((entry for entry in observations if _matches(entry, obs)), None)
        if obs_list is None:
            raise ValueError(f"Observation '{obs}' not found in observations.")

        # Fetch the corresponding parameter list
        param_index = observations.index(obs_list)
        obs_param_names = parameters[param_index]

        # Iterate over the parameters for this observation
        for param_index, param in enumerate(obs_param_names):
            # Check that the parameter index is within bounds of obs_samples
            if param_index < obs_samples.shape[1]:
                param_samples = obs_samples[:, param_index]

                # Calculate percentiles
                p16, p50, p84 = np.percentile(param_samples, [16, 50, 84])
                median = p50
                lower_error = p50 - p16
                upper_error = p84 - p50

                # Add to results
                results[obs][param] = {
                    "median": round(median, 3),
                    "lower_error": round(lower_error, 3),
                    "upper_error": round(upper_error, 3),
                }

                # Add LaTeX-formatted string for the table
                row.append(_format_pm(median, lower_error, upper_error))

                # Add structured values for the parameter
                structured_values[obs][param] = [
                    round(median, 3),
                    round(median + upper_error, 3),
                    round(median - lower_error, 3),
                ]
            else:
                print(
                    f"Warning: Parameter '{param}' index ({param_index}) "
                    "exceeds sample dimensions."
                )

        # Add the row to the LaTeX table
        latex_table.append(row)

    return results, latex_table, structured_values


def _stats_lensing_logL_safe(
    p0: dict, model_name: str
) -> tuple[float, dict, str | None]:
    """
    Try lensing logL at p0. If invalid, run a tiny Nelder–Mead polish in
    (ln10^10_As, tau_reio). Returns (logL, p_used, note).
    """
    # Use the *actual* lensing loglike from Statistical_packages, not Class_run
    logL = SP.cmb_lensing_loglike(p0, model_name)
    if isfinite(logL) and abs(logL) < 1e9:
        return logL, p0, None

    if ("ln10^10_As" not in p0) or ("tau_reio" not in p0):
        return (
            float("nan"),
            p0,
            "Lensing failed at summary point; no polish variables available.",
        )

    x0 = np.array([p0["ln10^10_As"], p0["tau_reio"]], dtype=float)

    def objective(x):
        p = dict(p0)
        p["ln10^10_As"], p["tau_reio"] = float(x[0]), float(x[1])
        v = SP.cmb_lensing_loglike(p, model_name)
        # maximize logL → minimize -logL; penalize invalid evaluations
        if (not isfinite(v)) or (abs(v) > 1e9):
            return 1e12
        return -float(v)

    # Small simplex, few iters (fast)
    res = minimize(
        objective,
        x0,
        method="Nelder–Mead",
        options={"maxiter": 60, "xatol": 1e-3, "fatol": 1e-3, "disp": False},
    )

    if (res.success is True) and isfinite(res.fun):
        p_best = dict(p0)
        p_best["ln10^10_As"], p_best["tau_reio"] = float(res.x[0]), float(res.x[1])
        # objective = -logL
        logL2 = -float(res.fun)
        if isfinite(logL2) and abs(logL2) < 1e9:
            return (
                logL2,
                p_best,
                "Lensing evaluated after a small polish in (ln10^10_As, τ).",
            )

    return (
        float("nan"),
        p0,
        "Lensing failed at summary point; polish did not find a valid nearby point.",
    )



def _resolve_gamma_for_obs(
    model_name: str,
    obs_key: str,
    CONFIG: dict,
    param_dict: dict,
    default_gamma: float = GAMMA_FS8_SINGLETON,
) -> float:
    """
    Return gamma for this observation set:
      - If sampled, read from param_dict["gamma"].
      - Else, if CONFIG recorded a fixed value for this group, use it.
      - Else fall back to default_gamma (≈ GR value).
    """
    # If gamma is in the posterior medians, just use it
    if "gamma" in param_dict:
        try:
            return float(param_dict["gamma"])
        except Exception:
            pass

    cfg = CONFIG.get(model_name, {})
    obs_index = next(
        (
            i
            for i, o in enumerate(cfg.get("observations", []))
            if "+".join(o) == obs_key or "_".join(o) == obs_key
        ),
        None,
    )
    if obs_index is not None:
        fixed_map = cfg.get("fs8_gamma_fixed_by_group", {})
        if obs_index in fixed_map:
            try:
                return float(fixed_map[obs_index])
            except Exception:
                pass

    # Final fallback: GR-like default
    return float(default_gamma)


def statistical_analysis(best_fit_values, data, CONFIG, true_model,
                         posterior_samples=None):
    """Compute statistics from retained samples, independently of plot medians.

    The original positional argument is retained for call-site compatibility;
    posterior_samples is required because rounded medians cannot define DIC.
    """
    if posterior_samples is None:
        raise ValueError("statistical_analysis requires posterior_samples; "
                         "pass the unrounded post-burn sample dictionary")
    from Kosmulator_main.Model_comparison import analyse_models
    return analyse_models(posterior_samples, data, CONFIG, true_model)


def provide_model_diagnostics(
    reduced_chi_squared, model_name: str = "", reference_chi_squared=None
) -> str:
    """
    Provide a quick diagnostic description of the model's performance.
    """
    # Make it robust to arrays/NaNs and avoid any external state.
    reduced_chi_squared = _scalarize(reduced_chi_squared)
    reference_chi_squared = _scalarize(reference_chi_squared)
    feedback = ""

    # Statistical Interpretation
    feedback += "Statistical Interpretation:\n"
    if 0.9 <= reduced_chi_squared <= 1.1:
        feedback += (
            "  - The model appears to fit the data very well. The reduced chi-squared is close to 1, "
            "indicating the residuals are consistent with the uncertainties.\n"
        )
    elif 0.5 <= reduced_chi_squared < 0.9:
        feedback += (
            "  - The reduced chi-squared is slightly below 1. This could indicate overfitting, "
            "or that the data uncertainties may be overestimated.\n"
        )
    elif reduced_chi_squared < 0.5:
        feedback += (
            "  - The reduced chi-squared is significantly below 1. This suggests possible overfitting "
            "or overly conservative error bars.\n"
        )
    elif 1.1 < reduced_chi_squared <= 3.0:
        feedback += (
            "  - The reduced chi-squared is above 1, but within an acceptable range. This indicates a "
            "reasonable fit, though there might be room for improvement in the model or data uncertainties.\n"
        )
    else:
        feedback += (
            "  - The reduced chi-squared is significantly above 3. This suggests the model does not fit "
            "the data well. Consider revising your model or checking for systematic errors in the data.\n"
        )

    # Benchmark Approach: Only applies to non-LCDM models
    if reference_chi_squared is not None and model_name.lower() != "lcdm":
        feedback += "\nBenchmark Comparison (Relative to LCDM):\n"
        if reduced_chi_squared < reference_chi_squared:
            feedback += (
                f"  - This model's reduced chi-squared ({reduced_chi_squared:.2f}) is lower than the "
                f"benchmark LCDM value ({reference_chi_squared:.2f}).\n"
            )
            feedback += (
                "    This could indicate overfitting or that uncertainties are playing a significant role.\n"
            )
        elif reduced_chi_squared > reference_chi_squared:
            feedback +=(
                f"  - This model's reduced chi-squared ({reduced_chi_squared:.2f}) is higher than the "
                f"benchmark LCDM value ({reference_chi_squared:.2f}).\n"
            )
            feedback += (
                "    This may suggest underfitting or that the model does not capture the data as well as LCDM.\n"
            )
        else:
            feedback += (
                f"  - This model's reduced chi-squared matches the benchmark LCDM value "
                f"({reference_chi_squared:.2f}), suggesting a comparable fit.\n"
            )

    # Special case for LCDM
    if model_name.lower() == "lcdm":
        feedback += (
            "\nThe LCDM model is widely regarded as a robust and well-tested benchmark model. "
            "It is recommended when comparing to other models to compare their reduced chi-squared "
            "values to the LCDM model's to determine whether over- or under-fitting happened "
            "irregardless of the uncertainties in the observations themselves.\n"
        )

    return feedback


def interpret_delta_aic_bic(delta_aic, delta_bic) -> str:
    """
    Turn ΔAIC / ΔBIC into human-readable model-comparison statements.
    """
    # Coerce to plain floats (works for Python floats, NumPy scalars, 0-d arrays)
    delta_aic = float(np.asarray(delta_aic).reshape(()))
    delta_bic = float(np.asarray(delta_bic).reshape(()))

    feedback = []

    # --- AIC ---
    if delta_aic < 2:
        feedback.append(f"Delta AIC: Indistinguishable (ΔAIC = {delta_aic:.2f}).")
    elif delta_aic < 4:
        feedback.append(
            f"Delta AIC: Slight evidence against the model (ΔAIC = {delta_aic:.2f})."
        )
    elif delta_aic < 7:
        feedback.append(
            f"Delta AIC: Positive evidence against the model (ΔAIC = {delta_aic:.2f})."
        )
    else:
        feedback.append(
            f"Delta AIC: Strong evidence against the model (ΔAIC = {delta_aic:.2f})."
        )

    # --- BIC ---
    if delta_bic < 2:
        feedback.append(f"Delta BIC: Indistinguishable (ΔBIC = {delta_bic:.2f}).")
    elif delta_bic < 6:
        feedback.append(
            f"Delta BIC: Weak evidence against the model (ΔBIC = {delta_bic:.2f})."
        )
    elif delta_bic < 10:
        feedback.append(
            f"Delta BIC: Moderate evidence against the model (ΔBIC = {delta_bic:.2f})."
        )
    else:
        feedback.append(
            f"Delta BIC: Strong evidence against the model (ΔBIC = {delta_bic:.2f})."
        )

    return "\n".join(feedback)
