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

            - Normal behaviour: exact match on "A+B" or "A_B".
            - Special case: the resolved label "PantheonP_SH0ES" maps to the
              PantheonPS row (the Pantheon+SH0ES data set).
            """
            joined_plus = "+".join(entry)
            joined_ud   = "_".join(entry)

            # Exact matches first
            if key == joined_plus or key == joined_ud:
                return True

            # Special alias: old key "PantheonP_SH0ES" corresponds to PantheonPS
            if key == "PantheonP_SH0ES" and joined_plus == "PantheonPS":
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


def statistical_analysis(best_fit_values, data, CONFIG, true_model):
    """
    Perform statistical analysis for all models and observation combinations,
    and calculate delta AIC/BIC relative to the true model.

    This processes each observation set individually by pairing it with
    its corresponding observation type from CONFIG.
    """
    results: dict[str, dict[str, dict[str, float]]] = {}
    reference_aic: dict[str, float] = {}
    reference_bic: dict[str, float] = {}

    for model_name, obs_results in best_fit_values.items():
        results[model_name] = {}
        for obs_name, params in obs_results.items():
            # Extract best-fit (median) values into a dictionary.
            param_dict = {param: values[0] for param, values in params.items()}
            num_params = len(param_dict)
            notes: list[str] = []

            # Recover the full observation list that corresponds to this best-fit key.
            obs_entry = None
            for j, obs_list in enumerate(CONFIG[model_name]["observations"]):
                key_j = U.generate_label(
                    obs_list, config_model=CONFIG[model_name], obs_index=j
                )
                if key_j == obs_name:
                    obs_entry = obs_list
                    obs_index = j
                    break
            if obs_entry is None:
                raise ValueError(
                    f"Observation {obs_name} not found in CONFIG for model {model_name}."
                )

            chi_squared_total = 0.0
            num_data_points_total = 0

            # Get the model function once for this model.
            MODEL_func = UDM.Get_model_function(model_name)
            # Get the list of observation types for this observation set.
            obs_types = CONFIG[model_name]["observation_types"][obs_index]

            # Loop over each individual observation in the set
            for i, obs in enumerate(obs_entry):
                obs_type = obs_types[i]
                obs_data = data.get(obs)
                if not obs_data:
                    raise ValueError(f"Observation data for {obs} not found.")

                # Handle Pantheon+ (with and without SH0ES) using the dedicated chi^2
                if obs in ("PantheonP", "PantheonPS"):
                    zHD = obs_data["zHD"]
                    m_b_corr = obs_data["m_b_corr"]
                    IS_CALIBRATOR = obs_data["IS_CALIBRATOR"]
                    CEPH_DIST = obs_data["CEPH_DIST"]

                    # Prefer precomputed cov (lower Cholesky). If missing, compute robustly now.
                    if "cov" not in obs_data:
                        L = U.compute_pantheon_cov(
                            data,  # full data dict (has indices/mask)
                            CONFIG[model_name],  # model-specific config
                            comm=None,           # here we don't MPI-broadcast
                            rank=0,
                            cov_file=obs_data["cov_path"],
                        )
                        obs_data["cov"] = L
                    cov = obs_data["cov"]

                    comoving_distances = UDM.Comoving_distance_vectorized(
                        MODEL_func, zHD, param_dict
                    )
                    distance_modulus = 25 + 5 * np.log10(comoving_distances * (1 + zHD))

                    chi_squared = Calc_PantP_chi(
                        m_b_corr, IS_CALIBRATOR, CEPH_DIST, cov, distance_modulus, param_dict
                    )
                    num_data_points_total += len(m_b_corr)

                elif obs == "BAO":
                    p = dict(param_dict)
                    chi_squared = Calc_BAO_chi(obs_data, MODEL_func, p, "BAO")
                    num_data_points_total += len(obs_data["covd1"])

                elif obs in ("DESI_DR1", "DESI_DR2"):
                    p = dict(param_dict)
                    calibrated = any(
                        ("BBN" in x) or ("CMB" in x) or ("THETA" in x) for x in obs_entry
                    )
                    # Let DESI chi2 know if it's being run with BBN etc.
                    type_tag = obs + ("+BBN" if calibrated else "")
                    chi_squared = Calc_DESI_chi(obs_data, MODEL_func, p, type_tag)
                    num_data_points_total += len(obs_data["redshift"])

                elif obs_type == "SNe":
                    redshift = obs_data["redshift"]
                    type_data = obs_data["type_data"]
                    type_data_error = obs_data["type_data_error"]
                    comoving_distances = UDM.Comoving_distance_vectorized(
                        MODEL_func, redshift, param_dict
                    )
                    model_val = 25 + 5 * np.log10(comoving_distances * (1 + redshift))
                    chi_squared = Calc_chi(
                        obs_type, type_data, type_data_error, model_val
                    )
                    num_data_points_total += len(type_data)

                elif obs_type in ["OHD", "CC"]:
                    redshift = obs_data["redshift"]
                    type_data = obs_data["type_data"]
                    type_data_error = obs_data["type_data_error"]
                    model_val = param_dict["H_0"] * np.array(
                        [MODEL_func(z, param_dict) for z in redshift]
                    )
                    chi_squared = Calc_chi(
                        obs_type, type_data, type_data_error, model_val
                    )
                    num_data_points_total += len(type_data)

                elif obs_type in ["f", "f_sigma_8"]:
                    redshift = obs_data["redshift"]
                    type_data = obs_data["type_data"]
                    type_data_error = obs_data["type_data_error"]

                    if obs_type == "f_sigma_8":
                        gamma = _resolve_gamma_for_obs(
                            model_name,
                            obs_name,
                            CONFIG,
                            param_dict,
                            default_gamma=GAMMA_FS8_SINGLETON,
                        )
                        gamma_was_sampled = "gamma" in param_dict
                        if not gamma_was_sampled:
                            notes.append(f"γ fixed to {gamma:.3f} (fσ8-only)")
                        Omega_z = UDM.matter_density_z_array(
                            redshift, param_dict, MODEL_func
                        )
                        I = UDM.integral_term_array(
                            redshift, param_dict, MODEL_func, gamma
                        )
                        model_val = float(param_dict["sigma_8"]) * (Omega_z**gamma) * np.exp(
                            -I
                        )
                    else:  # "f"
                        model_val = UDM.matter_density_z_array(
                            redshift, param_dict, MODEL_func
                        ) ** float(param_dict["gamma"])

                    chi_squared = Calc_chi(
                        obs_type, type_data, type_data_error, model_val
                    )
                    num_data_points_total += len(type_data)

                elif obs_type in ("BBN_DH", "BBN_DH_AlterBBN"):
                    mode = obs_data.get("mode", "mean")
                    if mode == "mean":
                        num_data_points_total += 1
                    else:
                        num_data_points_total += len(obs_data.get("systems", []))
                    chi_squared = Calc_BBN_DH_chi(obs_data, MODEL_func, param_dict, obs_type)

                elif obs in ("BBN_PryMordial", "BBN_prior"):
                    # Gaussian prior on Ω_b h^2 (no redshifted data points).
                    # Used in the sampler; for the *report*, we skip adding data χ² and N.
                    try:
                        mu = float(data[obs]["mu_obh2"])
                        sig = float(data[obs]["sigma_obh2"])
                        x = float(param_dict["Omega_bh^2"])
                        pull = (x - mu) / sig
                        # Optionally record pull in a table if desired.
                        _ = pull  # silence linters if unused
                    except Exception:
                        pass
                    continue

                elif obs_type == "CMB":
                    # Use Statistical_packages loglike functions; convert to chi^2 = -2 ln L
                    if obs == "CMB_lowl":
                        # Planck SimAll EE low-ℓ
                        chi_squared = -2.0 * SP.cmb_lowl_loglike(param_dict, model_name)
                        # SimAll EE has 30 low-ℓ points (ℓ = 2..29)
                        num_data_points_total += 30

                    elif obs == "CMB_hil":
                        # Planck high-ℓ TTTEEE (Plik)
                        like = SP._get_hil_like()
                        try:
                            raw_lmax = like.get_lmax()
                        except Exception:
                            raw_lmax = [2508, 0, 0, 0]

                        # raw_lmax might be dict or a sequence
                        if isinstance(raw_lmax, dict):
                            Ltt = int(raw_lmax.get("tt") or raw_lmax.get("TT") or 0)
                            Lee = int(raw_lmax.get("ee") or raw_lmax.get("EE") or 0)
                            Lte = int(raw_lmax.get("te") or raw_lmax.get("TE") or 0)
                        else:
                            L_vals = list(map(int, raw_lmax)) + [0, 0, 0, 0]
                            Ltt, Lee, _, Lte = L_vals[:4]

                        npts = max(Ltt - 1, 0) + max(Lee - 1, 0) + max(Lte - 1, 0)
                        chi_squared = -2.0 * SP.cmb_hil_loglike(param_dict, model_name)
                        num_data_points_total += npts

                    elif obs == "CMB_hil_TT":
                        # TT-only high-ℓ (Plik TT)
                        like = SP._get_hilTT_like()
                        try:
                            raw_lmax = like.get_lmax()
                        except Exception:
                            raw_lmax = 2508

                        if isinstance(raw_lmax, dict):
                            lTT = int(raw_lmax.get("tt") or raw_lmax.get("TT") or next(iter(raw_lmax.values())))
                        elif isinstance(raw_lmax, (list, tuple, np.ndarray)):
                            lTT = int(raw_lmax[0])
                        else:
                            lTT = int(raw_lmax)

                        loglike = SP.cmb_hilTT_loglike(param_dict, model_name)
                        chi_squared = -2.0 * float(loglike)

                        # Roughly count TT data points (ℓ=2..lTT → lTT-1)
                        num_data_points_total += max(lTT - 1, 0)

                    elif obs == "CMB_lensing":
                        has_primary_cmb = any(
                            x in {"CMB_hil", "CMB_hil_TT", "CMB_lowl"} for x in obs_entry
                        )
                        SP.set_lensing_mode(
                            "raw" if has_primary_cmb else "cmbmarged"
                        )
                        like = SP._get_lensing_like()

                        # discover bins...
                        n_bins = 8
                        try:
                            if hasattr(like, "get_lensing_nbins"):
                                n_bins = int(like.get_lensing_nbins())
                            elif hasattr(like, "get_lensing_bins"):
                                n_bins = len(like.get_lensing_bins())
                        except Exception:
                            pass

                        logL, p_used, note = _stats_lensing_logL_safe(
                            param_dict, model_name
                        )
                        if not (isfinite(logL) and abs(logL) < 1e9):
                            logger.error(
                                "Lensing logL looks invalid — skipping in stats"
                            )
                            if note:
                                notes.append(note)
                            continue

                        if note:
                            notes.append(note)

                        chi_squared = -2.0 * logL
                        num_data_points_total += n_bins

                    else:
                        raise ValueError(f"Unsupported CMB observation: {obs}")

                else:
                    raise ValueError(f"Unsupported observation type: {obs_type}")

                chi_squared_total += float(chi_squared)

            if num_data_points_total <= 0:
                logger.error(
                    "No valid data points contributed to stats; "
                    "skipping stats for %s.",
                    obs_name,
                )
                continue

            log_likelihood = -0.5 * chi_squared_total

            if obs_entry == ["PantheonP"]:
                n_data = num_data_points_total
                n_param = len(CONFIG[model_name]["parameters"][obs_index])  # should be 3
                dof = n_data - n_param
            else:
                dof = num_data_points_total - num_params

            if dof <= 0:
                raise ValueError(
                    "Degrees of freedom (DOF) is zero or negative. "
                    "Check your model or dataset."
                )

            reduced_chi_squared = chi_squared_total / dof
            aic = 2 * num_params - 2 * log_likelihood
            bic = num_params * np.log(num_data_points_total) - 2 * log_likelihood

            results[model_name][obs_name] = {
                "Log-Likelihood": log_likelihood,
                "Chi_squared": chi_squared_total,
                "Reduced_Chi_squared": reduced_chi_squared,
                "AIC": aic,
                "BIC": bic,
            }
            if notes:
                results[model_name][obs_name]["Note"] = " | ".join(notes)

            if model_name == true_model:
                reference_aic[obs_name] = aic
                reference_bic[obs_name] = bic

    # Calculate delta AIC and delta BIC relative to the true model.
    for model_name, obs_results in results.items():
        for obs_name, stats in obs_results.items():
            stats["dAIC"] = stats["AIC"] - reference_aic.get(obs_name, stats["AIC"])
            stats["dBIC"] = stats["BIC"] - reference_bic.get(obs_name, stats["BIC"])

    return results


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
