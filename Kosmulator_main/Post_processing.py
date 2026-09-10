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

    for obs, obs_data_input in samples.items():
        results[obs], structured_values[obs], row = {}, {}, []

        #Unpack the dictionary data and find the Maximum likelihood
        if isinstance(obs_data_input, dict):
            obs_samples = obs_data_input["samples"]
            log_like = obs_data_input.get("loglike")
        else:
            obs_samples = obs_data_input
            log_like = None

        if log_like is not None:
            ll_arr = np.asarray(log_like).ravel()
            finite = np.isfinite(ll_arr) & np.all(np.isfinite(obs_samples), axis=1)
            valid_ll = ll_arr[finite]
            valid_samples = obs_samples[finite]
            dev_samples = -2.0 * valid_ll

            mle_idx = np.nanargmax(valid_ll)
            mle_vector = valid_samples[mle_idx]

            #For DIC calculations
            D_bar = float(np.nanmean(dev_samples))
            D_var = float(4.0 * np.nanvar(valid_ll, ddof = 1))
            theta_bar = np.nanmean(valid_samples, axis = 0)

            # DEBUG — diagnose D_bar vs D_hat gap
            print(f"[DEBUG DEVIANCE] obs={obs}")
            print(f"[DEBUG DEVIANCE]   n_samples = {dev_samples.size}")
            print(f"[DEBUG DEVIANCE]   min={np.nanmin(dev_samples):.3f}  max={np.nanmax(dev_samples):.3f}")
            print(f"[DEBUG DEVIANCE]   mean={D_bar:.3f}  median={np.nanmedian(dev_samples):.3f}  std={np.sqrt(D_var):.3f}")

            

        else:
            mle_vector = None
            D_bar = None
            D_var = None
    



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

                #Grab MLE for this parameter
                mle_val = mle_vector[param_index] if mle_vector is not None else median
                mean_val = theta_bar[param_index] if theta_bar is not None else median

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
                    mle_val,
                    mean_val,
                ]
            else:
                print(
                    f"Warning: Parameter '{param}' index ({param_index}) "
                    "exceeds sample dimensions."
                )

        if D_bar is not None:
            structured_values[obs]["__D_bar__"] = float(D_bar)
            structured_values[obs]["__D_var__"] = float(D_var)

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


#Function to optimise MLE values

def find_polished_mle(
    compute_chi2_fn,
    params_dict_median: dict[str, float],
    prior_bounds: dict[str, tuple[float, float]] | None = None,
    max_evals: int = 300,
) -> tuple[dict[str, float], float]:
    """
    Find the Maximum Likelihood Estimate (MLE) by polishing
    the posterior median with Nelder-Mead simplex minimization.
    """
    p_names = list(params_dict_median.keys())
    x0 = np.array([params_dict_median[p] for p in p_names], dtype=float)

    # Initial baseline evaluation at median
    baseline_chi2, _ = compute_chi2_fn(params_dict_median)
    if not (np.isfinite(baseline_chi2) and abs(baseline_chi2) < 1e9):
        baseline_chi2 = 1e12

    def objective(x_vec: np.ndarray) -> float:
        if prior_bounds:
            for idx, p in enumerate(p_names):
                if p in prior_bounds:
                    lo, hi = prior_bounds[p]
                    if not (lo <= x_vec[idx] <= hi):
                        return 1e12

        p_candidate = {p: float(x_vec[idx]) for idx, p in enumerate(p_names)}
        try:
            val, _ = compute_chi2_fn(p_candidate)
            if np.isfinite(val) and abs(val) < 1e9:
                return float(val)
        except Exception:
            pass
        return 1e12

    res = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": max_evals,
            "maxfev": max_evals,
            "xatol": 1e-3,
            "fatol": 1e-3,
            "disp": False,
        },
    )

    if res.success and np.isfinite(res.fun) and (res.fun <= baseline_chi2):
        best_p = {p: float(res.x[idx]) for idx, p in enumerate(p_names)}
        return best_p, float(res.fun)

    return params_dict_median, baseline_chi2


def statistical_analysis(best_fit_values, data, CONFIG, reference_model):
    """
    Perform statistical analysis for all models and observation combinations,
    and calculate delta AIC/BIC relative to the reference model.

    This processes each observation set individually by pairing it with
    its corresponding observation type from CONFIG.
    """
    results: dict[str, dict[str, dict[str, float]]] = {}
    reference_aic: dict[str, float] = {}
    reference_bic: dict[str, float] = {}
    reference_aicc: dict[str, float] = {}
    reference_dic: dict[str, float] = {}
    reference_waic: dict[str, float] = {}
    reference_chi: dict[str, float] = {}
    

    for model_name, obs_results in best_fit_values.items():
        results[model_name] = {}
        for obs_name, params in obs_results.items():
            # Extract best-fit (median) values into a dictionary.

            D_bar = params.pop("__D_bar__", None) #Popped before the loop iterates through param
            D_var = params.pop("__D_var__", None)


            #Returns the mle_values needed for AIC and BIC
            param_dict = {
                param: (values[3] if len(values) > 3 else values[0])
                for param, values in params.items()
            }

            #Returns the mean_vals needed for DIC
            param_dict_mean = {
                param: (values[4] if len(values) > 4 else values[0])
                for param, values in params.items()
            }
            num_params = len(param_dict)
            #DEBUG STATEMENT
            print(f"[DEBUG] model={model_name} obs={obs_name}")
            print(f"[DEBUG]   param_dict = {param_dict}")
            print(f"[DEBUG]   num_params = {num_params}")
            for p, v in params.items():
                print(f"[DEBUG]   raw values for {p}: {v}")
            notes: list[str] = []

            # Recover the full observation list that corresponds to this best-fit key.
            obs_entry = None
            for j, obs_list in enumerate(CONFIG[model_name]["observations"]):
                def _norm_obs_key(s: str) -> str:
                    # Normalize Pantheon aliasing first (so we don't destroy underscores inside tokens)
                    s = str(s).replace("PantheonP_SH0ES", "PantheonPS").replace("Pantheon+SH0ES", "PantheonPS")
                    # Normalize join style for comparison
                    s = s.replace("_", "+")
                    return s

                key_j = U.generate_label(obs_list, config_model=CONFIG[model_name], obs_index=j)
                if _norm_obs_key(key_j) == _norm_obs_key(obs_name):
                    obs_entry = obs_list
                    obs_index = j
                    break
            if obs_entry is None:
                raise ValueError(
                    f"Observation {obs_name} not found in CONFIG for model {model_name}."
                )
            
            # DEBUG
            print(f"[DEBUG] obs_entry = {obs_entry}")
            chi_squared_total = 0.0
            num_data_points_total = 0

            # Get the model function once for this model.
            MODEL_func = UDM.Get_model_function(model_name)
            # Get the list of observation types for this observation set.
            obs_types = CONFIG[model_name]["observation_types"][obs_index]

            # --- NESTED EVALUATOR FUNCTION ---
            def _compute_chi2_total(p_eval: dict) -> tuple[float, int]:
                chi_total = 0.0
                n_points = 0

                for i, obs in enumerate(obs_entry):
                    obs_type = obs_types[i]
                    obs_data = data.get(obs)
                    if not obs_data:
                        raise ValueError(f"Observation data for {obs} not found.")

                    if obs in ("PantheonP", "PantheonPS"):
                        zHD = obs_data["zHD"]
                        m_b_corr = obs_data["m_b_corr"]
                        IS_CALIBRATOR = obs_data["IS_CALIBRATOR"]
                        CEPH_DIST = obs_data["CEPH_DIST"]

                        if "cov" not in obs_data:
                            L = U.compute_pantheon_cov(data, CONFIG[model_name], comm=None, rank=0, cov_file=obs_data["cov_path"])
                            obs_data["cov"] = L
                        cov = obs_data["cov"]

                        comoving_distances = UDM.Comoving_distance_vectorized(MODEL_func, zHD, p_eval)
                        distance_modulus = 25 + 5 * np.log10(comoving_distances * (1 + zHD))
                        chi_total += float(Calc_PantP_chi(m_b_corr, IS_CALIBRATOR, CEPH_DIST, cov, distance_modulus, p_eval))
                        n_points += len(m_b_corr)

                    elif obs == "BAO":
                        chi_total += float(Calc_BAO_chi(obs_data, MODEL_func, dict(p_eval), "BAO"))
                        n_points += len(obs_data["covd1"])

                    elif obs in ("DESI_DR1", "DESI_DR2"):
                        calibrated = any(("BBN" in x) or ("CMB" in x) or ("THETA" in x) for x in obs_entry)
                        type_tag = obs + ("+BBN" if calibrated else "")
                        chi_total += float(Calc_DESI_chi(obs_data, MODEL_func, dict(p_eval), type_tag))
                        n_points += len(obs_data["redshift"])

                    elif obs_type == "SNe":
                        redshift = obs_data["redshift"]
                        comoving_distances = UDM.Comoving_distance_vectorized(MODEL_func, redshift, p_eval)
                        model_val = 25 + 5 * np.log10(comoving_distances * (1 + redshift))
                        chi_total += float(Calc_Generic_SNe_chi(obs_data=obs_data, model=model_val, param_dict=p_eval))
                        n_points += len(redshift)

                    elif obs_type in ["OHD", "CC"]:
                        redshift = obs_data["redshift"]
                        model_val = p_eval["H_0"] * np.array([MODEL_func(z, p_eval) for z in redshift])
                        chi_total += float(Calc_chi(obs_type, obs_data["type_data"], obs_data["type_data_error"], model_val))
                        n_points += len(obs_data["type_data"])

                    elif obs_type in ["f", "f_sigma_8"]:
                        redshift = obs_data["redshift"]
                        if obs_type == "f_sigma_8":
                            gamma = _resolve_gamma_for_obs(model_name, obs_name, CONFIG, p_eval, default_gamma=GAMMA_FS8_SINGLETON)
                            if "gamma" not in p_eval and p_eval is param_dict:
                                notes.append(f"γ fixed to {gamma:.3f} (fσ8-only)")
                            Omega_z = UDM.matter_density_z_array(redshift, p_eval, MODEL_func)
                            I = UDM.integral_term_array(redshift, p_eval, MODEL_func, gamma)
                            model_val = float(p_eval["sigma_8"]) * (Omega_z**gamma) * np.exp(-I)
                        else:
                            model_val = UDM.matter_density_z_array(redshift, p_eval, MODEL_func) ** float(p_eval["gamma"])

                        chi_total += float(Calc_chi(obs_type, obs_data["type_data"], obs_data["type_data_error"], model_val))
                        n_points += len(obs_data["type_data"])

                    elif obs_type in ("BBN_DH", "BBN_DH_AlterBBN"):
                        mode = obs_data.get("mode", "mean")
                        n_points += 1 if mode == "mean" else len(obs_data.get("systems", []))
                        chi_total += float(Calc_BBN_DH_chi(obs_data, MODEL_func, p_eval, obs_type))

                    elif obs in ("BBN_PryMordial", "BBN_prior"):
                        continue

                    elif obs_type == "CMB":
                        if obs == "CMB_lowl":
                            chi_total += float(-2.0 * SP.cmb_lowl_loglike(p_eval, model_name))
                            n_points += 30
                        elif obs == "CMB_hil":
                            like = SP._get_hil_like()
                            try:
                                raw_lmax = like.get_lmax()
                            except Exception:
                                raw_lmax = [2508, 0, 0, 0]
                            if isinstance(raw_lmax, dict):
                                Ltt = int(raw_lmax.get("tt") or raw_lmax.get("TT") or 0)
                                Lee = int(raw_lmax.get("ee") or raw_lmax.get("EE") or 0)
                                Lte = int(raw_lmax.get("te") or raw_lmax.get("TE") or 0)
                            else:
                                L_vals = list(map(int, raw_lmax)) + [0, 0, 0, 0]
                                Ltt, Lee, _, Lte = L_vals[:4]
                            n_points += max(Ltt - 1, 0) + max(Lee - 1, 0) + max(Lte - 1, 0)
                            chi_total += float(-2.0 * SP.cmb_hil_loglike(p_eval, model_name))
                        elif obs == "CMB_hil_TT":
                            like = SP._get_hilTT_like()
                            try:
                                raw_lmax = like.get_lmax()
                            except Exception:
                                raw_lmax = 2508
                            lTT = int(raw_lmax.get("tt") or raw_lmax.get("TT") or next(iter(raw_lmax.values()))) if isinstance(raw_lmax, dict) else int(raw_lmax[0] if isinstance(raw_lmax, (list, tuple, np.ndarray)) else raw_lmax)
                            n_points += max(lTT - 1, 0)
                            chi_total += float(-2.0 * float(SP.cmb_hilTT_loglike(p_eval, model_name)))
                        elif obs == "CMB_lensing":
                            has_primary = any(x in {"CMB_hil", "CMB_hil_TT", "CMB_lowl"} for x in obs_entry)
                            SP.set_lensing_mode("raw" if has_primary else "cmbmarged")
                            like = SP._get_lensing_like()
                            n_bins = 8
                            try:
                                if hasattr(like, "get_lensing_nbins"):
                                    n_bins = int(like.get_lensing_nbins())
                                elif hasattr(like, "get_lensing_bins"):
                                    n_bins = len(like.get_lensing_bins())
                            except Exception:
                                pass
                            logL, _, note = _stats_lensing_logL_safe(p_eval, model_name)
                            if not (isfinite(logL) and abs(logL) < 1e9):
                                continue
                            if note and (note not in notes):
                                notes.append(note)
                            chi_total += float(-2.0 * logL)
                            n_points += n_bins
                        else:
                            raise ValueError(f"Unsupported CMB observation: {obs}")
                    else:
                        raise ValueError(f"Unsupported observation type: {obs_type}")

                return chi_total, n_points
            #primary calculator for AIC,BIC and AICc
            # --- Polish with Nelder-Mead to find the true minimum chi^2 ---
            prior_limits_container = CONFIG.get(model_name, {}).get("prior_limits", [])
            if isinstance(prior_limits_container, list):
                prior_map = prior_limits_container[obs_index] if obs_index < len(prior_limits_container) else None
            elif isinstance(prior_limits_container, dict):
                prior_map = prior_limits_container.get(obs_index, None)
            else:
                prior_map = None

            param_dict, chi_squared_total = find_polished_mle(
                compute_chi2_fn=_compute_chi2_total,
                params_dict_median=param_dict,
                prior_bounds=prior_map,
                max_evals=300,
            )

            # Re-evaluate once at the polished minimum to fetch exact N data points
            _, num_data_points_total = _compute_chi2_total(param_dict)

            if num_data_points_total <= 0:
                logger.error(
                    "No valid data points contributed to stats; "
                    "skipping stats for %s.",
                    obs_name,
                )
                continue

            log_likelihood = -0.5 * chi_squared_total


            # DEBUG STATEMENTS
            print(f"[DEBUG]   chi_squared_total = {chi_squared_total}")
            print(f"[DEBUG]   log_likelihood = {log_likelihood}")

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
            """
            In order to implement AICc, I need the size of the sample space being worked with.
            """
            reduced_chi_squared = chi_squared_total / dof
            aic = 2 * num_params - 2 * log_likelihood
            bic = num_params * np.log(num_data_points_total) - 2 * log_likelihood
            #In the calculation of AICc, the assumption is made that num_data_points_total can be used as sample space value, n
            if (num_data_points_total - num_params - 1) > 0:
                aicc = 2*num_params - 2*log_likelihood + (2*num_params * (num_params + 1))/(num_data_points_total - num_params - 1)
            else:
                aicc = aic
            dic = float("nan")
            p_D = float("nan")

            #DIC calculations: 
            """
            Note: D_hat is typically the deviance over the average of parameter values,
            but now since we are using maximum likelihood values, it can be shown that
            taking the deviance of the MLE returns the chi_squared_total

            """
            if D_bar is not None:
                # Spiegelhalter / DESI MAP formulation:
                # D_hat is the polished minimum chi-squared found by Nelder-Mead
                #If DIC values are not working, try the variance (Gelman) definition: p_D = D_var/2.0 and dic = D_bar + 2.0*p_D
                D_hat = chi_squared_total
                p_D = D_bar - D_hat       #Spiegelhalter: D_bar - D_hat   Gelman: D_var/2.0
                dic = D_hat + 2.0 * p_D  # Spiegelhalter: D_hat + 2.0*p_D

                print(f"[DEBUG DIC] model={model_name} obs={obs_name}")
                print(f"[DEBUG DIC]   D_bar (mean chi2 of chain) = {D_bar:.4f}")
                print(f"[DEBUG DIC]   D_var (var chi2 of chain) = {D_var:.4f}")
                print(f"[DEBUG DIC]   D_hat (min chi2 / MAP)     = {D_hat:.4f}")
                print(f"[DEBUG DIC]   p_D (effective param count)= {p_D:.4f}")
                print(f"[DEBUG DIC]   Calculated DIC             = {dic:.4f}")

            #WAIC Implememtation
            waic = float('nan')
            waic = 0

            
           

            results[model_name][obs_name] = {
                "Log-Likelihood": log_likelihood,
                "Chi_squared": chi_squared_total,
                "Reduced_Chi_squared": reduced_chi_squared,
                "AIC": aic,
                "BIC": bic,
                "AICc" : aicc,
                "DIC": dic,
                "WAIC": waic,
                "p_D": p_D,
            }
            if notes:
                results[model_name][obs_name]["Note"] = " | ".join(notes)

            if model_name == reference_model:
                reference_chi[obs_name] = chi_squared_total
                reference_aic[obs_name] = aic
                reference_bic[obs_name] = bic
                reference_aicc[obs_name] = aicc
                reference_dic[obs_name] = dic
                reference_waic[obs_name] = waic

    # Calculate delta values relative to the reference model.
    for model_name, obs_results in results.items():
        for obs_name, stats in obs_results.items():
            stats["dAIC"] = stats["AIC"] - reference_aic.get(obs_name, stats["AIC"])
            stats["dBIC"] = stats["BIC"] - reference_bic.get(obs_name, stats["BIC"])
            stats["dAICc"] = stats["AICc"] - reference_aicc.get(obs_name, stats["AICc"])
            stats["dDIC"] = stats["DIC"] -reference_dic.get(obs_name, stats["DIC"])
            stats["dWAIC"] = stats["WAIC"] - reference_waic.get(obs_name, stats["WAIC"])
            stats['dChi'] = stats['Chi_squared'] - reference_chi.get(obs_name, stats['Chi_squared'])

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

def interpret_delta_IC(
        delta_aic, 
        delta_bic,
        delta_aicc,
        delta_dic,
        delta_waic,
        ) -> str:
    """
    Turn ΔAIC / ΔBIC into human-readable model-comparison statements.
    """
    # Coerce to plain floats (works for Python floats, NumPy scalars, 0-d arrays)
    delta_aic = float(np.asarray(delta_aic).reshape(()))
    delta_bic = float(np.asarray(delta_bic).reshape(()))
    delta_aicc = float(np.asarray(delta_aicc).reshape(()))
    delta_dic = float(np.asarray(delta_dic).reshape(()))
    delta_waic = float(np.asarray(delta_waic).reshape(()))

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

    #--- AICc ---
    if delta_aicc < 2:
        feedback.append(
            f"Delta AICc: Indistinguishable (ΔAICc = {delta_aicc:.2f})."
        )
    elif delta_aicc < 4:
        feedback.append(
            f"Delta AICc: Slight evidence against the model (ΔAICc = {delta_aicc:.2f})."
        )
    elif delta_aicc < 7:
        feedback.append(
            f"Delta AICc: Positive evidence against the model (ΔAICc = {delta_aicc:.2f})."
        )
    else:
        feedback.append(
            f"Delta AICc: Strong evidence against the model (ΔAICc = {delta_aicc:.2f})."
        )

    #--- DIC ---
    if delta_dic < 2:
        feedback.append(
            f"Delta DIC: Indistinguishable (ΔDIC = {delta_dic:.2f})."
        )
    elif delta_dic < 4:
        feedback.append(
            f"Delta DIC: Slight evidence against the model (ΔDIC = {delta_dic:.2f})."
        )
    elif delta_dic < 7:
        feedback.append(
            f"Delta DIC Positive evidence against the model (ΔDIC = {delta_dic:.2f})."
        )
    else:
        feedback.append(
            f"Delta DIC: Strong evidence against the model (ΔDIC = {delta_dic:.2f})."
        )


    #--- WAIC  ---
    if delta_waic < 2:
        feedback.append(
            f"Delta WAIC: Indistinguishable (ΔWAIC = {delta_waic:.2f})."
        )
    elif delta_waic < 4:
        feedback.append(
            f"Delta WAIC: Slight evidence against the model (ΔWAIC = {delta_waic:.2f})."
        )
    elif delta_waic < 7:
        feedback.append(
            f"Delta WAIC Positive evidence against the model (ΔWAIC = {delta_waic:.2f})."
        )
    else:
        feedback.append(
            f"Delta WAIC: Strong evidence against the model (ΔWAIC = {delta_waic:.2f})."
        )


    return "\n".join(feedback)
