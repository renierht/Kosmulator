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
from scipy.special import logsumexp #Required for WAIC
from scipy.stats import chi2, norm #used by significance


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
    """
    value^{+plus}_{-minus} with the precision set by the uncertainty: the smaller
    error is shown to two significant figures and the value to the same decimal
    place (0.2974^{+0.0087}_{-0.0083}, 68.83^{+0.50}_{-0.50}, 1452^{+21}_{-20}).
    Four decimals when the errors are zero or not finite.
    """
    lo = abs(minus)
    hi = abs(plus)
    small = min(e for e in (lo, hi)) if np.isfinite(lo) and np.isfinite(hi) else float("nan")
    if np.isfinite(small) and small > 0:
        prec = int(min(max(1 - int(np.floor(np.log10(small))), 0), 8))
    else:
        prec = 4

    fmt = f"{{:.{prec}f}}"
    v_str  = fmt.format(value)
    lo_str = fmt.format(lo)
    hi_str = fmt.format(hi)
    return rf"${v_str}^{{+{hi_str}}}_{{-{lo_str}}}$"


# -----------------------------------------------------------------------------
# Significance calculator function for statistical analysis
# -----------------------------------------------------------------------------
def significance(dchi, dk):
    """
    Gaussian-equivalent significance of a chi^2 improvement, as in DESI DR2
    (arXiv:2503.14738, Eq. 22): the chi^2 CDF of -dchi with dk degrees of
    freedom, expressed as a two-sided N-sigma.

    dchi : chi^2_model - chi^2_reference at the best fits (negative = better fit)
    dk   : number of extra free parameters of the model (nested models)

    Returns 0 when the model does not improve the fit, and nan when dk <= 0
    (no extra parameters, e.g. the reference model itself).
    """
    try:
        dchi = float(dchi)
        dk = int(round(float(dk)))
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(dchi) or dk <= 0:
        return float("nan")
    if dchi >= 0:
        return 0.0
    p = chi2.sf(-dchi, df=dk)
    return float(np.round(norm.isf(p / 2), decimals=1))



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

            #Corrections need for better mle results
            top_n = min(5, len(valid_ll))
            top_idx = np.argsort(valid_ll)[-top_n:]
            best_candidate = valid_samples[top_idx]

            #For DIC calculations
            D_bar = float(np.nanmean(dev_samples))
            D_var = float(4.0 * np.nanvar(valid_ll, ddof = 1))
            theta_bar = np.nanmean(valid_samples, axis = 0)

            #WAIC set-up
            #Whats required is a log_like matrix. Once we got that we can use the previous functions

        else:
            mle_vector = None
            theta_bar  = None
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
        if obs_samples.shape[0] == 0:
            print(f"Warning: '{obs}' has no samples — skipping parameter statistics.")
            latex_table.append([])
            continue

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
            structured_values[obs]["__best_candidates__"] = best_candidate
            structured_values[obs]["__param_names__"] = obs_param_names
            structured_values[obs]["__samples__"] = obs_samples

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
    max_evals: int = 2000,
    candidate_starts: list[dict[str, float]] | None = None,
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

    starts = [x0]
    if candidate_starts:
        for cand in candidate_starts:
            starts.append(np.array([cand[p] for p in p_names], dtype = float))

    best_chi2 = baseline_chi2
    best_p = params_dict_median

    for start in starts:
        res = minimize(
            objective, 
            start, 
            method = "Nelder-Mead",
            options = {
                "maxiter": max_evals,
                "maxfev": max_evals,
                "xatol": 1e-3,
                "fatol":1e-3,
                "disp": False,
            },
        )
        # A run stopped by maxiter/maxfev still returns a valid, lower chi^2,
        # so any improvement is kept (res.success is not required).
        if np.isfinite(res.fun) and res.fun < best_chi2:
            best_chi2 = res.fun
            best_p = {p: float(res.x[idx]) for idx, p in enumerate(p_names)}
        

    return best_p, best_chi2

def compute_lppd(log_lik_matrix):
    """ 
    Couple notes on this:
    lppd - log pointwise posterior predictive density. Measure of how well fitted statistical model predicts observed data.
    """
    S = log_lik_matrix.shape[0] #1000
    c = np.max(log_lik_matrix, axis = 0) #finds max loglike for each samples parameter for all samples
    #subtracting c prevents underflow to 0 which could crash the program.
    lppd_i = c + np.log(np.sum(np.exp(log_lik_matrix - c), axis = 0)) - np.log(S)
    return np.sum(lppd_i)

def compute_p_waic(log_lik_matrix):
    """
    Effective parameter count for WAIC
    """ 
    pwi = np.var(log_lik_matrix, ddof = 1, axis = 0)
    return np.sum(pwi)

def compute_waic(log_lik_matrix):
    lppd = compute_lppd(log_lik_matrix)
    pw = compute_p_waic(log_lik_matrix)
    waic = -2.0 * lppd + 2.0 * pw
    return waic


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
    reference_pD : dict[str, float] = {}
    reference_k: dict[str, int] = {}
    

    for model_name, obs_results in best_fit_values.items():
        results[model_name] = {}
        for obs_name, params in obs_results.items():
            # Extract best-fit (median) values into a dictionary.

            D_bar = params.pop("__D_bar__", None) #Popped before the loop iterates through param
            D_var = params.pop("__D_var__", None)
            best_candidates = params.pop("__best_candidates__", None)
            cand_param_names = params.pop("__param_names__", None)
            obs_samples_for_waic = params.pop("__samples__", None)

            candidate_starts_list = None
            if best_candidates is not None and cand_param_names is not None:
                candidate_starts_list = [
                    {p: float(row[i]) for i, p in enumerate(cand_param_names)}
                    for row in best_candidates
                ]


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
                # Values fixed for this group (e.g. H_0 for uncalibrated SNe); the
                # same object when nothing is fixed, so the gamma note below still works
                p_eval = U.with_fixed_params(p_eval, CONFIG[model_name], obs_index)

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
                        # D_L = (1 + zHEL) D_M(zHD) when z_hel is present (Pantheon+, DES-Y5)
                        distance_modulus = 25 + 5 * np.log10(
                            U.sn_luminosity_distance(comoving_distances, zHD, obs_data.get("z_hel"))
                        )
                        chi_total += float(Calc_PantP_chi(m_b_corr, IS_CALIBRATOR, CEPH_DIST, cov, distance_modulus, p_eval,
                                                          marginalise=bool(obs_data.get("marginalise_offset", False))))
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
                        # (1 + z_HEL) prefactor when the dataset provides z_hel (DESY5)
                        model_val = 25 + 5 * np.log10(
                            U.sn_luminosity_distance(comoving_distances, redshift, obs_data.get("z_hel"))
                        )
                        chi_total += float(Calc_Generic_SNe_chi(obs_data=obs_data, model=model_val, param_dict=p_eval))
                        n_points += len(redshift)

                    elif obs_type in ["OHD", "CC"]:
                        redshift = obs_data["redshift"]
                        model_val = p_eval["H_0"] * np.array([MODEL_func(z, p_eval) for z in redshift])
                        chi_total += float(SP.Calc_obs_chi(obs_type, obs_data, model_val))
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

                        chi_total += float(SP.Calc_obs_chi(obs_type, obs_data, model_val))
                        n_points += len(obs_data["type_data"])

                    elif obs_type in ("BBN_DH", "BBN_DH_AlterBBN"):
                        mode = obs_data.get("mode", "mean")
                        n_points += 1 if mode == "mean" else len(obs_data.get("systems", []))
                        chi_total += float(Calc_BBN_DH_chi(obs_data, MODEL_func, p_eval, obs_type))

                    elif obs in ("BBN_PryMordial", "BBN_prior"):
                        # Counted in chi^2 (as in the sampler's log_like, so D_hat and
                        # D_bar measure the same thing, and the MAP matches DESI's
                        # definition), but not as a data point: it is a prior.
                        chi_total += SP.bbn_prior_chi2(obs_data, p_eval)

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

            # Evaluation cap per Nelder-Mead start: 2000 for late-time groups,
            # where a chi^2 call is cheap; 300 when CMB is in the group, where
            # every call runs CLASS and clik.
            has_cmb_group = any(str(t) == "CMB" for t in obs_types)
            param_dict, chi_squared_total = find_polished_mle(
                compute_chi2_fn=_compute_chi2_total,
                params_dict_median=param_dict,
                prior_bounds=prior_map,
                max_evals=300 if has_cmb_group else 2000,
                candidate_starts= candidate_starts_list,

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


            # k: sampled parameters plus the SN magnitude offsets fitted analytically
            # (1 per SN set with a marginalised offset, 2 for the official JLA)
            n_offsets = U.sn_fitted_offsets(obs_entry, data)
            if n_offsets:
                num_params = len(param_dict) + n_offsets
                notes.append(f"k includes {n_offsets} analytically fitted SN offset(s)")
            fixed_here = (CONFIG[model_name].get("fixed_params_by_group", {}) or {}).get(obs_index) or {}
            if "H_0" in fixed_here:
                notes.append("H_0 not sampled (uncalibrated SNe: the offset absorbs it)")
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
            waic = float('nan')

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
                p_D = D_bar - D_hat       #Spiegelhalter: D_bar - D_hat   Gelman: D_var/2.0 (Effective parameters)
                dic = D_hat + 2.0 * p_D  # Spiegelhalter: D_hat + 2.0*p_D

            #WAIC Implememtation
            if obs_samples_for_waic is not None:
                try:
                    all_rows = []
                    N_total = obs_samples_for_waic.shape[0]
                    S = 1000
                    shared_idx = np.random.choice(N_total, size = min(S, N_total), replace = False)
                    for single_obs, single_type in zip(obs_entry, obs_types):
                        if single_obs not in data:
                            continue
                        matrix = SP.build_log_like_matrix(
                            flat_samples=obs_samples_for_waic,
                            obs_data=data[single_obs],
                            obs_type=single_type,
                            obs_name=single_obs,
                            Model_func=MODEL_func,
                            CONFIG=CONFIG[model_name],
                            obs_index=obs_index,
                            S=1000,
                            idx = shared_idx,
                        )
                        if matrix.size > 0:
                            all_rows.append(matrix)

                    if all_rows:
                        full_matrix = np.concatenate(all_rows, axis=1)  # (S, N_total)
                        waic = compute_waic(full_matrix)
                        logger.debug("[WAIC] %s %s: %.4f", model_name, obs_name, waic)
                except Exception as e:
                    print(f"[WAIC WARNING] {model_name} {obs_name}: {e}")
                    waic = float('nan')

            
           

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
                "num_params": num_params,
                "dof": dof,
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
                reference_pD[obs_name] = p_D
                reference_k[obs_name] = num_params

    # Calculate delta values relative to the reference model.
    for model_name, obs_results in results.items():
        for obs_name, stats in obs_results.items():
            stats["dAIC"] = stats["AIC"] - reference_aic.get(obs_name, stats["AIC"])
            stats["dBIC"] = stats["BIC"] - reference_bic.get(obs_name, stats["BIC"])
            stats["dAICc"] = stats["AICc"] - reference_aicc.get(obs_name, stats["AICc"])
            stats["dDIC"] = stats["DIC"] -reference_dic.get(obs_name, stats["DIC"])
            stats["dWAIC"] = stats["WAIC"] - reference_waic.get(obs_name, stats["WAIC"])
            stats['dChi'] = stats['Chi_squared'] - reference_chi.get(obs_name, stats['Chi_squared'])
            # Degrees of freedom = extra free parameters relative to the reference
            # model (DESI DR2 Eq. 22), not the noisy difference in p_D.
            dk = stats["num_params"] - reference_k.get(obs_name, stats["num_params"])
            stats['sigma'] = significance(stats['dChi'], dk)

    return results


def provide_model_diagnostics(
    reduced_chi_squared,
    model_name: str = "",
    reference_chi_squared=None,
    dof=None,
    datasets=None,
    is_reference=None,
    reference_name=None,
) -> str:
    """
    Plain-language reading of the reduced chi-squared, chi^2_nu = chi^2 / dof.

    datasets: dataset tags of the group, used to name the likely cause of a low
    chi^2_nu. is_reference: True for the reference model (no comparison with
    itself); if None, any model whose name starts with "LCDM" counts as reference.
    reference_name: name of the reference model used in the text (default "LCDM").

    For a model that describes the data, chi^2_nu scatters around 1 with a
    standard deviation of sqrt(2/dof), so the same value can be ordinary for a
    small dataset and highly unusual for a large one.  When ``dof`` is given
    the reading uses that scatter and the chi^2 tail probability; otherwise
    it falls back to fixed bands.

    A k-parameter model fitted to N points already has E[chi^2_min] = N - k,
    which is why dof = N - k.  Fitting noise with a few parameters therefore
    cannot push chi^2_nu well below 1: a low value almost always means the
    quoted error bars are larger than the actual scatter of the data.
    """
    reduced_chi_squared = _scalarize(reduced_chi_squared)
    reference_chi_squared = _scalarize(reference_chi_squared)
    feedback = "Statistical Interpretation:\n"

    rcs = reduced_chi_squared
    try:
        nu = float(dof) if dof is not None else float("nan")
    except (TypeError, ValueError):
        nu = float("nan")

    # Name only the causes that apply to the datasets in this group
    tags = [str(t) for t in (datasets or [])]
    sn_tags = [t for t in tags if t in ("JLA", "JLA_legacy", "Pantheon", "PantheonP",
                                         "PantheonPS", "PantheonP_SH0ES", "DESY5", "Union3")]
    diag_tags = [t for t in tags if t in ("CC", "OHD", "f", "f_sigma_8", "BAO")]
    causes = []
    if sn_tags:
        causes.append("supernova covariances that include conservative systematic terms ("
                      + ", ".join(sn_tags) + ")")
    if diag_tags:
        causes.append("compilations with conservative errors, or correlated errors treated as "
                      "independent (" + ", ".join(diag_tags) + ")")
    if not causes:
        causes.append("conservative (overestimated) error bars or correlations treated as independent")
    low_text = (
        "This is too low to be overfitting: a model with a few free parameters "
        "cannot absorb that much chi-squared. The usual cause is "
        + "; or ".join(causes) + ". Compare models with dChi, sigma and the "
        "information criteria rather than with the absolute chi^2_nu."
    )
    if is_reference is None:
        is_reference = str(model_name).lower().startswith("lcdm")
    high_text = (
        "The data scatter more than their error bars allow. Possible causes are "
        "a model that misses structure in the data, underestimated or missing "
        "systematic errors, or tension between combined datasets."
    )

    if np.isfinite(rcs) and np.isfinite(nu) and nu > 0:
        spread = np.sqrt(2.0 / nu)
        n_sig = (rcs - 1.0) / spread
        chi2_val = rcs * nu
        span = f"(expected 1 +/- {spread:.2f} for {nu:.0f} dof"
        if abs(n_sig) <= 2.0:
            feedback += (
                f"  - chi^2_nu = {rcs:.3f} is consistent with 1 {span}). "
                "The residuals match the quoted uncertainties.\n"
            )
        elif n_sig < -2.0:
            p_low = chi2.cdf(chi2_val, nu)
            feedback += (
                f"  - chi^2_nu = {rcs:.3f} lies {abs(n_sig):.1f} sigma below 1 {span}; "
                f"P(chi^2 <= observed) = {p_low:.1e}). {low_text}\n"
            )
        elif rcs <= 3.0:
            p_high = chi2.sf(chi2_val, nu)
            feedback += (
                f"  - chi^2_nu = {rcs:.3f} lies {n_sig:.1f} sigma above 1 {span}; "
                f"P(chi^2 >= observed) = {p_high:.1e}). {high_text}\n"
            )
        else:
            feedback += (
                f"  - chi^2_nu = {rcs:.3f} is far above 1 {span}). The model does "
                "not describe these data; check the model, the data covariance "
                "and any calibration or nuisance parameters.\n"
            )
    else:
        # dof unknown: fixed bands
        if 0.9 <= rcs <= 1.1:
            feedback += (
                "  - The reduced chi-squared is close to 1: the residuals are "
                "consistent with the quoted uncertainties.\n"
            )
        elif rcs < 0.9:
            feedback += (
                "  - The reduced chi-squared is below 1. With only a few free "
                "parameters this is rarely overfitting; it usually means the "
                "quoted error bars are conservative. Compare models with dChi, "
                "sigma and the information criteria.\n"
            )
        elif rcs <= 3.0:
            feedback += f"  - The reduced chi-squared is above 1. {high_text}\n"
        else:
            feedback += (
                "  - The reduced chi-squared is far above 1. The model does not "
                "describe these data; check the model, the data covariance and "
                "any calibration or nuisance parameters.\n"
            )

    ref = str(reference_name) if reference_name else "LCDM"

    # Benchmark comparison: only for non-reference models
    if reference_chi_squared is not None and np.isfinite(reference_chi_squared) \
            and not is_reference:
        feedback += f"\nBenchmark Comparison (Relative to {ref}):\n"
        if rcs < reference_chi_squared:
            feedback += (
                f"  - chi^2_nu ({rcs:.3f}) is lower than for {ref} "
                f"({reference_chi_squared:.3f}): the fit improves by more than "
                "the change in the number of free parameters. Whether "
                "the improvement is significant is given by dChi, sigma and the "
                "information criteria.\n"
            )
        elif rcs > reference_chi_squared:
            feedback += (
                f"  - chi^2_nu ({rcs:.3f}) is higher than for {ref} "
                f"({reference_chi_squared:.3f}): any extra parameters do not "
                "improve the fit enough to offset the degrees of freedom "
                "they use.\n"
            )
        else:
            feedback += (
                f"  - chi^2_nu matches the {ref} value ({reference_chi_squared:.3f}), "
                "a comparable fit.\n"
            )

    if is_reference:
        feedback += (
            f"\n{ref} is the reference model. Because chi^2_nu depends on how the "
            "data uncertainties were estimated, its absolute value says more "
            "about the dataset than about the model; comparisons of models on "
            "the same data (dChi, sigma, dAIC, dBIC, dDIC) remove that "
            "dependence.\n"
        )

    return feedback

def interpret_delta_IC(
        delta_aic, 
        delta_bic,
        delta_aicc,
        delta_dic,
        delta_waic,
        sigma,
        ) -> str:
    """
    Turn ΔAIC / ΔBIC / ΔAICc / ΔDIC / ΔWAIC and the significance into
    human-readable model-comparison statements.

    Sign convention: Δ = value(model) - value(reference). A positive Δ is
    evidence AGAINST the model; a negative Δ is evidence IN FAVOUR of it. The
    strength is set by |Δ|. Always returns six lines in the order
    AIC, BIC, AICc, DIC, WAIC, Significance (Plots.py indexes them by position).
    """
    # Coerce to plain floats (works for Python floats, NumPy scalars, 0-d arrays)
    delta_aic = float(np.asarray(delta_aic).reshape(()))
    delta_bic = float(np.asarray(delta_bic).reshape(()))
    delta_aicc = float(np.asarray(delta_aicc).reshape(()))
    delta_dic = float(np.asarray(delta_dic).reshape(()))
    delta_waic = float(np.asarray(delta_waic).reshape(()))
    sigma = float(np.asarray(sigma).reshape(()))

    feedback = []

    def _ic_line(label, delta, edges, words):
        """One statement for one criterion; strength from |delta|, direction from its sign."""
        sym = f"Δ{label}"
        if np.isnan(delta):
            return f"Delta {label}: Not available ({sym} = nan)."
        size = abs(delta)
        if size < edges[0]:
            return f"Delta {label}: Indistinguishable ({sym} = {delta:.2f})."
        if size < edges[1]:
            word = words[0]
        elif size < edges[2]:
            word = words[1]
        else:
            word = words[2]
        direction = "against" if delta > 0 else "in favour of"
        return f"Delta {label}: {word} evidence {direction} the model ({sym} = {delta:.2f})."

    # Bin edges and labels are unchanged from the original thresholds.
    feedback.append(_ic_line("AIC",  delta_aic,  (2, 4, 7),  ("Slight", "Positive", "Strong")))
    feedback.append(_ic_line("BIC",  delta_bic,  (2, 6, 10), ("Weak", "Moderate", "Strong")))
    feedback.append(_ic_line("AICc", delta_aicc, (2, 4, 7),  ("Slight", "Positive", "Strong")))
    feedback.append(_ic_line("DIC",  delta_dic,  (2, 4, 7),  ("Slight", "Positive", "Strong")))
    feedback.append(_ic_line("WAIC", delta_waic, (2, 4, 7),  ("Slight", "Positive", "Strong")))

    #--- Significance ---
    if sigma <= 1:
        feedback.append(
            f"Significance: Inconclusive (Sigma = {sigma:.2f})."
        )
    elif sigma > 1 and sigma <= 2:
        feedback.append(
            f"Significance: Slight consideration for the model (Sigma = {sigma:.2f})."
        )
    elif sigma > 2 and sigma <= 3:
        feedback.append(
            f"Significance: Considerable consideration for the model (Sigma = {sigma:.2f})."
        )
    elif sigma > 3 and sigma <= 4:
        feedback.append(
            f"Significance: Strong consideration for the model (Sigma = {sigma:.2f})."
        )
    elif sigma > 4 and sigma <= 5:
        feedback.append(
            f"Significance: Very strong consideration for the model (Sigma = {sigma:.2f})."
        )
    elif sigma > 5:
        feedback.append(
            f"Significance: Overwhelming consideration for the model (Sigma = {sigma:.2f})."
        )
    else:
        feedback.append(f"Significance: Not available (Sigma = {sigma}).")



    return "\n".join(feedback)
