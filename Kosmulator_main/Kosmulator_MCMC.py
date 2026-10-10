#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main MCMC driver (Zeus / emcee) for a single model + observation set.

Public entry point:
    run_mcmc(...)

Design notes:
- Likelihood/prior glue lives in top-level helpers (no nested defs).
- Initial position generation, re-seeding of invalid walkers, and
  resume logic are factored out for clarity.
"""
from __future__ import annotations

import os
import time
import logging
import h5py
from typing import Callable, Dict, Any, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import emcee
from scipy import optimize

from Kosmulator_main import utils
from Kosmulator_main import Statistical_packages as SP
import Kosmulator_main.constants as K
from Plots.Plot_functions import compute_rd as _compute_rd
from Kosmulator_main import rd_helpers as RD

logger = logging.getLogger(__name__)

# Optional Zeus
try:
    import zeus
except Exception:
    zeus = None

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ───────────────────────────────────────────────────────────────────────────────
# Prior / Likelihood glue (vectorised where possible)
# ───────────────────────────────────────────────────────────────────────────────

def _theta_params(theta: np.ndarray, CONFIG: dict, obs_index: int) -> dict:
    """Parameter dict for one walker: sampled values, derived background values and,
    for BBN groups without a sampled r_d, the calibrated r_d."""
    params = CONFIG["parameters"][obs_index]
    param_dict = {p: theta[i] for i, p in enumerate(params)}
    # Derive background vars if using CMB params / 100theta_s
    param_dict = utils.ensure_background_params(param_dict)
    # Values the configuration fixes for this group (e.g. H_0 for uncalibrated SNe)
    param_dict = utils.with_fixed_params(param_dict, CONFIG, obs_index)
    # If BBN is present in this observation set and r_d wasn't sampled,
    # compute r_d from (Omega_bh^2, Omega_m, H_0[, N_eff]).
    RD._maybe_calibrate_rd(param_dict, CONFIG, obs_index)
    return param_dict


def _sn_batched(obs_data: dict, obs_type: str) -> bool:
    """SN sets whose chi^2 is evaluated for all walkers at once (all but the official JLA,
    whose covariance depends on alpha and beta)."""
    t = obs_type[0] if isinstance(obs_type, list) else obs_type
    return t == "SNe" and not obs_data.get("jla_full")


def _sn_residual(param_dict: dict, obs_data: dict, MODEL_func: Callable, obs: str):
    """SN residual vector for one walker (None if the distances are unphysical);
    its chi^2 is SP.sn_chi2_rows(obs_data, obs, residuals)."""
    z = obs_data["zHD"] if obs in ("PantheonP", "PantheonPS") else obs_data["redshift"]
    d_c = utils.Comoving_distance_vectorized(MODEL_func, z, param_dict)
    # (1 + z_HEL) prefactor when the dataset provides z_hel (DESY5); otherwise
    # identical to the original d_c * (1 + z).
    y_dl = utils.sn_luminosity_distance(d_c, z, obs_data.get("z_hel"))
    if (not np.isfinite(y_dl).all()) or (np.min(y_dl) <= 0):
        return None
    model = 25.0 + 5.0 * np.log10(y_dl)
    if obs in ("PantheonP", "PantheonPS"):
        return SP.pantheon_residual(obs_data["m_b_corr"], obs_data["IS_CALIBRATOR"], obs_data["CEPH_DIST"],
                                    model, param_dict, bool(obs_data.get("marginalise_offset", False)))
    return SP.generic_sn_residual(obs_data, model, param_dict)


def _sn_loglike_batch(theta_batch: np.ndarray, obs_data: dict, CONFIG: dict, MODEL_func: Callable,
                      obs: str, obs_index: int) -> np.ndarray:
    """log-likelihood of one SN set for every walker: residuals walker by walker, then
    the chi^2 of all of them in one matrix product (SP.sn_chi2_rows)."""
    n = theta_batch.shape[0]
    out = np.full(n, -np.inf)
    rows, idx = [], []
    for j in range(n):
        r = _sn_residual(_theta_params(theta_batch[j], CONFIG, obs_index), obs_data, MODEL_func, obs)
        if r is not None:
            rows.append(r)
            idx.append(j)
    if rows:
        out[np.asarray(idx)] = -0.5 * SP.sn_chi2_rows(obs_data, obs, np.vstack(rows))
    return out


def model_likelihood(
    theta: np.ndarray,
    obs_data: dict,
    obs_type: str,
    CONFIG: dict,
    MODEL_func: Callable,
    model_name,
    obs: str,
    obs_index: int,
) -> float:
    """
    Return log-likelihood (-0.5*chi2) for one theta row and one dataset.
    """
    if isinstance(obs_type, list):
        obs_type = obs_type[0]

    param_dict = _theta_params(theta, CONFIG, obs_index)

    # Dedicated CMB branches
    if obs == "CMB_hil":
        return SP.cmb_hil_loglike(param_dict, model_name)
    if obs == "CMB_hil_TT":
        return SP.cmb_hilTT_loglike(param_dict, model_name)
    if obs == "CMB_lowl":
        return SP.cmb_lowl_loglike(param_dict, model_name)
    if obs == "CMB_lowl_TT":
        return SP.cmb_lowlTT_loglike(param_dict, model_name)
    if obs == "CMB_lensing":
        # Determine if we are using the RAW or CMBMARGED lensing likelihood.
        groups = CONFIG.get("observations", [])
        group = groups[obs_index] if obs_index < len(groups) else obs
        if not isinstance(group, (list, tuple)):
            group = [group]

        # Check if any primary Planck CMB likelihood is present in this specific group
        primary_cmb_tags = set(K.CMB_PRIMARY_TAGS)
        has_primary_cmb = any(str(g).strip() in primary_cmb_tags for g in group)

        # Rule: If primary CMB is present, use raw. If primary CMB is absent, use marged.
        mode = "raw" if has_primary_cmb else "cmbmarged"
        SP.set_lensing_mode(mode)

        grp_str = "+".join(map(str, group))
        _last = getattr(SP, "_last_lensing_mode_logged", None)
        if mode != _last:
            import multiprocessing as mp
            import sys
            
            # 1. Bulletproof check to ensure only the true master core prints
            is_mpi_master = True
            try:
                from mpi4py import MPI
                if MPI.COMM_WORLD.Get_size() > 1:
                    is_mpi_master = (MPI.COMM_WORLD.Get_rank() == 0)
            except Exception:
                pass
                
            is_local_master = (mp.current_process().name == "MainProcess")

            # 2. Only log and flush if we are on the main coordinating core
            if is_mpi_master and is_local_master:
                log.info("[CMB-lensing switch] group=%s \u2192 mode=%s", grp_str, mode)
                
                # 3. Safely flush the standard output and log handlers
                sys.stdout.flush()
                for handler in log.handlers:
                    handler.flush()
            
            # 4. Update the local variable on EVERY core so they don't try to print again
            SP._last_lensing_mode_logged = mode

        # This returns a scalar log-likelihood
        return SP.cmb_lensing_loglike(param_dict, model_name)


    # BAO / DESI (DESI VI files; Calc_DESI_chi applies the r_d policy)
    if obs in ("BAO", "DESI_DR1", "DESI_DR2"):
        return -0.5 * SP.Calc_DESI_chi(obs_data, MODEL_func, param_dict, obs_type)

    # BBN: either full DH dataset or prior
    if obs in ("BBN_DH", "BBN_DH_AlterBBN") or obs_type == "BBN_DH":
        return -0.5 * SP.Calc_BBN_DH_chi(obs_data, MODEL_func, param_dict, "BBN_DH")

    if obs in ("BBN_PryMordial", "BBN_prior"):
        # Gaussian prior on Omega_b h^2 (or 2D with N_eff); shared with the statistics step
        return -0.5 * SP.bbn_prior_chi2(obs_data, param_dict)

    # SN sets except the official JLA: the same code as the batched path (one row)
    if _sn_batched(obs_data, obs_type):
        r = _sn_residual(param_dict, obs_data, MODEL_func, obs)
        if r is None:
            return -np.inf
        return -0.5 * float(SP.sn_chi2_rows(obs_data, obs, r)[0])

    # Non-CMB standard data containers (Pantheon+ is always handled above)
    z         = obs_data["redshift"]
    type_data = obs_data["type_data"]
    type_err  = obs_data["type_data_error"]

    # Predictions per type
    if obs_type == "SNe":
        d_c = utils.Comoving_distance_vectorized(MODEL_func, z, param_dict)
        # (1 + z_HEL) prefactor when the dataset provides z_hel (DESY5); otherwise
        # identical to the original d_c * (1 + z).
        y_dl = utils.sn_luminosity_distance(d_c, z, obs_data.get("z_hel"))
        if (not np.isfinite(y_dl).all()) or (np.min(y_dl) <= 0):
            return -np.inf
        model = 25.0 + 5.0 * np.log10(y_dl)

    elif obs_type in ["OHD", "CC"]:
        E_z = utils.E_of_z(z, MODEL_func, param_dict)
        if (not np.isfinite(E_z).all()) or np.any(E_z <= 0):
            return -np.inf
        model = param_dict["H_0"] * E_z

    elif obs_type in ["f_sigma_8", "f"]:
        gamma = None
        if obs_type == "f_sigma_8":
            # read fixed gamma for this obs_index if it was removed from params
            gamma = param_dict.get("gamma", None)
            if gamma is None:
                gamma = CONFIG.get("fs8_gamma_fixed_by_group", {}).get(obs_index, K.GAMMA_FS8_SINGLETON)
        # f-only still samples gamma (no fix) unless you choose otherwise
        model = utils.growth_prediction(obs_type, obs_data, param_dict, MODEL_func, gamma)
        if not np.isfinite(model).all():
            return -np.inf

    else:
        return -np.inf

    # --- Calculate Chi^2 ---
    if obs_type == "SNe":
        # Only the official JLA reaches this point (alpha/beta-dependent covariance)
        chi2 = SP.Calc_Generic_SNe_chi(
            obs_data=obs_data,      # <--- CHANGED from current_data to obs_data
            model=model,            # <--- Ensure this matches your local var (usually 'model' or 'model_val')
            param_dict=param_dict   # <--- Ensure this matches your local var (usually 'param_dict')
        )

    else:
        # Fallback for CC, f_sigma_8, OHD, etc.
        chi2 = SP.Calc_obs_chi(obs_type, obs_data, model)

    return -0.5 * chi2


def log_prior_all(theta_batch: np.ndarray, CONFIG: Dict[str, Any], obs_index: int) -> np.ndarray:
    """Vectorised top-hat priors with optional restrictions + coupled background checks."""
    nwalkers, ndim = theta_batch.shape
    lp = np.zeros(nwalkers)

    params = CONFIG["parameters"][obs_index]

    # 1) top-hat bounds
    for i, p in enumerate(params):
        low, high = CONFIG["prior_limits"][obs_index][p]
        mask = (theta_batch[:, i] < low) | (theta_batch[:, i] > high)
        lp[mask] = -np.inf

    # 1b) Gaussian priors of the Planck nuisance parameters (calibrations, dust
    #     amplitudes) and the SZ prior on ksz_norm + 1.6 A_sz, as in Planck's baseline
    #     and Cobaya (constants.PLANCK_GAUSSIAN_PRIORS, PLANCK_SZ_PRIOR)
    for i, p in enumerate(params):
        g = K.PLANCK_GAUSSIAN_PRIORS.get(p)
        if g is not None:
            lp += -0.5 * ((theta_batch[:, i] - g[0]) / g[1]) ** 2
    if ("ksz_norm" in params) and ("A_sz" in params):
        c_sz, mu_sz, sig_sz = K.PLANCK_SZ_PRIOR
        sz = theta_batch[:, params.index("ksz_norm")] + c_sz * theta_batch[:, params.index("A_sz")]
        lp += -0.5 * ((sz - mu_sz) / sig_sz) ** 2

    # 2) per-parameter restrictions (1D predicates)
    restr = CONFIG.get("restrictions", {})
    for i, p in enumerate(params):
        if p in restr:
            valid = np.array([restr[p](v) for v in theta_batch[:, i]])
            lp[~valid] = -np.inf

    # 3) restrictions involving multiple sampled parameters
    coupled_restr = CONFIG.get("coupled_restrictions", [])
    for restriction in coupled_restr:
        valid = np.array([
            restriction({name: theta_batch[walker, i] for i, name in enumerate(params)})
            for walker in range(nwalkers)
        ])
        lp[~valid] = -np.inf

    # --- Coupled background consistency (always true physically) ---
    params = CONFIG["parameters"][obs_index]

    if ("Omega_m" in params) and ("Omega_b" in params):
        i_m = params.index("Omega_m")
        i_b = params.index("Omega_b")
        bad = theta_batch[:, i_b] > theta_batch[:, i_m]
        lp[bad] = -np.inf

    # --- Derived ωb, ωcdm bounds (use prior_limits even if not sampled) ---
    prior_global = CONFIG.get("prior_limits_global", {})

    if ("Omega_b" in params) and ("H_0" in params):
        i_b = params.index("Omega_b")
        i_h0 = params.index("H_0")
        h = theta_batch[:, i_h0] / 100.0
        omega_b = theta_batch[:, i_b] * (h * h)

        if "Omega_bh^2" in prior_global:
            ob_lo, ob_hi = prior_global["Omega_bh^2"]
            lp[(omega_b < ob_lo) | (omega_b > ob_hi)] = -np.inf

    if ("Omega_m" in params) and ("Omega_b" in params) and ("H_0" in params):
        i_m = params.index("Omega_m")
        i_b = params.index("Omega_b")
        i_h0 = params.index("H_0")
        h = theta_batch[:, i_h0] / 100.0
        omega_cdm = (theta_batch[:, i_m] - theta_batch[:, i_b]) * (h * h)

        if "Omega_dh^2" in prior_global:
            od_lo, od_hi = prior_global["Omega_dh^2"]
            lp[(omega_cdm < od_lo) | (omega_cdm > od_hi)] = -np.inf

    return lp


def log_likelihood_all(
    theta_batch: np.ndarray,
    data: Dict[str, Any],
    CONFIG: Dict[str, Any],
    MODEL_func: Callable,
    model_name,
    obs: List[str],
    Type: List[str],
    obs_index: int,
) -> np.ndarray:
    """Vector of total log-likelihood across all requested datasets."""
    nwalkers, ndim = theta_batch.shape
    ll = np.zeros(nwalkers, dtype=float)

    # A group with Planck data: the model's CLASS build is loaded before any term, so a term
    # that uses CLASS without loading it (BAO's CMB-calibrated r_d, rd_helpers) gets that
    # build from a worker's first point on (no-op once loaded)
    if any(str(t) == "CMB" for t in Type):
        from Kosmulator_main import Class_run as _CR
        _CR.ensure_class_ready(model_name)

    for obs_name, obs_type in zip(obs, Type):
        if _sn_batched(data[obs_name], obs_type):
            # SN chi^2 of all walkers in one matrix product (the large covariances)
            ll += _sn_loglike_batch(theta_batch, data[obs_name], CONFIG, MODEL_func, obs_name, obs_index)
            continue
        for j in range(nwalkers):
            val = model_likelihood(
                theta_batch[j],
                data[obs_name],
                obs_type,
                CONFIG,
                MODEL_func,
                model_name,
                obs_name,
                obs_index,
            )
            ll[j] += float(val)

    return ll

def emcee_prob(theta, data, Type, CONFIG, MODEL_func, model_name, obs, obs_index):
    """Scalar log-posterior for emcee. Returns (log_post, log_like_blob)."""
    arr = np.atleast_2d(theta)
    lp = log_prior_all(arr, CONFIG, obs_index)

    if not np.all(np.isfinite(lp)):
        return -np.inf, np.nan

    ll = log_likelihood_all(arr, data, CONFIG, MODEL_func, model_name, obs, Type, obs_index)

    # Force scalar for single-walker call
    lp0 = float(np.asarray(lp, dtype=float).ravel()[0])
    ll0 = float(np.asarray(ll, dtype=float).ravel()[0])

    return lp0 + ll0, ll0


def emcee_prob_vectorized(theta, data, Type, CONFIG, MODEL_func, model_name, obs, obs_index):
    """
    emcee_prob for a whole set of walkers at once (emcee's vectorize=True): theta is
    (n, ndim); returns one (log_post, log_like) pair per walker, the same values as
    emcee_prob, so the stored log_prob and log_like blobs keep their layout.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    lp = np.atleast_1d(np.asarray(log_prior_all(theta, CONFIG, obs_index), dtype=float))
    ll = np.full(theta.shape[0], np.nan)
    ok = np.isfinite(lp)
    if ok.any():
        try:
            v = np.asarray(log_likelihood_all(theta[ok], data, CONFIG, MODEL_func, model_name,
                                              obs, Type, obs_index), dtype=float).ravel()
            if v.shape[0] != int(ok.sum()):
                raise ValueError("vectorised likelihood returned the wrong shape")
            ll[ok] = v
        except Exception:
            for k in np.where(ok)[0]:
                ll[k] = emcee_prob(theta[k], data, Type, CONFIG, MODEL_func, model_name, obs, obs_index)[1]
    post = np.where(ok, lp + ll, -np.inf)
    post[~np.isfinite(post)] = -np.inf
    return [(float(a), float(b)) for a, b in zip(post, ll)]


def batch_post(theta, data, CONFIG, MODEL_func, model_name, obs, Type, obs_index):
    """Vectorised log-posterior used by Zeus (and emcee diagnostics)."""
    theta = np.atleast_2d(theta)
    nwalkers = theta.shape[0]
    out = np.full(nwalkers, -np.inf, dtype=float)

    lp = np.atleast_1d(log_prior_all(theta, CONFIG, obs_index))
    valid = np.isfinite(lp)
    if not np.any(valid):
        return out[0] if nwalkers == 1 else out

    try:
        ll = log_likelihood_all(
            theta[valid], data, CONFIG, MODEL_func, model_name, obs, Type, obs_index
        )
        ll = np.asarray(ll, dtype=float).ravel()
        if ll.shape[0] != valid.sum():
            raise ValueError(
                f"Vectorized likelihood returned wrong shape: {ll.shape}, "
                f"expected {valid.sum()}"
            )
        out[valid] = lp[valid] + ll
    except Exception:
        # Fallback: scalar loop over valid walkers
        for k, th in zip(np.where(valid)[0], theta[valid]):
            try:
                ll1 = log_likelihood_all(
                    th[None, :],
                    data,
                    CONFIG,
                    MODEL_func,
                    model_name,
                    obs,
                    Type,
                    obs_index,
                )
                ll1 = float(np.asarray(ll1, dtype=float).ravel()[0])
                out[k] = lp[k] + ll1
            except Exception:
                out[k] = -np.inf

    return out[0] if nwalkers == 1 else out

def _zeus_logpost_vectorized(theta, data, CONFIG, MODEL_func, model_name, obs, Type, obs_index):
    """Zeus log-posterior for vectorised models."""
    val = batch_post(theta, data, CONFIG, MODEL_func, model_name, obs, Type, obs_index)
    # CRITICAL FIX: .ravel() guarantees a 1D array, preventing 0-d iteration crashes in Zeus
    return np.asarray(val, dtype=float).ravel()


def _zeus_logpost_scalar(theta, data, CONFIG, MODEL_func, model_name, obs, Type, obs_index):
    """Zeus log-posterior for non-vectorised models (scalar)."""
    val = batch_post(theta, data, CONFIG, MODEL_func, model_name, obs, Type, obs_index)
    return float(np.asarray(val, dtype=float).ravel()[0])


# ───────────────────────────────────────────────────────────────────────────────
# Initial positions / optimisation helpers
# ───────────────────────────────────────────────────────────────────────────────

def neg_log_prob(theta, data, CONFIG, MODEL_func, model_name, obs, Type, obs_index) -> float:
    """Negative log-posterior for SciPy optimisation."""
    arr = np.atleast_2d(theta)
    lp = log_prior_all(arr, CONFIG, obs_index)
    if not np.all(np.isfinite(lp)):
        return np.inf
    ll = log_likelihood_all(
        arr, data, CONFIG, MODEL_func, model_name, obs, Type, obs_index
    )
    tot = np.asarray(lp) + np.asarray(ll)

    # minimize() requires a scalar objective; if vectorised returns arrays, reduce them.
    if tot.ndim > 0:
        tot = np.sum(tot)

    return -float(tot)


def optimise_initial_guess(
    reference_vals: np.ndarray,
    bounds: List[Tuple[float, float]],
    nlp_fn: Callable[[np.ndarray], float],
    maxiter: int,
    maxfun: int,
    disp: bool,
    quiet_cap: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Nelder-Mead simplex optimization to find a robust MAP center across non-smooth boundaries.
    """
    
    NO_OPT = os.environ.get("KOSM_NO_OPT", "0") == "1"
    if NO_OPT:
        return np.asarray(reference_vals, float), None

    # Nelder-Mead natively supports parameter box bounds in modern SciPy
    sol = optimize.minimize(
        nlp_fn,
        reference_vals,
        bounds=bounds,
        method="Nelder-Mead",
        options={
            "maxiter": maxiter,
            "maxfev": maxfun,
            "disp": disp,
            "adaptive": True,  # Scales simplex geometry for higher dimensions
        },
    )

    ic = np.asarray(sol.x, float)

    # If the simplex failed to move away from the starting guess, log or inspect
    if not sol.success:
        if quiet_cap and int(getattr(sol, "nfev", 0)) >= int(maxfun):
            # CMB groups: the cap is meant to stop it (item 34)
            logger.info("Nelder-Mead pre-fit stopped at its cap of %d calls "
                        "(it only centres the initial walker ball).", int(maxfun))
        else:
            logger.warning("Nelder-Mead pre-fit did not achieve full convergence: %s", sol.message)

    # Nelder-Mead gives no Hessian, so estimate the diagonal curvature by finite
    # differences (2*ndim + 1 calls). It sets the size of the initial walker ball.
    if os.environ.get("KOSM_NO_CURVATURE", "0") == "1":
        return ic, None
    try:
        return ic, _diag_conditional_variance(nlp_fn, ic, bounds)
    except Exception as e:  # never block a run on this
        logger.warning("Curvature estimate for the initial ball failed (%s); using prior widths.", e)
        return ic, None


def _diag_conditional_variance(
    nlp_fn: Callable[[np.ndarray], float],
    x: np.ndarray,
    bounds: List[Tuple[float, float]],
    rel_step: float = 1e-3,
) -> np.ndarray:
    """
    Conditional variances 1/H_ii of the posterior at x, from central second
    differences of -log posterior with step rel_step * (prior width). The
    stencil is moved inside the prior when x sits on a bound. Returns inf
    where the curvature is not positive (flat or unconstrained directions).
    """
    x = np.asarray(x, float)
    lows = np.array([b[0] for b in bounds], float)
    highs = np.array([b[1] for b in bounds], float)
    spans = np.maximum(highs - lows, 1e-300)
    var = np.full(x.size, np.inf)
    f_x = float(nlp_fn(x))
    for i in range(x.size):
        h = rel_step * spans[i]
        if not (h > 0) or spans[i] <= 2.0 * h:
            continue
        c = min(max(x[i], lows[i] + h), highs[i] - h)
        def f_at(v):
            y = x.copy(); y[i] = v
            return float(nlp_fn(y))
        f_c = f_x if c == x[i] else f_at(c)
        H = (f_at(c + h) - 2.0 * f_c + f_at(c - h)) / (h * h)
        if np.isfinite(H) and H > 0:
            var[i] = 1.0 / H
    return var


def _zeus_warmup(
    pos0: np.ndarray,
    logprob_fn: Callable,
    args: tuple,
    nwalker: int,
    ndim: int,
    vectorize: bool,
    pool,
    n_steps: int,
    segment: int,
    lows: np.ndarray,
    highs: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    """
    Throw-away zeus warm-up before the recorded chain. After every `segment`
    steps, walkers with 2 (max log-post - log-post) > chi2.isf(1e-6, ndim) are
    moved onto randomly chosen good walkers plus a small jitter (1% of the good
    walkers' spread). Only the starting ensemble of the real run changes, so the
    recorded chain is untouched. Returns (positions, number of resets).
    """
    from scipy.stats import chi2 as _chi2
    thr = 0.5 * float(_chi2.isf(1e-6, ndim))
    X = np.array(pos0, float)
    n_reset, done_w = 0, 0
    segment = max(1, int(segment))
    while done_w < n_steps:
        k = min(segment, n_steps - done_w)
        s = _zeus_ensemble(nwalker, ndim, logprob_fn, args=args, pool=pool,
                           vectorize=vectorize, verbose=False)
        s.run_mcmc(X, k, progress=False)
        X = np.array(s.get_chain()[-1], float)
        lp = np.asarray(s.get_log_prob()[-1], float)
        done_w += k
        finite = np.isfinite(lp)
        if not finite.any():
            continue
        good = finite & (lp >= np.nanmax(lp[finite]) - thr)
        bad = ~good
        if bad.any() and good.sum() >= 2:
            src = rng.choice(np.where(good)[0], size=int(bad.sum()))
            spread = np.std(X[good], axis=0)
            X[bad] = _reflect_into(
                X[src] + 0.01 * spread * rng.standard_normal((int(bad.sum()), ndim)), lows, highs
            )
            n_reset += int(bad.sum())
    return X, n_reset


def _reflect_into(x: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    """Fold points back into [low, high] by reflection at the bounds (keeps the spread)."""
    span = np.maximum(highs - lows, 1e-300)
    y = np.mod(np.asarray(x, float) - lows, 2.0 * span)
    y = np.where(y > span, 2.0 * span - y, y)
    return lows + y


def make_initial_positions(
    ic: np.ndarray,
    prior_map: Dict[str, Tuple[float, float]],
    param_names: List[str],
    nwalker: int,
    rng: np.random.Generator,
    base_frac: float,
    hessian_diag: Optional[np.ndarray] = None,
    sigma_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Propose the initial walker ball around 'ic', respecting priors and avoiding duplicates.
    Returns (pos0, spans, lows, highs, std).

    Per parameter the ball has width sigma_frac * sqrt(conditional variance) from
    `hessian_diag` (variances 1/H_ii), so every walker starts in the bulk of the
    posterior. Where the curvature is unknown or flat, the width is base_frac of
    the prior range. Points outside the prior are reflected back in, so a starting
    point on a bound (e.g. n = 0, or H_0 at its lower edge) still gets a spread;
    clipping put all walkers on the bound, and an ensemble with no spread in a
    coordinate can never move it.
    """
    lows  = np.array([prior_map[p][0] for p in param_names], float)
    highs = np.array([prior_map[p][1] for p in param_names], float)
    spans = np.maximum(highs - lows, 1e-12)
    ndim = len(param_names)

    # Ensure IC inside priors
    ic = np.clip(np.asarray(ic, float), lows, highs)

    max_scale = base_frac * spans
    std = max_scale.copy()
    if hessian_diag is not None and np.size(hessian_diag) == ndim:
        var = np.asarray(hessian_diag, float)
        ok = np.isfinite(var) & (var > 0)
        std[ok] = sigma_frac * np.sqrt(var[ok])
    std = np.clip(std, 1e-9 * spans, max_scale)

    pos0 = _reflect_into(ic + rng.normal(size=(nwalker, ndim)) * std, lows, highs)

    # De-duplicate (up to a few retries)
    for _ in range(5):
        uniq, idx = np.unique(np.round(pos0, decimals=12), axis=0, return_index=True)
        if len(idx) == nwalker:
            break
        dup_mask = np.ones(nwalker, dtype=bool)
        dup_mask[idx] = False
        pos0[dup_mask] = _reflect_into(
            ic + std * rng.normal(size=(dup_mask.sum(), ndim)), lows, highs
        )

    return pos0, spans, lows, highs, std


def regenerate_invalid_walkers(
    pos0: np.ndarray,
    post0: np.ndarray,
    ic: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    spans: np.ndarray,
    rng: np.random.Generator,
    PLOT_SETTINGS: Dict[str, Any],
    obs: List[str],
    Type: List[str],
    data: Dict[str, Any],
    CONFIG: Dict[str, Any],
    MODEL_func: Callable,
    model_name: str,
    obs_index: int,
    scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Re-generate any invalid (non-finite posterior) walkers with shrinking radius,
    starting from the initial-ball width `scale` (0.1 of the prior range if None).
    """
    bad = np.where(~np.isfinite(post0))[0]
    if not bad.size:
        return pos0

    regen_max_tries = PLOT_SETTINGS.get("init_regen_tries", 20)
    regen_shrink    = PLOT_SETTINGS.get("init_regen_shrink", 0.7)
    jitter = (np.asarray(scale, float).copy() if scale is not None else 0.1 * spans)

    tries = 0
    while bad.size and tries < regen_max_tries:
        nbad = bad.size
        cand = ic + rng.normal(size=(nbad, ic.size)) * jitter
        cand = _reflect_into(cand, lows, highs)

        pos0[bad] = cand
        post0 = batch_post(
            pos0, data, CONFIG, MODEL_func, model_name, obs, Type, obs_index
        )
        bad = np.where(~np.isfinite(post0))[0]

        jitter *= regen_shrink
        tries += 1

    if bad.size:
        eps = 1e-4 * spans
        pos0[bad] = np.clip(
            ic + rng.normal(size=(bad.size, ic.size)) * eps, lows, highs
        )
    return pos0


# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────

_BLAS_LIMITER = None

# ── Points the likelihood could not compute (item 47) ──────────────────────────
# The CMB likelihoods return a sentinel (-1e10) when CLASS or clik fails at a point
# (e.g. tau_reio where reionisation cannot be computed); the sampler then rejects the
# point. The sampler's map is wrapped so that the main process counts these points
# among the values the pool, the MPI ranks or this process return (serial). Not
# counted: samplers that run vectorised (CMB groups only with --force_zeus or
# --force_vectorisation) and the zeus warm-up.
_FAILED_POINTS = {"failed": 0, "evaluated": 0}


class _CountingMap:
    """map for emcee/zeus: the pool's map (or the built-in one) that counts the returned
    log-posteriors at or below constants.LOGLIKE_FAILED_BELOW (finite)."""

    def __init__(self, pool):
        self.pool = pool

    def map(self, fn, iterable):
        res = list(map(fn, iterable) if self.pool is None else self.pool.map(fn, iterable))
        bad = 0
        for r in res:
            v = r[0] if isinstance(r, (tuple, list)) else r
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and v <= K.LOGLIKE_FAILED_BELOW:
                bad += 1
        _FAILED_POINTS["evaluated"] += len(res)
        _FAILED_POINTS["failed"] += bad
        return res


def _counting_pool(pool, vectorize: bool):
    """The pool to give the sampler: a counting map unless the sampler is vectorised
    (then it calls the log-probability itself and ignores the pool)."""
    return pool if vectorize else _CountingMap(pool)


# ── Pool workers: data sent once per group, not with every walker evaluation ────
# The likelihood arguments (data, CONFIG, ...) are written to a file once per
# observation group; each pool task carries only theta and the file's path, and a
# worker loads the file the first time it sees that path. Before, every task
# pickled the whole data dictionary (26 MB with JLA, 74 MB with DES-Y5 and
# Pantheon+). The file sits next to the chain, so MPI ranks on other nodes can
# read it from the shared file system.
_SHARED_ARGS: Dict[str, tuple] = {}
_SHARED_FILES: List[str] = []


class _PoolLogProb:
    """Picklable log-probability for pool workers (see the note above)."""

    def __init__(self, fn: Callable, path: str):
        self.fn, self.path = fn, path

    def __call__(self, theta):
        args = _SHARED_ARGS.get(self.path)
        if args is None:
            import pickle
            with open(self.path, "rb") as fh:
                args = pickle.load(fh)
            _SHARED_ARGS.clear()            # keep only the current group's data
            _SHARED_ARGS[self.path] = args
        return self.fn(theta, *args)


def _pool_logprob(fn: Callable, args: tuple, near: str) -> _PoolLogProb:
    """Write `args` once (next to the chain file `near`) and return the pool callable."""
    import pickle
    import tempfile
    folder = os.path.dirname(os.path.abspath(near)) if near else tempfile.gettempdir()
    os.makedirs(folder, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=".kosm_pool_args_", suffix=".pkl", dir=folder)
    with os.fdopen(fd, "wb") as fh:
        pickle.dump(args, fh, protocol=pickle.HIGHEST_PROTOCOL)
    _SHARED_FILES.append(path)
    _SHARED_ARGS.clear()
    _SHARED_ARGS[path] = args               # the main process needs no reload
    return _PoolLogProb(fn, path)


def _remove_shared_files() -> None:
    while _SHARED_FILES:
        path = _SHARED_FILES.pop()
        _SHARED_ARGS.pop(path, None)
        try:
            os.remove(path)
        except OSError:
            pass


def _zeus_ensemble(*args, **kwargs):
    """
    zeus.EnsembleSampler without its side effect on logging: zeus replaces every
    handler of the root logger with a plain one (and sets the root level), so all
    later Kosmulator messages lost their "INFO |" / "WARNING |" prefix. The
    previous handlers and level are restored.
    """
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    try:
        return zeus.EnsembleSampler(*args, **kwargs)
    finally:
        for h in list(root.handlers):
            if h not in handlers:
                root.removeHandler(h)
        for h in handlers:
            if h not in root.handlers:
                root.addHandler(h)
        root.setLevel(level)


def _limit_main_blas(uses_pool: bool, engine: str = "", vectorised: bool = True, label: str = "") -> None:
    """
    Groups that sample in this process without a worker pool (vectorised zeus and
    emcee, or --num_cores 1) use constants.MAIN_BLAS_THREADS_NO_POOL BLAS/OpenMP
    threads; run_mcmc restores the previous setting when the group ends. Logs one
    line saying where the likelihood is evaluated.
    """
    global _BLAS_LIMITER
    n = getattr(K, "MAIN_BLAS_THREADS_NO_POOL", None)
    if uses_pool:
        log.info("[%s] %s: likelihood evaluated by the worker pool", label, engine)
        return
    log.info("[%s] %s: runs in this process (%s), BLAS threads: %s", label, engine,
             "vectorised" if vectorised else "one walker per call", n if n else "library default")
    if not n or _BLAS_LIMITER is not None:
        return
    try:
        from threadpoolctl import threadpool_limits
        _BLAS_LIMITER = threadpool_limits(limits=int(n))
    except Exception:
        _BLAS_LIMITER = None


def _restore_main_blas() -> None:
    global _BLAS_LIMITER
    if _BLAS_LIMITER is not None:
        try:
            _BLAS_LIMITER.restore_original_limits()
        except Exception:
            pass
        _BLAS_LIMITER = None


def run_mcmc(*args, **kwargs):
    """
    Run MCMC sampling for one observation group (see _run_mcmc_impl), then
    report any r_d fallbacks (EH98 instead of CLASS, rejected points) that
    happened in this process during the run. Pool workers report their own
    fallbacks through rd_helpers warnings.
    """
    RD.reset_rd_fallback_counts()
    _FAILED_POINTS.update(failed=0, evaluated=0)
    try:
        return _run_mcmc_impl(*args, **kwargs)
    finally:
        _restore_main_blas()
        _remove_shared_files()
        label = kwargs.get("obs_key") or "+".join(map(str, kwargs.get("obs") or []))
        counts = RD.rd_fallback_counts()
        if counts:
            log.warning(
                "[%s | %s] r_d fallbacks during sampling (this process): %s",
                kwargs.get("model_name", "?"), label,
                "; ".join(f"{k}: {v}" for k, v in counts.items()),
            )
        nf, ne = _FAILED_POINTS["failed"], _FAILED_POINTS["evaluated"]
        if nf:
            log.warning(
                "[%s | %s] %d of %d points evaluated during sampling (%.2f%%) could not be "
                "computed (CLASS or likelihood error: log-posterior <= %.0e) and were rejected; "
                "the reasons are in the warnings above (pool workers: in their output).",
                kwargs.get("model_name", "?"), label, nf, ne, 100.0 * nf / max(ne, 1),
                K.LOGLIKE_FAILED_BELOW,
            )


def _run_mcmc_impl(
    data,
    saveChains,
    chain_path,
    overwrite,
    MODEL_func,
    CONFIG,
    autoCorr,
    parallel,
    model_name,
    obs,
    Type,
    colors,
    convergence,
    last_obs,
    PLOT_SETTINGS,
    obs_index,
    use_mpi,
    num_cores,
    pool,
    vectorised,
    resumeChains=False,
    obs_key=None,
):
    """
    Run MCMC sampling (Zeus preferred for vectorised models, else emcee).
    """
    
    def _choose_engine(
        can_vec: bool,
        model_name: str,
        has_cmb: bool,
        has_bbn: bool,
    ) -> str:
        """
        Decide which MCMC engine to use for THIS observation group.

        Modes:
          - 'single'  : one engine per model, from K.engine_for_model.
          - 'mixed'   : same, but cross-engine chain reuse is allowed.
          - 'fastest' : per-observation choice (Zeus for simple LSS; EMCEE for CMB/BBN).
        """
        mode = getattr(K, "engine_mode", "mixed")

        # Hard CLI overrides always win
        if getattr(K, "force_emcee", False):
            return "emcee"
        
        if getattr(K, "force_zeus", False) and zeus is not None:
            return "zeus"

        # In 'single' or 'mixed', we obey the per-model main engine decided in MCMC_setup.
        if mode in ("single", "mixed"):
            eng_map = getattr(K, "engine_for_model", {})
            eng = eng_map.get(model_name)
            if eng in ("zeus", "emcee"):
                return eng
            # Fallback if somehow not set (shouldn't happen)
            return "zeus" if (can_vec and zeus is not None) else "emcee"

        if mode == "fastest":
            # Fastest logic (unchanged):
            #   - If model can vectorise AND this obs-set has no CMB/BBN → Zeus
            #   - Otherwise → EMCEE
            if (not has_cmb) and (not has_bbn) and can_vec and (zeus is not None):
                return "zeus"
            else:
                return "emcee"

        # Unknown mode → behave like a 'mixed' fallback
        return "zeus" if (can_vec and zeus is not None) else "emcee"
    
    # Resolved observation label (prefer the precomputed key from caller)
    try:
        _resolved_key = obs_key or utils.generate_label(
            obs, config_model=CONFIG, obs_index=obs_index
        )
    except Exception:
        _resolved_key = "+".join(obs) if isinstance(obs, (list, tuple)) else str(obs)
    _resolved_label = str(_resolved_key).replace("+", "_")

    # Identify if this run includes any CMB dataset
    obs_lower = [str(o).lower() for o in (obs or [])]
    has_cmb   = any(o.startswith("cmb_") for o in obs_lower)
    
    # Detect BBN in this observation set
    has_bbn_type = any(
        (str(t).lower().startswith("bbn") or "bbn" in str(t).lower())
        for t in (Type or [])
    )
    has_bbn_tag = any(
        str(o).lower().startswith("bbn") or "bbn" in str(o).lower()
        for o in (obs or [])
    )
    has_bbn = has_bbn_type or has_bbn_tag


    # 1) Parameter / run config
    param_names = CONFIG["parameters"][obs_index]
    reference_vals   = CONFIG["reference_values"][obs_index]
    prior_map   = CONFIG["prior_limits"][obs_index]
    nsteps      = CONFIG["nsteps"]
    burn        = CONFIG["burn"]
    nwalker     = CONFIG["nwalker"]
    ndim        = CONFIG["ndim"][obs_index]

    # 2) Build negative log-posterior
    nlp = lambda th: neg_log_prob(
        th, data, CONFIG, MODEL_func, model_name, obs, Type, obs_index
    )

    # 3) SciPy optimisation for IC (quiet)
    do_ic = not (
        saveChains
        and resumeChains
        and os.path.exists(chain_path)
        and not overwrite
        and any(str(t).startswith("CMB") for t in Type)
    )
    pos0     = None
    sol_diag = None
    if do_ic:
        bounds = [prior_map[p] for p in param_names]
        # The pre-fit only centres the initial ball; the burn-in does the rest. A CMB
        # point costs seconds of serial CLASS + Planck time while the pool waits, so
        # CMB groups get at most constants.PREFIT_MAXFEV_CMB calls (item 34);
        # KOSM_OPT_MAXFUN overrides both caps.
        maxfun_default = int(K.PREFIT_MAXFEV_CMB) if has_cmb else 2000
        ic, sol_diag = optimise_initial_guess(
            reference_vals,
            bounds,
            nlp,
            maxiter=int(os.environ.get("KOSM_OPT_MAXITER", "2000")),
            maxfun=int(os.environ.get("KOSM_OPT_MAXFUN", str(maxfun_default))),
            disp=False,
            quiet_cap=has_cmb,
        )
        print(f"SciPy optimized IC: {ic}\n")

        jitter_frac = PLOT_SETTINGS.get("init_jitter_frac", 0.10)
        rng = np.random.default_rng(PLOT_SETTINGS.get("seed", None))
        pos0, spans, lows, highs, init_std = make_initial_positions(
            ic,
            prior_map,
            param_names,
            nwalker,
            rng,
            base_frac=jitter_frac,
            hessian_diag=sol_diag,
            sigma_frac=float(PLOT_SETTINGS.get("init_sigma_frac", 0.5)),
        )

        # Evaluate posterior at initial positions and regenerate if needed
        post0 = batch_post(
            pos0, data, CONFIG, MODEL_func, model_name, obs, Type, obs_index
        )
        pos0 = regenerate_invalid_walkers(
            pos0,
            post0,
            ic,
            lows,
            highs,
            spans,
            rng,
            PLOT_SETTINGS,
            obs,
            Type,
            data,
            CONFIG,
            MODEL_func,
            model_name,
            obs_index,
            scale=init_std,
        )

    # ── Zeus branch ────────────────────────────────────────────────────────────
    engine   = _choose_engine(vectorised, model_name, has_cmb, has_bbn)
    if engine == "zeus" and int(ndim) == 1:
        # zeus moves a walker along the difference of two other walkers. With one
        # parameter two walkers can be arbitrarily close, the slice then needs more
        # than zeus's 10^4 expansions and the run stops ("Number of expansions
        # exceeded"; 7 of 8 runs of 3000 steps on a 1-D Gaussian, zeus 2.5.4).
        # One-parameter groups (e.g. LCDM with one uncalibrated SN set) use emcee.
        log.info("[%s | %s] one sampled parameter: emcee instead of zeus "
                 "(zeus's differential move is unreliable in one dimension)", model_name, _resolved_key)
        engine = "emcee"
    use_zeus = (engine == "zeus" and zeus is not None)
    
    if use_zeus:
        zeus_chain = chain_path.replace(".h5", "_zeus.h5")
        exists     = os.path.exists(zeus_chain)
        do_resume  = saveChains and resumeChains
        do_overw   = saveChains and overwrite

        if do_overw and exists:
            try:
                os.remove(zeus_chain)
            except FileNotFoundError:
                pass
            exists = False

        # Fast load of completed chain when not resuming
        if saveChains and exists and not do_resume:
            print(f"[INFO] Zeus: loading chain from {zeus_chain}\n")
            with h5py.File(zeus_chain, "r") as f:
                samples = f["samples"][:]  # (iters, nwalker, ndim)
            return samples[burn:, :, :].reshape(-1, ndim)

        # Determine fresh vs resume
        if do_resume and exists:
            with h5py.File(zeus_chain, "r") as f:
                old = f["samples"][:]
            done = old.shape[0]
            log.info(f"[RESUME][Zeus] Loaded {done} steps from {zeus_chain}")
            if done >= nsteps:
                return old[burn:, :, :].reshape(-1, ndim)
            pos0         = old[-1]
            steps_to_run = nsteps - done
        else:
            if pos0 is None:
                lows  = np.array([prior_map[p][0] for p in param_names], dtype=float)
                highs = np.array([prior_map[p][1] for p in param_names], dtype=float)
                span  = np.maximum(highs - lows, 1e-12)
                ic    = np.clip(np.array(reference_vals, dtype=float), lows, highs)
                rng   = np.random.default_rng(PLOT_SETTINGS.get("seed", None))
                pos0  = ic + 0.05 * span * rng.normal(size=(nwalker, ndim))
                pos0  = np.clip(pos0, lows, highs)
            done         = 0
            steps_to_run = nsteps

        # Decide whether this *observation set* can be treated as vectorised for Zeus.

        # 1) Hard override: --force_vectorisation means "always vectorise".
        if getattr(K, "force_vectorisation", False):
            zeus_vectorize = True
        else:
            zeus_vectorize = bool(vectorised)   # model-level ability

            if not zeus_vectorize:
                if (not has_cmb) and (not has_bbn):
                    zeus_vectorize = True

            # 2) Expensive late-time likelihoods (official JLA): use the pool
            if zeus_vectorize and (pool is not None) and any(
                o in getattr(K, "POOL_PREFERRED_DATASETS", set()) for o in obs
            ):
                zeus_vectorize = False

        # Pool usage:
        # - zeus_vectorize=False  → use Pool (parallel across cores)
        # - zeus_vectorize=True   → no Pool (vectorised, single-core)
        pool_for_zeus = None if zeus_vectorize else pool
        _limit_main_blas(pool_for_zeus is not None, "zeus", zeus_vectorize, f"{model_name} | {_resolved_key}")
        
        buffer_after_burn = int(
            PLOT_SETTINGS.get("autocorr_buffer_after_burn", 0)
        )
        iters_per_cb = int(PLOT_SETTINGS.get("autocorr_check_every", 100))

        # Precision switch for CMB
        switch_iter_local = max(1, burn - done) if has_cmb else None
        if has_cmb:
            try:
                SP.set_precision(
                    lmax_cap=1200,
                    accuracy_boost=0.5,
                    lAccuracyBoost=0.5,
                    lSampleBoost=0.5,
                    accurate_lensing=0,
                )
            except Exception:
                pass

        # Convergence monitor: one stopping rule for zeus and emcee (utils.ConvergenceMonitor),
        # checked on the global chain after burn-in, earlier steps of a resumed run included.
        prefix_chain = None
        if done > 0:
            try:
                with h5py.File(zeus_chain, "r") as f:
                    prefix_chain = np.asarray(f["samples"][:done], dtype=float)
            except Exception:
                prefix_chain = None
        monitor = utils.ConvergenceMonitor(
            burn=burn,
            rules=utils.convergence_rules(PLOT_SETTINGS, convergence),
            check_every=iters_per_cb,
            earliest_stop=burn + buffer_after_burn,
            consecutive=int(PLOT_SETTINGS.get("tau_consecutive", 2)),
            param_names=list(param_names),
        )
        plot_path = os.path.join(
            PLOT_SETTINGS["autocorr_save_path"], model_name, "auto_corr",
            f"{_resolved_label}.png",
        )
        import Plots.Plots as MP   # lazy: Plots imports Kosmulator_main modules
        plot_cb = lambda mon: MP.convergence_plot(mon, plot_path, model_name, _resolved_key, PLOT_SETTINGS)

        # Append writer for HDF (injected into the callback)
        writer = None
        if saveChains:
            writer = utils.AppendProgressCallback(
                filename=zeus_chain, ncheck=iters_per_cb
            )

        callbacks = utils.make_zeus_callbacks(
            monitor,
            ncheck=iters_per_cb,
            done=done,
            prefix_chain=prefix_chain,
            plot_func=plot_cb,
            append_writer=writer,
            precision_switch_iter=switch_iter_local,
            fine_kwargs={
                "lmax_cap": None,
                "accuracy_boost": 1.0,
                "lAccuracyBoost": 1.0,
                "lSampleBoost": 1.0,
                "accurate_lensing": 1,
            }
            if has_cmb
            else None,
            debug=bool(PLOT_SETTINGS.get("debug", False)),
        )

        # Optional: τ probe for CMB (debug only)
        if has_cmb and PLOT_SETTINGS.get("debug_tau_probe", False):
            try:
                params = CONFIG["parameters"][obs_index]
                p0_map = {
                    k: float(v)
                    for k, v in zip(params, np.asarray(pos0[0]).ravel())
                }
                if "tau_reio" in p0_map:
                    probe_fn = None
                    if any("cmb_lowl" in o for o in obs_lower):
                        probe_fn = SP.cmb_lowl_loglike
                    elif any("cmb_hil" in o for o in obs_lower):
                        probe_fn = SP.cmb_hil_loglike
                    elif any("cmb_lensing" in o for o in obs_lower):
                        probe_fn = SP.cmb_lensing_loglike
                    if probe_fn is not None:
                        _ = float(probe_fn(p0_map))
                        p1 = dict(p0_map)
                        p1["tau_reio"] = p1["tau_reio"] + 0.01
                        _ = float(probe_fn(p1))
            except Exception:
                pass

        # Run Zeus
        start = time.time()
        logprob_fn = (
            _zeus_logpost_vectorized if zeus_vectorize else _zeus_logpost_scalar
        )
        zeus_args = (data, CONFIG, MODEL_func, model_name, obs, Type, obs_index)
        if pool_for_zeus is not None:
            # Pool: the arguments travel once per group (file), each task only theta
            logprob_fn = _pool_logprob(logprob_fn, zeus_args, zeus_chain)
            zeus_args = ()

        # Warm-up (fresh runs only, nothing saved): reset walkers that sit far
        # below the ensemble before the recorded chain starts. A zeus walker that
        # starts beyond a likelihood barrier cannot step out across it.
        n_warm = int(PLOT_SETTINGS.get("zeus_warmup_steps", 200) or 0)
        if done == 0 and n_warm > 0 and pos0 is not None:
            try:
                pos0, n_reset = _zeus_warmup(
                    np.asarray(pos0, float), logprob_fn,
                    zeus_args,
                    nwalker, ndim, zeus_vectorize, pool_for_zeus, n_warm,
                    int(PLOT_SETTINGS.get("zeus_warmup_segment", 25)),
                    np.array([prior_map[p][0] for p in param_names], float),
                    np.array([prior_map[p][1] for p in param_names], float),
                    np.random.default_rng(PLOT_SETTINGS.get("seed", None)),
                )
                if n_reset:
                    log.info("[%s | %s] zeus warm-up: %d walker reset(s) in %d steps",
                             model_name, _resolved_key, n_reset, n_warm)
            except Exception as e:
                log.warning("zeus warm-up skipped (%s); starting from the initial ball", e)

        sampler = _zeus_ensemble(
            nwalker,
            ndim,
            logprob_fn,
            args=zeus_args,
            pool=_counting_pool(pool_for_zeus, zeus_vectorize),
            vectorize=zeus_vectorize,
        )
        try:
            sampler.run_mcmc(pos0, steps_to_run, callbacks=callbacks)
        finally:
            log.info("Zeus took %s", utils.format_elapsed_time(time.time() - start))
            try:
                if writer is not None:
                    writer(
                        sampler.iteration,
                        sampler.get_chain(flat=False),
                        sampler.get_log_prob(),
                        force=True,
                    )
            except Exception:
                pass

        # Read back and ensure > burn rows (rare extension)
        with h5py.File(zeus_chain, "r") as f:
            all_samples = f["samples"][:]
        if all_samples.shape[0] <= burn:
            extend = max(buffer_after_burn, iters_per_cb)
            sampler.run_mcmc(None, extend, callbacks=callbacks)
            try:
                if writer is not None:
                    writer(
                        sampler.iteration,
                        sampler.get_chain(flat=False),
                        sampler.get_log_prob(),
                        force=True,
                    )
            except Exception:
                pass
            with h5py.File(zeus_chain, "r") as f:
                all_samples = f["samples"][:]
        # Save the post-burn-in log-probabilities, flattened step-major so they
        # line up with all_samples[burn:].reshape(-1, ndim) returned below.
        # (zeus's own get_log_prob(flat=True) is walker-major, order='F'.)
        if saveChains:
            with h5py.File(zeus_chain, "a") as h5f:
                try:
                    flat_log_prob = utils.zeus_flat_log_like(h5f, burn, n_rows=all_samples.shape[0])
                    if flat_log_prob is None:
                        raise ValueError("no log-probabilities aligned with the stored samples")
                    for key in ("log_prob", "log_like"):
                        if key in h5f:
                            del h5f[key]
                    h5f.create_dataset("log_prob", data=flat_log_prob)
                    # Flat top-hat priors everywhere → log_prob == log_like exactly
                    h5f.create_dataset("log_like", data=flat_log_prob)
                    h5f.attrs["log_like_order"] = "step"
                except Exception as e:
                    print(f"\n[WARNING] Could not save 'log_prob'/'log_like' from Zeus: {e}")

        # Verdict of the convergence rule on the full recorded chain, with the
        # numbers behind each condition (also drawn in the final plot).
        st = monitor.final(all_samples.shape[0], all_samples)
        converged = bool(st.get("converged", False))
        try:
            plot_cb(monitor)
        except Exception:
            pass
        (log.info if converged else log.warning)(
            monitor.message(st, model_name, _resolved_key, "zeus")
        )
        # Stuck-walker check on the recorded post-burn-in chain
        stuck = []
        try:
            with h5py.File(zeus_chain, "r") as f:
                if "log_prob_chain" in f:
                    stuck = utils.find_stuck_walkers(f["log_prob_chain"][burn:all_samples.shape[0]], ndim)
        except Exception:
            stuck = []
        if len(stuck):
            log.warning(
                "[%s | %s] %d zeus walker(s) stayed far below the ensemble after burn-in "
                "(walkers %s). Means, DIC and WAIC are unreliable for this group; rerun "
                "(the warm-up resets such walkers) or use --force_emcee.",
                model_name, _resolved_key, len(stuck), list(map(int, stuck)),
            )
        try:
            with h5py.File(zeus_chain, "a") as f:
                f.attrs["converged"] = bool(converged)
                f.attrs["stuck_walkers"] = np.asarray(stuck, dtype=int)
        except Exception:
            pass

        return all_samples[burn:, :, :].reshape(-1, ndim)

    # ── emcee branch ────────────────────────────────────────────────────────────
    else:
        backend = None

        # Vectorisable groups without CMB, BBN or a pool-preferred dataset run in this
        # process with emcee's vectorize=True (one likelihood call per half-ensemble,
        # as for zeus). The worker pool is kept for CMB/BBN groups (CLASS per walker)
        # and non-vectorisable models, where one likelihood call is expensive.
        emcee_vectorize = (
            bool(vectorised or getattr(K, "force_vectorisation", False))
            and not (has_cmb or has_bbn)
            and not any(o in getattr(K, "POOL_PREFERRED_DATASETS", set()) for o in (obs or []))
        )
        emcee_pool = None if emcee_vectorize else pool
        emcee_fn = emcee_prob_vectorized if emcee_vectorize else emcee_prob
        emcee_args = (data, Type, CONFIG, MODEL_func, model_name, obs, obs_index)
        if emcee_pool is not None:
            # Pool: the arguments travel once per group (file), each task only theta
            emcee_fn = _pool_logprob(emcee_prob, emcee_args, chain_path)
            emcee_args = ()
        _limit_main_blas(emcee_pool is not None, "emcee", emcee_vectorize, f"{model_name} | {_resolved_key}")

        # ------------------------------------------------------------
        # Backend setup
        # ------------------------------------------------------------
        if saveChains:
            if overwrite and os.path.exists(chain_path):
                try:
                    os.remove(chain_path)
                except FileNotFoundError:
                    pass
            backend = emcee.backends.HDFBackend(chain_path)

        # ------------------------------------------------------------
        # Debug / print controls (from CLI → CONFIG["debug"])
        # ------------------------------------------------------------
        dbg = CONFIG.get("debug", {}) if isinstance(CONFIG, dict) else {}
        print_enabled = bool(dbg.get("print_loglike", False))
        print_every = int(dbg.get("print_loglike_every", 1) or 1)
        print_every = max(1, print_every)

        # ------------------------------------------------------------
        # Resume if requested and chain exists
        # ------------------------------------------------------------
        if saveChains and resumeChains and os.path.exists(chain_path) and not overwrite:
            backend = emcee.backends.HDFBackend(chain_path)
            current = backend.iteration
            print(
                f"[RESUME] Found existing chain with {current} steps. Resuming to {nsteps}."
            )

            if current >= nsteps:
                return backend.get_chain(discard=burn, flat=True)

            print(f"\nInitialising ensemble of {nwalker} walkers...")
            try:
                last_state = backend.get_chain(flat=False)[-1]
            except IndexError:
                last_state = None

            if last_state is not None:
                sampler = emcee.EnsembleSampler(
                    nwalker,
                    ndim,
                    emcee_fn,
                    args=emcee_args,
                    pool=_counting_pool(emcee_pool, emcee_vectorize),
                    backend=backend,
                    vectorize=emcee_vectorize,
                )

                if autoCorr:
                    local_burn = max(0, burn - current)
                    obs_for_plot = [_resolved_key]

                    res = utils.emcee_autocorr_stopping(
                        last_state,
                        sampler,
                        nsteps - current,
                        model_name,
                        colors,
                        obs_for_plot,
                        PLOT_SETTINGS,
                        convergence=convergence,
                        last_obs=last_obs,
                        resume_offset=current,
                        local_burn=local_burn,
                        global_burn=burn,
                        buffer_after_burn=PLOT_SETTINGS.get(
                            "autocorr_buffer_after_burn", 0
                        ),
                        print_enabled=print_enabled,
                        print_every=print_every,
                        param_names=list(param_names),
                    )
                    flat_samples = res[0] if isinstance(res, tuple) else res
                else:
                    # Manual loop to allow step printing (Pool-safe)
                    for state in sampler.sample(
                        last_state, iterations=nsteps - current, progress=True
                    ):
                        it = current + sampler.iteration
                        if (
                            print_enabled
                            and (it % print_every) == 0
                            and utils.is_rank0()
                            and utils.is_main_process()
                        ):
                            lp = getattr(state, "log_prob", None)
                            if lp is None:
                                try:
                                    lp_all = sampler.get_log_prob()
                                    lp = lp_all[-1] if getattr(lp_all, "ndim", 0) > 1 else lp_all
                                except Exception:
                                    lp = None
                            if lp is not None:
                                lp = np.asarray(lp, dtype=float)
                                if lp.size:
                                    tqdm.write(
                                        f"[EMCEE Step {it}] Log-Post: "
                                        f"Max={np.nanmax(lp):.4f} | Mean={np.nanmean(lp):.4f}"
                                    )

                    flat_samples = sampler.get_chain(discard = burn, flat = True)
                #Extract and write log_prob + loglike on resume
                if saveChains:
                    flat_log_prob = sampler.get_log_prob(discard = burn, flat = True)
                    blobs = sampler.get_blobs(discard = burn, flat = True)
                    with h5py.File(chain_path, "a") as h5f:
                        # Resumed run: flag it only if the convergence rule was met
                        h5f.attrs["converged"] = bool(getattr(sampler, "kosm_converged", False))
                        if "log_prob" in h5f:
                            del h5f["log_prob"]
                        h5f.create_dataset("log_prob",data = flat_log_prob)

                        if blobs is not None:
                            flat_log_like = np.squeeze(
                                np.asarray(blobs, dtype = float)
                            )
                            if "log_like" in h5f:
                                del h5f["log_like"]
                            h5f.create_dataset("log_like", data = flat_log_like)
            return flat_samples

        # ------------------------------------------------------------
        # Fresh emcee run
        # ------------------------------------------------------------
        if pos0 is None:
            lows  = np.array([prior_map[p][0] for p in param_names], dtype=float)
            highs = np.array([prior_map[p][1] for p in param_names], dtype=float)
            span  = np.maximum(highs - lows, 1e-12)
            ic    = np.clip(np.array(reference_vals, dtype=float), lows, highs)
            rng   = np.random.default_rng(PLOT_SETTINGS.get("seed", None))
            pos0  = ic + 0.05 * span * rng.normal(size=(nwalker, ndim))
            pos0  = np.clip(pos0, lows, highs)

        start = time.time()
        print(f"\nInitialising ensemble of {nwalker} walkers...")
        sampler = emcee.EnsembleSampler(
            nwalker,
            ndim,
            emcee_fn,
            args=emcee_args,
            pool=_counting_pool(emcee_pool, emcee_vectorize),
            backend=backend,
            vectorize=emcee_vectorize,
        )

        if autoCorr:
            obs_for_plot = [_resolved_key]
            res = utils.emcee_autocorr_stopping(
                pos0,
                sampler,
                nsteps,
                model_name,
                colors,
                obs_for_plot,
                PLOT_SETTINGS,
                convergence=convergence,
                last_obs=last_obs,
                resume_offset=0,
                local_burn=burn,
                global_burn=burn,
                buffer_after_burn=PLOT_SETTINGS.get(
                    "autocorr_buffer_after_burn", 0
                ),
                print_enabled=print_enabled,
                print_every=print_every,
                param_names=list(param_names),
            )
            flat_samples = res[0] if isinstance(res, tuple) else res
            flat_log_prob = sampler.get_log_prob(discard = burn, flat = True)
            blobs = sampler.get_blobs(discard = burn, flat = True)
        else:
            # Manual loop to allow step printing (Pool-safe)
            for state in sampler.sample(pos0, iterations=nsteps, progress=True):
                it = sampler.iteration
                if (
                    print_enabled
                    and (it % print_every) == 0
                    and utils.is_rank0()
                    and utils.is_main_process()
                ):
                    lp = getattr(state, "log_prob", None)
                    if lp is None:
                        try:
                            lp_all = sampler.get_log_prob()
                            lp = lp_all[-1] if getattr(lp_all, "ndim", 0) > 1 else lp_all
                        except Exception:
                            lp = None
                    if lp is not None:
                        lp = np.asarray(lp, dtype=float)
                        if lp.size:
                            tqdm.write(
                                f"[EMCEE Step {it}] Log-Post: "
                                f"Max={np.nanmax(lp):.4f} | Mean={np.nanmean(lp):.4f}"
                            )

            flat_samples = sampler.get_chain(discard=burn, flat=True)
            flat_log_prob = sampler.get_log_prob(discard=burn, flat=True)
            blobs = sampler.get_blobs(discard = burn, flat = True)

        print(f"Emcee sampling took {utils.format_elapsed_time(time.time() - start)}\n")

        if saveChains:
            # Flag the chain only if the autocorrelation rule was actually met
            # (previously every finished run was flagged converged).
            with h5py.File(chain_path, "a") as h5f:
                h5f.attrs["converged"] = bool(getattr(sampler, "kosm_converged", False))

                #Saving the log_probs to use for stats
                if "log_prob" in h5f:
                    del h5f["log_prob"]
                h5f.create_dataset("log_prob", data = flat_log_prob)

                #Save raw log_like blob
                if blobs is not None:
                    flat_log_like = np.squeeze(np.asarray(blobs, dtype = float))
                    if "log_like" in h5f:
                        del h5f["log_like"]
                    h5f.create_dataset("log_like", data = flat_log_like)

        return flat_samples


def load_mcmc_results(output_path: str, file_name: str, CONFIG: dict):
    """Load samples and saved likelihood values from an emcee chain."""
    chain_path = os.path.join(output_path, file_name)
    backend = emcee.backends.HDFBackend(chain_path)
    burn = CONFIG.get("burn", 0)
    samples = backend.get_chain(discard=burn, flat=True)

    loglike = None
    with h5py.File(chain_path, "r") as h5f:
        if "log_like" in h5f:
            candidate = np.asarray(h5f["log_like"], dtype=float).reshape(-1)
            if candidate.size == samples.shape[0]:
                loglike = candidate

    return {"samples": samples, "loglike": loglike}
