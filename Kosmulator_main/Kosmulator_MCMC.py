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

    params = CONFIG["parameters"][obs_index]
    param_dict = {p: theta[i] for i, p in enumerate(params)}

    # Derive background vars if using CMB params / 100theta_s
    param_dict = utils.ensure_background_params(param_dict)

    # If BBN is present in this observation set and r_d wasn't sampled,
    # compute r_d from (Omega_bh^2, Omega_m, H_0[, N_eff]).
    RD._maybe_calibrate_rd(param_dict, CONFIG, obs_index)

    # Dedicated CMB branches
    if obs == "CMB_hil":
        return SP.cmb_hil_loglike(param_dict, model_name)
    if obs == "CMB_hil_TT":
        return SP.cmb_hilTT_loglike(param_dict, model_name)
    if obs == "CMB_lowl":
        return SP.cmb_lowl_loglike(param_dict, model_name)
    if obs == "CMB_lensing":
        # Decide "raw" vs "cmbmarged" based on what this group contains.
        # Plumbing for cmbmarged stays in place but we currently always run RAW
        # (the CMB-marged lensing likelihood is not yet wired in fully).
        groups = CONFIG.get("observations", [])
        group = groups[obs_index] if obs_index < len(groups) else obs
        if not isinstance(group, (list, tuple)):
            group = [group]

        has_primary_cmb = any(g in {"CMB_hil", "CMB_hil_TT", "CMB_lowl"} for g in group)
        # For now we *always* use RAW, but we keep the plumbing ready:
        mode = "raw"  # if has_primary_cmb else "cmbmarged"   # <- future switch
        SP.set_lensing_mode(mode)

        grp_str = "+".join(map(str, group))
        _last = getattr(SP, "_last_lensing_mode_logged", None)
        if mode != _last:
            log.info("[CMB-lensing switch] group=%s \u2192 mode=%s", grp_str, mode)
            SP._last_lensing_mode_logged = mode

        # This returns a scalar log-likelihood
        return SP.cmb_lensing_loglike(param_dict, model_name)


    # BAO / DESI: enforce r_d policy
    if obs == "BAO":
        return -0.5 * SP.Calc_BAO_chi(obs_data, MODEL_func, param_dict, obs_type)

    if obs in ("DESI_DR1", "DESI_DR2"):
        return -0.5 * SP.Calc_DESI_chi(obs_data, MODEL_func, param_dict, obs_type)

    # BBN: either full DH dataset or prior
    if obs in ("BBN_DH", "BBN_DH_AlterBBN") or obs_type == "BBN_DH":
        return -0.5 * SP.Calc_BBN_DH_chi(obs_data, MODEL_func, param_dict, "BBN_DH")

    if obs in ("BBN_PryMordial", "BBN_prior"):
        obh2 = float(param_dict["Omega_bh^2"])
        # 2D case if both mean for N_eff and covariance are provided
        if ("cov" in obs_data) and ("mu_Neff" in obs_data):
            x   = np.array([obh2, float(param_dict["N_eff"])], dtype=float)
            mu  = np.array([float(obs_data["mu_obh2"]), float(obs_data["mu_Neff"])], dtype=float)
            cov = np.array(obs_data["cov"], dtype=float)
            try:
                inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(cov, rcond=1e-12)
            d = x - mu
            return -0.5 * float(d @ (inv @ d))
        # 1D default (ΛCDM prior on Ω_b h^2)
        mu, sig = float(obs_data["mu_obh2"]), float(obs_data["sigma_obh2"])
        return -0.5 * ((obh2 - mu) / sig) ** 2

    # Non-CMB standard data containers
    if obs in ("PantheonP", "PantheonPS"):
        z    = obs_data["zHD"]
        mb   = obs_data["m_b_corr"]
        trig = obs_data["IS_CALIBRATOR"]
        cep  = obs_data["CEPH_DIST"]
        cov  = obs_data.get("cov")
    else:
        z         = obs_data["redshift"]
        type_data = obs_data["type_data"]
        type_err  = obs_data["type_data_error"]

    # Predictions per type
    if obs_type == "SNe":
        d_c = utils.Comoving_distance_vectorized(MODEL_func, z, param_dict)
        y_dl = d_c * (1.0 + z)
        if (not np.isfinite(y_dl).all()) or (np.min(y_dl) <= 0):
            return -np.inf
        model = 25.0 + 5.0 * np.log10(y_dl)

    elif obs_type in ["OHD", "CC"]:
        E_z = utils.E_of_z(z, MODEL_func, param_dict)
        if (not np.isfinite(E_z).all()) or np.any(E_z <= 0):
            return -np.inf
        model = param_dict["H_0"] * E_z

    elif obs_type in ["f_sigma_8", "f"]:
        Omz = utils.matter_density_z_array(z, param_dict, MODEL_func)
        if (not np.isfinite(Omz).all()) or np.any(Omz <= 0):
            return -np.inf

        if obs_type == "f_sigma_8":
            # read fixed gamma for this obs_index if it was removed from params
            gamma = param_dict.get("gamma", None)
            if gamma is None:
                gamma = CONFIG.get("fs8_gamma_fixed_by_group", {}).get(obs_index, K.GAMMA_FS8_SINGLETON)
            gamma = float(gamma)

            integ = utils.integral_term_array(z, param_dict, MODEL_func, gamma)
            if not np.isfinite(integ).all():
                return -np.inf
            model = param_dict["sigma_8"] * (Omz ** gamma) * np.exp(-integ)
        else:
            # f-only still samples gamma (no fix) unless you choose otherwise
            model = Omz ** float(param_dict["gamma"])

    else:
        return -np.inf

    if obs in ("PantheonP", "PantheonPS"):
        chi2 = SP.Calc_PantP_chi(mb, trig, cep, cov, model, param_dict)
    else:
        chi2 = SP.Calc_chi(obs_type, type_data, type_err, model)
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

    # 2) per-parameter restrictions (1D predicates)
    restr = CONFIG.get("restrictions", {})
    for i, p in enumerate(params):
        if p in restr:
            valid = np.array([restr[p](v) for v in theta_batch[:, i]])
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

    for obs_name, obs_type in zip(obs, Type):
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
    
    # Check priors
    if not np.all(np.isfinite(lp)):
        # Return -inf for posterior, and NaN for likelihood (blob)
        return -np.inf, np.nan 
    
    ll = log_likelihood_all(
        arr, data, CONFIG, MODEL_func, model_name, obs, Type, obs_index
    )
    
    log_post = float(lp + ll)
    log_like = float(ll)
    
    return log_post, log_like

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
    """Zeus log-posterior for vectorised models (always returns 1D array)."""
    arr = np.atleast_2d(theta)
    out = batch_post(arr, data, CONFIG, MODEL_func, model_name, obs, Type, obs_index)
    return np.asarray(out, dtype=float).ravel()


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
    return -float(lp + ll)


def optimise_initial_guess(
    true_vals: np.ndarray,
    bounds: List[Tuple[float, float]],
    nlp_fn: Callable[[np.ndarray], float],
    maxiter: int,
    maxfun: int,
    disp: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    L-BFGS-B to find a decent initial center; returns (ic, Hinv_diag or None).
    """
    NO_OPT = os.environ.get("KOSM_NO_OPT", "0") == "1"
    if NO_OPT:
        return np.asarray(true_vals, float), None

    sol = optimize.minimize(
        nlp_fn,
        true_vals,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "maxfun": maxfun, "disp": False},
    )
    ic = np.asarray(sol.x, float)
    Hinv = None
    try:
        Hinv_like = sol.hess_inv
        Hinv = (
            Hinv_like.todense()
            if hasattr(Hinv_like, "todense")
            else np.asarray(Hinv_like, float)
        )
        Hinv = np.asarray(Hinv, float)
        if Hinv.ndim == 2:
            Hinv = np.diag(Hinv)  # take diagonal for scale proposal
    except Exception:
        Hinv = None
    return ic, Hinv


def make_initial_positions(
    ic: np.ndarray,
    prior_map: Dict[str, Tuple[float, float]],
    param_names: List[str],
    nwalker: int,
    rng: np.random.Generator,
    base_frac: float,
    hessian_diag: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Propose initial walker cloud around 'ic', respecting priors and avoiding duplicates.
    Returns (pos0, spans, lows, highs).
    """
    lows  = np.array([prior_map[p][0] for p in param_names], float)
    highs = np.array([prior_map[p][1] for p in param_names], float)
    spans = np.maximum(highs - lows, 1e-12)

    # Ensure IC inside priors
    ic = np.clip(ic, lows, highs)

    # Distance to nearest bound per-dim
    room = np.maximum(np.minimum(ic - lows, highs - ic), 1e-12)

    # Base scale as fraction of prior width, limited by room
    base_scale = np.minimum(base_frac * spans, 0.5 * room)

    # Try anisotropic scale from inverse Hessian diag
    if (
        hessian_diag is not None
        and hessian_diag.size == len(param_names)
        and np.all(np.isfinite(hessian_diag))
    ):
        std = np.sqrt(np.maximum(hessian_diag, 1e-16))
        std = np.minimum(std, base_scale)
        pos0 = ic + rng.normal(size=(nwalker, len(param_names))) * std
    else:
        pos0 = ic + rng.normal(size=(nwalker, len(param_names))) * base_scale

    pos0 = np.clip(pos0, lows, highs)

    # De-duplicate (up to a few retries)
    for _ in range(5):
        uniq, idx = np.unique(np.round(pos0, decimals=12), axis=0, return_index=True)
        if len(idx) == nwalker:
            break
        dup_mask = np.ones(nwalker, dtype=bool)
        dup_mask[idx] = False
        pos0[dup_mask] = ic + 0.5 * base_scale * rng.normal(
            size=(dup_mask.sum(), len(param_names))
        )
        pos0 = np.clip(pos0, lows, highs)

    return pos0, spans, lows, highs


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
) -> np.ndarray:
    """
    Re-generate any invalid (non-finite posterior) walkers with shrinking radius.
    """
    bad = np.where(~np.isfinite(post0))[0]
    if not bad.size:
        return pos0

    regen_max_tries = PLOT_SETTINGS.get("init_regen_tries", 20)
    regen_shrink    = PLOT_SETTINGS.get("init_regen_shrink", 0.7)
    jitter = 0.1 * spans  # initial proposal scale

    tries = 0
    while bad.size and tries < regen_max_tries:
        nbad = bad.size
        cand = ic + rng.normal(size=(nbad, ic.size)) * jitter
        cand = np.clip(cand, lows, highs)

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
# Zeus autocorr plotting callback
# ───────────────────────────────────────────────────────────────────────────────

class ZeusAutoCorrPlotter:
    """
    Plot |Δτ| / τ versus *global* iteration. Closes its figure every call.
    """

    def __init__(
        self,
        model_name,
        obs_label,
        global_burn,
        target_tau,
        done,
        ncheck,
        out_dir,
        settings,
    ):
        import os

        self.model_name  = model_name
        self.obs_label   = obs_label
        self.global_burn = int(global_burn)
        self.target_tau  = float(target_tau)
        self.done        = int(done)
        self.ncheck      = int(ncheck)
        self.out_dir     = out_dir
        self.settings    = settings or {}
        self.x, self.y   = [], []  # global iters, |Δτ|/τ
        self._prev_tau   = None

        os.makedirs(self.out_dir, exist_ok=True)
        self.file_path = os.path.join(self.out_dir, f"{self.obs_label}.png")

    def __call__(self, estimates, iteration):
        import numpy as np
        import matplotlib.pyplot as plt

        if estimates is None:
            return

        tau = float(np.asarray(estimates, dtype=float)[-1])

        # Wait for a previous τ̂ so that Δτ is well-defined
        if self._prev_tau is None:
            self._prev_tau = tau
            return

        d_tau = abs(tau - self._prev_tau)
        frac  = d_tau / max(tau, 1e-12)
        self._prev_tau = tau

        global_iter = self.done + int(iteration)
        self.x.append(global_iter)
        self.y.append(max(frac, 1e-300))  # keep > 0 for log plots if needed

        fig, ax = plt.subplots(figsize=(7, 4), dpi=self.settings.get("dpi", 200))
        ax.plot(self.x, self.y, marker="o", ls="-", lw=1.5)
        ax.set_yscale("linear")
        ax.set_xlabel("Global iteration")
        ax.set_ylabel(r"$|\Delta \tau|/\tau$ (fractional change)")

        ax.axvline(
            self.global_burn,
            color="r",
            ls="--",
            alpha=0.6,
            label="burn",
        )
        ax.axvline(
            self.global_burn
            + self.settings.get("autocorr_buffer_after_burn", 1000),
            color="k",
            ls=":",
            alpha=0.6,
            label="burn+buffer",
        )
        ax.axhline(
            self.target_tau,
            color="0.5",
            ls="--",
            alpha=0.6,
            label=f"target frac={self.target_tau:g}",
        )

        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(self.file_path, bbox_inches="tight")
        plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────

def run_mcmc(
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
    true_vals   = CONFIG["true_values"][obs_index]
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
        ic, sol_diag = optimise_initial_guess(
            true_vals,
            bounds,
            nlp,
            maxiter=int(os.environ.get("KOSM_OPT_MAXITER", "30")),
            maxfun=int(os.environ.get("KOSM_OPT_MAXFUN", "60")),
            disp=False,
        )
        print(f"SciPy optimized IC: {ic}\n")

        jitter_frac = PLOT_SETTINGS.get("init_jitter_frac", 0.10)
        rng = np.random.default_rng(PLOT_SETTINGS.get("seed", None))
        pos0, spans, lows, highs = make_initial_positions(
            ic,
            prior_map,
            param_names,
            nwalker,
            rng,
            base_frac=jitter_frac,
            hessian_diag=sol_diag,
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
        )

    # ── Zeus branch ────────────────────────────────────────────────────────────
    engine   = _choose_engine(vectorised, model_name, has_cmb, has_bbn)
    use_zeus = (engine == "zeus" and zeus is not None)
    #print(f"[DEBUG] Engine for { _resolved_label }: {engine} (mode={getattr(K, 'engine_mode', 'mixed')}, has_cmb={has_cmb}, has_bbn={has_bbn}, can_vec={vectorised})")
    
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
                ic    = np.clip(np.array(true_vals, dtype=float), lows, highs)
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

        # Pool usage:
        # - zeus_vectorize=False  → use Pool (parallel across cores)
        # - zeus_vectorize=True   → no Pool (vectorised, single-core)
        pool_for_zeus = None if zeus_vectorize else pool
        
        buffer_after_burn = int(
            PLOT_SETTINGS.get("autocorr_buffer_after_burn", max(1000, burn // 5))
        )
        iters_per_cb = int(PLOT_SETTINGS.get("autocorr_check_every", 100))
        target_tau   = float(convergence)
        local_gate   = max(0, (burn + buffer_after_burn) - done)

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

        # Zeus plotter
        obs_label = _resolved_label
        out_dir   = os.path.join(
            PLOT_SETTINGS["autocorr_save_path"], model_name, "auto_corr"
        )
        plot_cb = ZeusAutoCorrPlotter(
            model_name,
            obs_label,
            burn,
            target_tau,
            done,
            iters_per_cb,
            out_dir,
            PLOT_SETTINGS,
        )

        # Append writer for HDF (injected into composite callback)
        writer = None
        if saveChains:
            writer = utils.AppendProgressCallback(
                filename=zeus_chain, ncheck=iters_per_cb
            )

        # Build the composite callback
        callbacks = utils.make_zeus_callbacks(
            burn=local_gate,
            nsteps=nsteps,
            target_autocorr=target_tau,
            plot_func=plot_cb,
            debug=bool(PLOT_SETTINGS.get("debug", False)),
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
            ncheck=iters_per_cb,
            append_writer=writer,
            consecutive_required=int(PLOT_SETTINGS.get("tau_consecutive", 1)),
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
        sampler = zeus.EnsembleSampler(
            nwalker,
            ndim,
            logprob_fn,
            args=(data, CONFIG, MODEL_func, model_name, obs, Type, obs_index),
            pool=pool_for_zeus,
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
                    )
            except Exception:
                pass
            with h5py.File(zeus_chain, "r") as f:
                all_samples = f["samples"][:]

        return all_samples[burn:, :, :].reshape(-1, ndim)

    # ── emcee branch ────────────────────────────────────────────────────────────
    else:
        backend = None

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
                    emcee_prob,
                    args=(data, Type, CONFIG, MODEL_func, model_name, obs, obs_index),
                    pool=pool,
                    backend=backend,
                )

                if autoCorr:
                    local_burn = max(0, burn - current)
                    obs_for_plot = [_resolved_key]

                    flat_samples = utils.emcee_autocorr_stopping(
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
                            "autocorr_buffer_after_burn", 1000
                        ),
                        print_enabled=print_enabled,
                        print_every=print_every,
                    )
                    return flat_samples
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

                    return sampler.get_chain(discard=burn, flat=True)

        # ------------------------------------------------------------
        # Fresh emcee run
        # ------------------------------------------------------------
        if pos0 is None:
            lows  = np.array([prior_map[p][0] for p in param_names], dtype=float)
            highs = np.array([prior_map[p][1] for p in param_names], dtype=float)
            span  = np.maximum(highs - lows, 1e-12)
            ic    = np.clip(np.array(true_vals, dtype=float), lows, highs)
            rng   = np.random.default_rng(PLOT_SETTINGS.get("seed", None))
            pos0  = ic + 0.05 * span * rng.normal(size=(nwalker, ndim))
            pos0  = np.clip(pos0, lows, highs)

        start = time.time()
        print(f"\nInitialising ensemble of {nwalker} walkers...")
        sampler = emcee.EnsembleSampler(
            nwalker,
            ndim,
            emcee_prob,
            args=(data, Type, CONFIG, MODEL_func, model_name, obs, obs_index),
            pool=pool,
            backend=backend,
        )

        if autoCorr:
            obs_for_plot = [_resolved_key]
            flat_samples = utils.emcee_autocorr_stopping(
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
                    "autocorr_buffer_after_burn", 1000
                ),
                print_enabled=print_enabled,
                print_every=print_every,
            )
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

        print(f"Emcee sampling took {utils.format_elapsed_time(time.time() - start)}\n")
        if saveChains:
            with h5py.File(chain_path, "a") as h5f:
                h5f.attrs["converged"] = True

        return flat_samples


def load_mcmc_results(output_path: str, file_name: str, CONFIG: dict):
    """Load a saved HDFBackend chain and return the flat samples (post-burn)."""
    backend = emcee.backends.HDFBackend(os.path.join(output_path, file_name))
    burn = CONFIG.get("burn", 0)
    return backend.get_chain(discard=burn, flat=True)
