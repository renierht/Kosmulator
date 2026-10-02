#!/usr/bin/env python3
"""
Marcel_chain.py — post-hoc statistical analysis for Marcel's MCMC chains.

Computes AIC, AICc, BIC, DIC, WAIC for all configured chains under
    MCMC_Chains/Marcel_Chains/{CONSTRAINT}/{model}/{obs_dir}/
and writes stats_summary.txt files to
    Statistical_analysis_tables/Marcel_Chains/{CONSTRAINT}/{model}/

Configure the section below, then run:
    python Marcel_chain.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

# ─── Path setup ──────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(PROJ_ROOT / "Kosmulator_main"))

# ─────────────────────────────────────────────────────────────────────────────
#  USER CONFIGURATION — edit here
# ─────────────────────────────────────────────────────────────────────────────

CHAIN_BASE    = "MCMC_Chains/Marcel_Chains"
STATS_BASE    = "Statistical_analysis_tables/Marcel_Chains"

# Which constraint sub-folder to analyse
CONSTRAINT    = "Free"          # Free | Positive | Positive_stable

# Burn-in steps to discard (adjust to match how the chains were run)
BURN          = 1000

model_names   = ["LCDM_v", "NonLinear_IDE_2"]
reference_model = "LCDM_v"

# Each inner list is one combined-likelihood run
observations  = [
    ["DESI_DR2", "PantheonPS"],
    ["BBN_PryMordial", "CC", "DESI_DR2", "Pantheon"],
    ["BBN_PryMordial", "CC", "DESI_DR2", "PantheonPS"],
]

prior_limits = {
    "Omega_m":    (0.01, 1.0),
    "Omega_b":    (0.01, 0.06),
    "H_0":        (40.0, 100.0),
    "r_d":        (0.01, 1000.0),
    "M_abs":      (-30.0, -5.0),
    "gamma":      (0.01, 1.0),
    "sigma_8":    (0.01, 1.0),
    "n":          (0.0, 0.6),
    "q0":         (-0.8, -0.01),
    "q1":         (-0.75, 1.0),
    "beta":       (0.01, 5.0),
    "tau_reio":   (0.04, 0.09),
    "Omega_dh^2": (0.05, 0.2),
    "Omega_bh^2": (0.015, 0.031),
    "ln10^10_As": (2.5, 3.5),
    "n_s":        (0.9, 1.1),
    "100theta_s": (1.035, 1.047),
    "alpha":      (0.00, 1.00),
    "B":          (0.00, 0.333),
    "f1":         (0.01, 100.0),
    "w0":         (-3.0, 1.0),
    "wa":         (-3.0, 2.0),
    "w":          (-2.0, -1.0),
    "delta":      (0.0, 0.5),
    # Add any NonLinear_IDE_2-specific parameters here, e.g.:
    # "xi":       (-1.0, 1.0),
}

reference_values = {
    "Omega_m":    0.315,
    "H_0":        67.4,
    "sigma_8":    0.8,
    "r_d":        147.5,
    "M_abs":      -19.2,
    "tau_reio":   0.054,
    "Omega_dh^2": 0.12,
    "Omega_bh^2": 0.0224,
    "n_s":        0.9624,
    "ln10^10_As": 3.045,
    "Omega_b":    0.05,
    "w0":         -1.0,
    "wa":         0.0,
}

# ─────────────────────────────────────────────────────────────────────────────
#  Imports (after path setup so Kosmulator_main is on sys.path)
# ─────────────────────────────────────────────────────────────────────────────

import h5py
import emcee

import User_defined_modules as UDM
from Kosmulator_main import Config
from Kosmulator_main.Post_processing import (
    calculate_asymmetric_from_samples,
    statistical_analysis,
    interpret_delta_IC,
)
from Kosmulator_main.utils import save_stats_to_file, generate_label

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("Marcel_chain")


# ─────────────────────────────────────────────────────────────────────────────
#  Chain loader — handles both zeus (_zeus.h5) and emcee (.h5) formats
# ─────────────────────────────────────────────────────────────────────────────

def load_chain(chain_dir: str, label: str, burn: int) -> dict | None:
    """
    Load flat_samples and log_like from {label}_zeus.h5 or {label}.h5.

    Returns {"samples": ndarray (N, k), "loglike": ndarray (N,) or None},
    or None if no chain file is found.
    """
    zeus_path  = os.path.join(chain_dir, f"{label}_zeus.h5")
    emcee_path = os.path.join(chain_dir, f"{label}.h5")

    # Zeus format: datasets "samples" (nsteps, nwalkers, ndim) and "log_like"
    if os.path.exists(zeus_path):
        with h5py.File(zeus_path, "r") as f:
            all_samples = f["samples"][:]
            if "log_like" in f:
                loglike_raw = np.array(f["log_like"], dtype=float)
            elif "log_prob" in f:
                loglike_raw = np.array(f["log_prob"], dtype=float)
            else:
                loglike_raw = None

        flat_samples = all_samples[burn:].reshape(-1, all_samples.shape[-1])
        flat_loglike = loglike_raw[burn:].reshape(-1) if loglike_raw is not None else None
        log.info("  [zeus]  %s  (%d samples after burn)", zeus_path, flat_samples.shape[0])
        return {"samples": flat_samples, "loglike": flat_loglike}

    # Emcee format: standard HDFBackend + custom "log_like" dataset
    if os.path.exists(emcee_path):
        backend = emcee.backends.HDFBackend(emcee_path, read_only=True)
        flat_samples = backend.get_chain(discard=burn, flat=True)
        flat_loglike = None
        with h5py.File(emcee_path, "r") as f:
            if "log_like" in f:
                candidate = np.asarray(f["log_like"], dtype=float).reshape(-1)
                if candidate.size == flat_samples.shape[0]:
                    flat_loglike = candidate
        log.info("  [emcee] %s  (%d samples after burn)", emcee_path, flat_samples.shape[0])
        return {"samples": flat_samples, "loglike": flat_loglike}

    log.warning("  No chain found for '%s' in %s", label, chain_dir)
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Build CONFIG and load all observation data
    log.info("Building CONFIG for models: %s", model_names)
    models = UDM.Get_model_names(model_names)
    CONFIG, data = Config.create_config(
        models=models,
        reference_values=reference_values,
        prior_limits=prior_limits,
        restrictions=UDM.Get_model_restrictions(model_names),
        coupled_restrictions=UDM.Get_model_coupled_restrictions(model_names),
        observation=observations,
        nwalkers=32,
        nsteps=1000,
        burn=BURN,
        model_name=model_names,
    )

    # 2. Load chains for each model and build the structured input for stats
    all_best_fit: dict = {}

    for model_name in model_names:
        log.info("=== Loading chains: %s ===", model_name)
        cfg = CONFIG[model_name]
        samples_for_model: dict = {}

        for obs_index, obs_group in enumerate(cfg["observations"]):
            label     = generate_label(obs_group, config_model=cfg, obs_index=obs_index)
            label_dir = label.replace("+", "_")
            chain_dir = os.path.join(CHAIN_BASE, CONSTRAINT, model_name, label_dir)

            chain = load_chain(chain_dir, label, BURN)
            if chain is None:
                continue
            if chain["samples"].shape[0] == 0:
                log.warning("  Chain '%s' has 0 samples after burn — skipping.", label)
                continue
            samples_for_model[label] = chain

        if not samples_for_model:
            log.warning("No chains loaded for %s — skipping.", model_name)
            continue

        # 3. Compute medians, D_bar, store samples for WAIC
        _, _, structured_values = calculate_asymmetric_from_samples(
            samples=samples_for_model,
            parameters=cfg["parameters"],
            observations=cfg["observations"],
        )
        all_best_fit[model_name] = structured_values

    if not all_best_fit:
        log.error("No chains were loaded at all. Check CHAIN_BASE and CONSTRAINT.")
        return

    # 4. Compute AIC, AICc, BIC, DIC, WAIC and deltas relative to reference_model
    log.info("Running statistical analysis...")
    results = statistical_analysis(
        best_fit_values=all_best_fit,
        data=data,
        CONFIG=CONFIG,
        reference_model=reference_model,
    )

    # 5. Write stats_summary.txt and print interpretation per model
    for model_name, obs_results in results.items():
        out_dir = os.path.join(STATS_BASE, CONSTRAINT, model_name)
        os.makedirs(out_dir, exist_ok=True)

        stats_list = []
        for obs_name, stats in obs_results.items():
            stats_list.append({"Observation": obs_name, **stats})

            interp = interpret_delta_IC(
                stats.get("dAIC",  float("nan")),
                stats.get("dBIC",  float("nan")),
                stats.get("dAICc", float("nan")),
                stats.get("dDIC",  float("nan")),
                stats.get("dWAIC", float("nan")),
                stats.get("sigma", 0.0),
            )
            print(f"\n[{model_name}] {obs_name}\n{interp}")

        save_stats_to_file(model_name, out_dir, stats_list)
        log.info("Saved: %s/stats_summary.txt", out_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
