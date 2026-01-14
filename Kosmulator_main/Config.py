#!/usr/bin/env python3
"""
Core configuration builder for Kosmulator.

Responsibilities:
  - Load all observational datasets (BAO, DESI, Pantheon+, BBN, CMB, etc.).
  - Canonicalise and de-duplicate observation groups.
  - Inject any required parameters into each model per observation group
    (H0, r_d, CMB nuisance, BBN boxes, fσ8 gamma policy, etc.).
  - Construct the CONFIG dict consumed by the MCMC layer:
      CONFIG[model_name] = {
          "observations", "observation_types",
          "parameters", "true_values", "prior_limits",
          "restrictions", "ndim",
          "rd_policy", "pantheonp_mode",
          "fs8_gamma_fixed_by_group",
          "nwalker", "nwalker_by_obs",
      }
  - Build a single shared `data` dict with loaded datasets.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Callable, Optional

import numpy as np
import pandas as pd


from Kosmulator_main import utils as U  # Pantheon+ covariance helpers
from Kosmulator_main.utils import (
    canonicalise_and_dedup_observations,
    _inject_planck_nuisance_defaults,
)

from Kosmulator_main import constants as K
from Kosmulator_main.constants import (
    PLANCK_NUISANCE_DEFAULTS,
    PLANCK_TT_ONLY_NUISANCE,
    PLANCK_TTTEEE_NUISANCE,
)

# Optional MPI (kept here in case we want to broadcast covariances in future)
try:
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover
    MPI = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Generic data loaders
# ---------------------------------------------------------------------------

def load_data(file_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Generic loader used for "simple" datasets.

    Behaviour:
      - If CSV/ECSV: return a dict of {col_name: np.ndarray}.
      - Else: treat as whitespace text; assume columns:
          col0 = redshift
          col1 = value / type_data
          col2 = error (optional → default 1.0)
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", path)
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in ('.csv', '.ecsv'):
        df = pd.read_csv(path, sep=None, engine='python')
        return {col: df[col].values for col in df.columns}

    # Fallback: plain whitespace text
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        'redshift': data[:, 0],
        'type_data': data[:, 1],
        'type_data_error': data[:, 2] if data.shape[1] > 2 else np.ones(data.shape[0]),
    }


# ---------------------------------------------------------------------------
# Pantheon+ helper
# ---------------------------------------------------------------------------

def prepare_pantheonP_data(data, z_min: float = 0.01, mode: str = "PplusSH0ES"):
    """
    Prepare Pantheon+ for either:
      • "Pplus"       : Pantheon+ without SH0ES anchor (exclude calibrators)
      • "PplusSH0ES"  : Pantheon+ with SH0ES anchor (include calibrators)

    Notes:
      - The "mode" string is treated case-insensitively and matched on "sh0es".
      - We return masks + sliced arrays plus a `cov_path` that downstream
        code uses to build/apply the covariance.
    """
    mode_norm = (mode or "PplusSH0ES").strip().lower()
    use_sh0es = ("sh0es" in mode_norm)

    z    = np.asarray(data["zHD"], dtype=float)
    mb   = np.asarray(data["m_b_corr"], dtype=float)
    trig = np.asarray(data["IS_CALIBRATOR"], dtype=int)
    cep  = np.asarray(data["CEPH_DIST"], dtype=float)

    if use_sh0es:
        # Include low-z SH0ES calibrators
        mask = (z > z_min) | (trig > 0)
        trig_eff = trig
    else:
        # Pure Hubble-flow subset: drop calibrators completely
        mask = (z > z_min) & (trig == 0)
        trig_eff = np.zeros_like(trig)

    idx = np.where(mask)[0]
    return {
        "mode": ("PplusSH0ES" if use_sh0es else "Pplus"),
        "mask": mask,
        "indices": idx,
        "zHD": z[mask],
        "m_b_corr": mb[mask],
        "IS_CALIBRATOR": trig_eff[mask],
        "CEPH_DIST": cep[mask],  # unused if all flags are 0, but kept for completeness
        "cov_path": data.get(
            "cov_path",
            os.path.join(K.OBSERVATIONS_BASE, "PantheonP.cov"),
        ),
    }


from pathlib import Path
from typing import Union, Dict
import numpy as np

def load_named_sne_with_zcmb(file_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Loader for SNe datasets with a header and named columns, e.g.:

      DESY5.dat  (CSV):
        CID,IDSURVEY,zCMB,zHD,zHEL,MU,MUERR_FINAL

      Union3.txt (whitespace):
        #name zcmb zhel dz mb dmb x1 dx1 color dcolor 3rdvar d3rdvar ...

    Behaviour:
      - Uses the CMB-frame redshift column (zCMB/zcmb) as `redshift`.
      - Uses MU/mb as `type_data` (distance-modulus–like quantity).
      - Uses MUERR_FINAL/dmb/etc. as `type_data_error` (σ_μ).
      - For Union3, parses lines manually to tolerate missing trailing columns.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"{file_path} is empty.")

    first = lines[0]

    # --------------------------------------------------
    # Branch 1: CSV-style (DESY5) — header has commas
    # --------------------------------------------------
    if "," in first:
        arr = np.genfromtxt(
            path,
            names=True,
            delimiter=",",
            dtype=None,
            encoding=None,
        )

        if arr.dtype.names is None:
            raise ValueError(f"{file_path} has no named columns; cannot parse as SNe.")

        name_map = {n.lower(): n for n in arr.dtype.names}

        # redshift
        z_name = None
        for cand in ("zcmb", "z_cmb", "zhd", "z"):
            if cand in name_map:
                z_name = name_map[cand]
                break
        if z_name is None:
            raise ValueError(
                f"Could not find a zCMB-like column in {file_path} "
                f"(columns={arr.dtype.names})"
            )

        # distance modulus / magnitude
        mu_name = None
        for cand in ("mu", "mb"):
            if cand in name_map:
                mu_name = name_map[cand]
                break
        if mu_name is None:
            raise ValueError(
                f"Could not find a MU/mb column in {file_path} "
                f"(columns={arr.dtype.names})"
            )

        # error on distance modulus
        err_name = None
        for cand in ("muerr_final", "dmb", "dmu", "sigma_mu"):
            if cand in name_map:
                err_name = name_map[cand]
                break

        z = np.asarray(arr[z_name], dtype=float)
        mu = np.asarray(arr[mu_name], dtype=float)

        if err_name is not None:
            sigma = np.asarray(arr[err_name], dtype=float)
        else:
            sigma = np.ones_like(z)

        if not np.any(sigma > 0):
            sigma = np.ones_like(z)

        return {
            "redshift": z,
            "type_data": mu,
            "type_data_error": sigma,
        }

    # --------------------------------------------------
    # Branch 2: whitespace + variable columns (Union3)
    # --------------------------------------------------
    header = first.lstrip("#").strip()
    if not header:
        raise ValueError(f"{file_path} header is empty or malformed.")

    names = header.split()
    name_to_idx = {n.lower(): i for i, n in enumerate(names)}

    def find_idx(*cands):
        for c in cands:
            if c.lower() in name_to_idx:
                return name_to_idx[c.lower()]
        return None

    idx_z   = find_idx("zcmb", "z_cmb", "z")
    idx_mu  = find_idx("mu", "mb")
    idx_err = find_idx("dmu", "dmb", "sigma_mu")

    if idx_z is None or idx_mu is None:
        raise ValueError(
            f"Missing zcmb/mb columns in {file_path} header: {names}"
        )

    z_vals: list[float] = []
    mu_vals: list[float] = []
    err_vals: list[float] = []

    for line in lines[1:]:
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue

        parts = line.split()
        # Need indices up to whichever of z, mu, err is highest
        max_idx = max(idx_z, idx_mu, idx_err or 0)
        if len(parts) <= max_idx:
            # Line shorter than expected → skip
            continue

        try:
            z_vals.append(float(parts[idx_z]))
            mu_vals.append(float(parts[idx_mu]))

            if idx_err is not None and len(parts) > idx_err:
                err_vals.append(float(parts[idx_err]))
            else:
                err_vals.append(1.0)
        except ValueError:
            # Non-numeric junk line → skip
            continue

    z = np.asarray(z_vals, dtype=float)
    mu = np.asarray(mu_vals, dtype=float)
    sigma = np.asarray(err_vals, dtype=float)

    if not np.any(sigma > 0):
        sigma = np.ones_like(z)

    return {
        "redshift": z,
        "type_data": mu,
        "type_data_error": sigma,
    }


# ---------------------------------------------------------------------------
# DESI BAO VI loaders
# ---------------------------------------------------------------------------

def load_DESI_data(file_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Parse DESI BAO VI files.

    Assumptions:
      - We search for a marker line such as:
          "#These are the chosen ones to compute:"
        (we accept a few common variants to be robust to typos).
      - After the marker, each non-comment line is:
          col0 = (ignored)
          col1 = z
          col2 = measurement value
          col3 = error
          col4 = type code (int)
    """
    text = Path(file_path).read_text().splitlines()
    candidates = [
        "#Theese are the chosen one to compute:",
        "#These are the chosen ones to compute:",
        "#These are the chosen one(s) to compute:",
    ]
    start = None
    for m in candidates:
        for i, line in enumerate(text):
            if m.lower() in line.lower():
                start = i + 1
                break
        if start is not None:
            break
    if start is None:
        raise ValueError(
            f"DESI VI marker not found in {file_path}. "
            "Refusing to parse to avoid reading headers."
        )

    redshifts, values, errors, types = [], [], [], []
    for line in text[start:]:
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        redshifts.append(float(parts[1]))
        values.append(float(parts[2]))
        errors.append(float(parts[3]))
        types.append(int(parts[4]))

    return {
        'redshift': np.array(redshifts, float),
        'measurement': np.array(values, float),
        'measurement_error': np.array(errors, float),
        'type': np.array(types, int),
    }


def load_DESI_cov(file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a DESI VI covariance *or* inverse-covariance matrix and return (cov, inv_cov).

    Heuristic:
      - DESI covariance diagonals are O(1e-4 ... 1e-1),
        inverse-cov diagonals are O(10 ... 1e4).
      - We treat median(diag) > 1 as "looks like inverse-covariance".
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("DESI covariance file not found: %s", path)
        raise FileNotFoundError(f"File not found: {path}")

    arr = np.loadtxt(path)

    # Handle flattened forms (optionally with a leading N)
    if arr.ndim == 1:
        v = arr
        if len(v) > 1 and int(round(v[0])) ** 2 == (len(v) - 1):
            n = int(round(v[0]))
            mat = v[1:].reshape(n, n)
        else:
            n = int(round(np.sqrt(len(v))))
            if n * n != len(v):
                raise ValueError(f"Cannot reshape DESI cov from len={len(v)}")
            mat = v.reshape(n, n)
    else:
        mat = arr

    # Symmetrise (defensive)
    mat = 0.5 * (mat + mat.T)

    # Quick PD-friendly inverse
    try:
        inv_mat = np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        inv_mat = np.linalg.pinv(mat, rcond=1e-12)

    # Auto-detect whether 'mat' is actually an inverse-covariance
    diag_med = float(np.median(np.diag(mat)))
    looks_like_inv = diag_med > 1.0

    if looks_like_inv:
        # File is inv-covariance: return (cov, inv_cov) = (inv(mat), mat)
        cov, inv_cov = inv_mat, mat
    else:
        # File is covariance as-is
        cov, inv_cov = mat, inv_mat

    # Final symmetrise (for numerical hygiene)
    cov     = 0.5 * (cov + cov.T)
    inv_cov = 0.5 * (inv_cov + inv_cov.T)

    # Light sanity guard: prefer PD-ish cov; fall back to pseudo-inverse if needed
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        logger.warning("DESI cov not PD under Cholesky; using pseudo-inverse fallback.")
        inv_cov = np.linalg.pinv(cov, rcond=1e-12)
        cov     = 0.5 * (cov + cov.T)

    return cov, inv_cov


# ---------------------------------------------------------------------------
# Load all observation data for a given CONFIG
# ---------------------------------------------------------------------------

def load_all_data(config, prior_limits=None, logger=None) -> Dict[str, Any]:
    """
    Load all datasets required by the observation list in `config`.

    Inputs
    ------
    config : dict-like
        Must contain:
          - config["observations"]: List[List[str]]
          - config["prior_limits"]: used for BBN grid building (if given).
    prior_limits : dict, optional
        External prior_limits dict. If None, fall back to config["prior_limits"].

    Returns
    -------
    observation_data : dict
        A dict keyed by observation tag (e.g. "BAO", "PantheonP") or
        special keys for CMB lensing paths, BBN backend config, etc.
    """
    observation_data: Dict[str, Any] = {}

    # Allow callers to omit prior_limits; we fall back to CONFIG content.
    if prior_limits is None:
        prior_limits = config.get("prior_limits", {})

    flat_obs = {o for grp in config["observations"] for o in grp}
    for grp in config["observations"]:
        # Safety: mixed CMB TTTEEE + TT only in one group is not supported
        if "CMB_hil" in grp and "CMB_hil_TT" in grp:
            raise ValueError(
                "Invalid group: do not include both CMB_hil (TTTEEE) and "
                "CMB_hil_TT (TT-only) in the same observation group."
            )

    # Decide lensing file once based on global combo policy:
    # We provide both raw and CMB-marged; choice is made in the likelihood code.
    _lensing_file_nonmarg = os.path.join(
        K.OBSERVATIONS_BASE,
        "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing",
    )
    _lensing_file_cmbmarg = os.path.join(
        K.OBSERVATIONS_BASE,
        "smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_CMBmarged.clik_lensing",
    )

    observation_data["CMB_lensing_RAW"]       = _lensing_file_nonmarg
    observation_data["CMB_lensing_CMBMARGED"] = _lensing_file_cmbmarg

    # Per-observation loader
    for obs_list in config["observations"]:
        for obs in obs_list:
            file_path = os.path.join(K.OBSERVATIONS_BASE, f"{obs}.dat")

            # ------------------
            # Pantheon+
            # ------------------
            if obs in ("PantheonP", "PantheonPS"):
                # Always read the canonical Pantheon+ file
                pantheon_file = os.path.join(K.OBSERVATIONS_BASE, "PantheonP.dat")
                df = pd.read_csv(pantheon_file, sep=r'\s+')
                pantheon_data = {
                    "zHD": df["zHD"].values,
                    "m_b_corr": df["m_b_corr"].values,
                    "IS_CALIBRATOR": df["IS_CALIBRATOR"].values,
                    "CEPH_DIST": df["CEPH_DIST"].values,
                    "cov_path": os.path.join(K.OBSERVATIONS_BASE, "PantheonP.cov"),
                }

                # Observation name controls the mode:
                #   PantheonP  → Hubble-flow only (no SH0ES)
                #   PantheonPS → Hubble-flow + SH0ES calibrators
                if obs == "PantheonP":
                    mode = "Pplus"
                else:
                    mode = "PplusSH0ES"

                observation_data[obs] = prepare_pantheonP_data(
                    pantheon_data,
                    mode=mode,
                )

                # Pre-build covariance + lower Cholesky and attach it
                try:
                    L = U.compute_pantheon_cov(
                        observation_data,       # needs indices/mask for this tag
                        config,                 # for presence checks
                        comm=None,              # no MPI broadcast here
                        rank=0,
                        cov_file=observation_data[obs]['cov_path'],
                        obs_tag=obs,
                    )
                    if L is not None:
                        # Store Cholesky L and per-SN σ = sqrt(sum_j L_ij^2)
                        observation_data[obs]["cov"] = L
                        observation_data[obs]["type_data_error"] = np.sqrt(
                            np.sum(L**2, axis=1)
                        )
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        "%s cov precompute failed: %s", obs, e
                    )

            # ------------------
            # BAO (13-element vector with fixed order)
            # ------------------
            elif obs == "BAO":
                observation_data[obs] = {
                    "covd1": np.loadtxt(file_path),
                    # 12-slot observed vector in the SAME order as the chi^2 builder
                    "obs_vec": np.array([
                        7.92512927,
                        13.6200308, 20.98334647,
                        16.84645313, 20.07872919,
                        21.70841761, 17.87612922,
                        27.78720817, 13.82372285,
                        26.07217182,
                        39.70838281, 8.52256583,
                    ], dtype=float),
                    # Meta to keep plotting/chi² in sync
                    "z_slots": [
                        0.295, 0.510, 0.510, 0.706, 0.706, 0.930,
                        0.930, 1.317, 1.317, 1.491, 2.330, 2.330
                    ],
                    "code_slots": [
                        3, 8, 6, 8, 6, 8,
                        6, 8, 6, 3, 8, 6  # 3=DV/rd, 8=DM/rd, 6=DH/rd
                    ],
                }

            # ------------------
            # Planck CMB paths
            # ------------------
            elif obs == 'CMB_hil':
                observation_data[obs] = os.path.join(
                    K.OBSERVATIONS_BASE, "plik_rd12_HM_v22b_TTTEEE.clik"
                )

            elif obs == 'CMB_hil_TT':
                observation_data[obs] = os.path.join(
                    K.OBSERVATIONS_BASE, "plik_rd12_HM_v22_TT.clik"
                )

            elif obs == 'CMB_lowl':
                observation_data[obs] = os.path.join(
                    K.OBSERVATIONS_BASE,
                    "simall_100x143_offlike5_EE_Aplanck_B.clik",
                )

            elif obs == 'CMB_lensing':
                # We store RAW here; likelihood will decide whether to swap to CMBMARGED
                observation_data[obs] = _lensing_file_nonmarg

            # ------------------
            # DESI DR1 / DR2
            # ------------------
            elif obs == "DESI_DR1":
                # If your current files are named Isma_desi_*.txt, keep them here;
                # otherwise rename them to DESI_DR1_* accordingly.
                vi_path  = os.path.join(K.OBSERVATIONS_BASE, "Isma_desi_VI.txt")
                cov_path = os.path.join(K.OBSERVATIONS_BASE, "Isma_desi_covtot_VI.txt")

                desi = load_DESI_data(vi_path)
                cov, inv_cov = load_DESI_cov(cov_path)
                desi["cov"] = cov
                desi["inv_cov"] = inv_cov

                n = len(desi["redshift"])
                if cov.shape != (n, n):
                    logger.error(
                        "DESI_DR1 covariance shape %s != (%d,%d)",
                        cov.shape, n, n,
                    )

                observation_data[obs] = desi

            elif obs == "DESI_DR2":
                # DR2 uses the same VI structure
                vi_path  = os.path.join(K.OBSERVATIONS_BASE, "DESI_DR2_synced.txt")
                cov_path = os.path.join(K.OBSERVATIONS_BASE, "DESI_DR2_covtot.txt")

                desi = load_DESI_data(vi_path)
                cov, inv_cov = load_DESI_cov(cov_path)
                desi["cov"] = cov
                desi["inv_cov"] = inv_cov

                n = len(desi["redshift"])
                if cov.shape != (n, n):
                    logger.error(
                        "DESI_DR2 covariance shape %s != (%d,%d)",
                        cov.shape, n, n,
                    )

                observation_data[obs] = desi

            # ------------------
            # BBN (approx-only)
            # ------------------
            elif obs == "BBN_DH":
                observation_data[obs] = {
                    # "mean": uses weighted mean DH;
                    # "per_system": uses full list of individual systems (default).
                    "mode": "per_system",
                    "units": "scaled1e6",  # PDG numbers as printed (1e6 * D/H)
                    "S": 1.137,            # scale factor for error inflation
                    "weighted_mean": {"DH": 25.47, "sigma": 0.29},
                    "bbn_model": "approx",  # simple BBN backend
                    "systems": [
                        {"name": "SDSS J1419+0829", "DH": 25.06, "sig_up": 0.52, "sig_dn": 0.52},
                        {"name": "HS 0105+1619",    "DH": 25.76, "sig_up": 1.54, "sig_dn": 1.54},
                        {"name": "QSO B0913+0715",  "DH": 25.29, "sig_up": 1.05, "sig_dn": 1.05},
                        {"name": "SDSS J1358+0349", "DH": 26.18, "sig_up": 0.72, "sig_dn": 0.72},
                        {"name": "SDSS J1358+6522", "DH": 25.82, "sig_up": 0.71, "sig_dn": 0.71},
                        {"name": "SDSS J1558-0031", "DH": 24.04, "sig_up": 1.44, "sig_dn": 1.44},
                        {"name": "PKS 1937-1009 A", "DH": 24.49, "sig_up": 2.80, "sig_dn": 2.80},
                        {"name": "QSO J1444+2919",  "DH": 19.68, "sig_up": 3.3,  "sig_dn": 2.8},
                        {"name": "PKS 1937-1009 B", "DH": 26.24, "sig_up": 0.48, "sig_dn": 0.48},
                        {"name": "QSO 1009+2956",   "DH": 24.77, "sig_up": 4.1,  "sig_dn": 3.5},
                        {"name": "QSO 1243+307",    "DH": 23.88, "sig_up": 0.82, "sig_dn": 0.82},
                    ],
                    "bbn_model_effective": "approx",
                }

            # ------------------
            # BBN + AlterBBN grid
            # ------------------
            elif obs == "BBN_DH_AlterBBN":
                # Use all individual systems by default (change to "mean" for PDG value)
                observation_data[obs] = {
                    "mode": "per_system",  # or "mean"
                    "units": "scaled1e6",
                    "S": 1.137,
                    "weighted_mean": {"DH": 25.47, "sigma": 0.29},
                    "bbn_model": "alterbbn",  # request high-precision BBN backend

                    # Precomputed grid path; comment this out to run AlterBBN live.
                    # Running AlterBBN live can be ~10x slower for MCMC.
                    "bbn_grid_path": os.path.join(K.OBSERVATIONS_BASE, K.BBN_GRID_RELATIVE),

                    # Optional external binary (subprocess) fallback; unused by default.
                    "alterbbn_bin": None,

                    # Default τ_n if not varied (priors can open this axis).
                    "tau_n": K.TAU_N_DEFAULT,

                    # Grid resolution; explicit step sizes recommended so coverage
                    # tracks priors. If explicit nodes are supplied, they override
                    # these step sizes.
                    "obh2_step": 1e-5,
                    "Neff_step": 0.02,
                    "tau_step":  1.0,

                    # Optional explicit grids:
                    # "obh2_grid": [...],
                    # "Neff_grid": [...],
                    # "tau_grid":  [...],

                    "alterbbn_extra": {},
                }

                # Build prior boxes for grid-building from priors (with safe defaults)
                pri_src = prior_limits or {}
                priors = {
                    "Omega_bh^2": pri_src.get("Omega_bh^2", (0.019, 0.026)),
                    "N_eff":      pri_src.get("N_eff",      (K.N_EFF_DEFAULT, K.N_EFF_DEFAULT)),
                    "tau_n":      pri_src.get(
                        "tau_n",
                        (observation_data[obs]["tau_n"],
                         observation_data[obs]["tau_n"]),
                    ),
                }

                from Statistical_packages import ensure_bbn_backend
                observation_data[obs] = ensure_bbn_backend(
                    observation_data[obs],
                    logger=logger,
                    priors=priors,
                )

                # Nice logging for visibility
                cfg = observation_data['BBN_DH_AlterBBN']
                logger.info("BBN resolved backend: %s", cfg.get("bbn_model_effective"))
                if cfg.get("bbn_model_effective") == "alterbbn_grid":
                    g = cfg["bbn_grid"]
                    logger.info(
                        "Grid: obh2[%d]=[%.6f..%.6f], neff[%d]=%s, tau[%d]=%s, "
                        "logspace=%s, DH.shape=%s",
                        len(g["obh2"]), g["obh2"][0], g["obh2"][-1],
                        len(g.get("neff", [])), "present" if "neff" in g else "none",
                        len(g.get("tau",  [])), "present" if "tau"  in g else "none",
                        g.get("logspace", False), np.shape(g["DH"]),
                    )

            # ------------------
            # BBN prior on Ω_b h^2 from PRyMordial
            # ------------------
            elif obs == "BBN_PryMordial":
                # DESI DR2 "BBN prior": Gaussian on Ω_b h^2 from PRyMordial,
                # with conservative nuclear-rate marginalization.
                #   Ω_b h^2 = 0.02218 ± 0.00055  (ΛCDM)
                observation_data[obs] = {
                    "mu_obh2":    0.02218,
                    "sigma_obh2": 0.00055,
                    # For a 2D (Ω_b h^2, N_eff) prior, you can add:
                    # "mu_Neff": ...,
                    # "cov": [[sigma_obh2**2, rho*sigma_obh2*sigma_neff],
                    #         [rho*sigma_obh2*sigma_neff, sigma_neff**2]],
                }

            # ------------------
            # DESY5 / Union3 SNe Hubble diagrams
            # ------------------
            elif obs in ("DESY5", "Union3"):
                if obs == "DESY5":
                    sne_path = os.path.join(K.OBSERVATIONS_BASE, "DESY5.dat")
                else:  # "Union3"
                    sne_path = os.path.join(K.OBSERVATIONS_BASE, "Union3.txt")

                observation_data[obs] = load_named_sne_with_zcmb(sne_path)
            # ------------------
            # Default loader
            # ------------------
            else:
                observation_data[obs] = load_data(file_path)

    return observation_data


# ---------------------------------------------------------------------------
# CONFIG builder
# ---------------------------------------------------------------------------

def create_config(
    models: Dict[str, Any],
    true_values: Union[Dict[str, float], Any] = None,
    prior_limits: Dict[str, Tuple[float, float]] = None,
    restrictions: Dict[str, Dict[str, Callable[[float], bool]]] = None,
    observation: List[List[str]] = None,
    nwalkers: int = 20,
    nsteps: int = 200,
    burn: int = 20,
    model_name: List[str] = None,
    pantheonp_mode: str = "PplusSH0ES",  # DEPRECATED: use PantheonP/PantheonPS tags instead
    logger=None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Main entry point: construct per-model CONFIG and a shared `data` dict.

    Parameters
    ----------
    models : dict
        Model registry, e.g. {"LCDM_v": {"parameters": [...], ...}, ...}.
        `parameters` here is the *core* model parameter list, not yet expanded
        per observation group.
    true_values : dict
        Global "true" parameter guesses (used as initial means).
    prior_limits : dict
        Global prior boxes: {param: (low, high)}. Each parameter used in any
        observation group must appear here.
    restrictions : dict
        Optional per-model sanity functions, e.g.
          restrictions["LCDM_v"]["Omega_m"] = lambda x: 0.0 < x < 1.0
    observation : list of list of str
        Observation groups: [["CC","OHD"], ["PantheonP"], ["BAO","CMB_hil_TT"], ...].
    nwalkers, nsteps, burn : int
        Sampler steering parameters (base values; we may bump nwalker for safety).
    model_name : list of str
        List of keys in `models` to use for this run.
    pantheonp_mode : str
        One of {"Pplus", "PplusSH0ES"}; controls Pantheon+ anchor strategy.

    Returns
    -------
    config : dict
        Per-model configuration dict (see module docstring).
    data : dict
        Loaded observational datasets.
    """
    # Defaults
    observation  = observation  or [['PantheonP']]
    true_values  = true_values  or {}
    prior_limits = prior_limits or {}
    restrictions = restrictions or {}

    # Normalise observation groups (order, duplicates, logging)
    observation = canonicalise_and_dedup_observations(observation, logger)

    if model_name is None:
        raise ValueError("`model_name` must be provided as a list of model keys.")

    # Map observation tag → likelihood/type label
    obs_type_map = {
        # --- SNe-like (distance modulus) ---
        'JLA':       'SNe',
        'Pantheon':  'SNe',
        'PantheonP': 'SNe',
        'PantheonPS': 'SNe',
        'DESY5':     'SNe',
        'Union3':    'SNe',

        # --- Background expansion ---
        'OHD': 'OHD',
        'CC':  'CC',

        # --- BAO / distance ratios ---
        'BAO':       'BAO',
        'DESI_DR1':  'DESI_DR1',
        'DESI_DR2':  'DESI_DR2',

        # --- Growth / RSD ---
        'f_sigma_8': 'f_sigma_8',
        'f':         'f',

        # --- BBN ---
        'BBN_DH':          'BBN_DH',
        'BBN_DH_AlterBBN': 'BBN_DH',
        'BBN_PryMordial' : 'BBN_prior',

        # --- CMB ---
        'CMB_hil':    'CMB',
        'CMB_hil_TT': 'CMB',
        'CMB_lowl':   'CMB',
        'CMB_lensing':'CMB',
    }

    # ----------------------------------------------------------------------
    # 1. If any CMB dataset is requested, ensure nuisance priors exist
    #    globally (we'll later prune them per group based on clik).
    # ----------------------------------------------------------------------
    flat_obs = {o for grp in observation for o in grp}
    if {"CMB_hil", "CMB_hil_TT", "CMB_lowl", "CMB_lensing"} & flat_obs:
        try:
            # Lazy import to avoid loading clik when not needed
            from Kosmulator_main.Class_run import (
                get_clik_hil,
                get_clik_hilTT,
                get_clik_lowl,
                get_clik_lensing,
                quiet_cstdio,
            )
            names: set[str] = set()
            if "CMB_hil" in flat_obs:
                with quiet_cstdio():
                    names |= set(get_clik_hil().get_extra_parameter_names())
            if "CMB_hil_TT" in flat_obs:
                with quiet_cstdio():
                    names |= set(get_clik_hilTT().get_extra_parameter_names())
            if "CMB_lowl" in flat_obs:
                with quiet_cstdio():
                    names |= set(get_clik_lowl().get_extra_parameter_names())
            if "CMB_lensing" in flat_obs:
                with quiet_cstdio():
                    names |= set(get_clik_lensing().get_extra_parameter_names())
        except Exception:
            # If clik isn't available, fall back to our default nuisance table
            names = set()

        # Minimum safe set: if clik didn’t return anything, use the default
        defaults = set(PLANCK_NUISANCE_DEFAULTS.keys())
        if not names:
            names = defaults
        else:
            # Always allow A_planck as a special case
            if "A_planck" in defaults:
                names.add("A_planck")

        _inject_planck_nuisance_defaults(true_values, prior_limits, sorted(names))

    # ----------------------------------------------------------------------
    # 2. Expand each model's core parameter list with obs-required params
    # ----------------------------------------------------------------------
    models_aug = Add_required_parameters(models, observation)

    config: Dict[str, Any] = {}

    for mod in model_name:
        mod_data   = models_aug[mod]
        base_param = mod_data["parameters"]  # list-of-lists per obs-group
        obs_types  = [[obs_type_map[o] for o in grp] for grp in observation]

        # ------------------------------------------------------------------
        # 2a. Build parameter sets, true_values, prior_limits per group
        # ------------------------------------------------------------------
        param_sets: List[List[str]] = []
        tv_sets:     List[np.ndarray] = []
        pl_sets:     List[Dict[str, Tuple[float, float]]] = []
        ndim_sets:   List[int] = []

        for pset in base_param:
            # Flatten nested parameter containers (sets/tuples) while keeping
            # ordering and uniqueness.
            flat_pset: List[str] = []
            for item in pset:
                if isinstance(item, (list, tuple, set)):
                    for s in item:
                        if s not in flat_pset:
                            flat_pset.append(s)
                else:
                    if item not in flat_pset:
                        flat_pset.append(item)

            tv_vec: List[float] = []
            pl_map: Dict[str, Tuple[float, float]] = {}
            for p in flat_pset:
                if p not in prior_limits:
                    raise ValueError(
                        f"Missing prior for '{p}'. Please add "
                        f"prior_limits['{p}'] = (low, high)."
                    )
                lo, hi = prior_limits[p]
                pl_map[p] = (lo, hi)
                tv_vec.append(true_values.get(p, 0.5 * (lo + hi)))

            param_sets.append(flat_pset)
            tv_sets.append(np.asarray(tv_vec, dtype=float))
            pl_sets.append(pl_map)
            ndim_sets.append(len(flat_pset))

        # Initial config snapshot (we will update parameters after policies)
        config[mod] = {
            "model_name":        mod,
            "observations":      observation,
            "observation_types": obs_types,
            "parameters":        param_sets,
            "true_values":       tv_sets,
            "prior_limits":      pl_sets,
            "restrictions":      restrictions.get(mod, {}),
            "ndim":              ndim_sets,
            "nsteps":            nsteps,
            "burn":              burn,
            "nwalker":           int(nwalkers),    # temporary; recomputed below
            "nwalker_by_obs":    [int(nwalkers)] * len(ndim_sets),
        }
        # Debug flags (set by utils.parse_cli_args -> constants K)
        config[mod]["debug"] = {
            "print_loglike": bool(getattr(K, "print_loglike", False)),
            "print_loglike_every": int(getattr(K, "print_loglike_every", 1) or 1),
        }

        # ------------------------------------------------------------------
        # 3. r_d policy & fσ8 singleton gamma policy
        # ------------------------------------------------------------------
        tv_global = true_values or {}

        rd_fixed = float(tv_global.get("r_d_fixed", K.R_D_SINGLETON))     # <- fixed singleton value
        rd_mu    = float(tv_global.get("r_d", rd_fixed))                  # <- free/gaussian mean (defaults to fixed)

        config[mod]["rd_policy"] = {
            "singleton_mode": "gaussian",
            "fixed_value": rd_fixed,
            "gaussian": {"mu": rd_mu, "sigma": 2.0},
            "cmb_calibrates_rd": True,
        }

        def _has_early_calibrator(grp: list[str]) -> bool:
            early = {
                "BBN_DH", "BBN_DH_AlterBBN", "BBN_PryMordial",
                "CMB_hil", "CMB_hil_TT", "CMB_lowl",
            }  # note: CMB_lensing excluded by design
            return any(x in early for x in grp)

        def _has_bao_desi(grp: list[str]) -> bool:
            return any(x in {"BAO", "DESI_DR1", "DESI_DR2"} for x in grp)

        def _is_bao_desi_singleton(grp: list[str]) -> bool:
            return len(grp) == 1 and grp[0] in {"BAO", "DESI_DR1", "DESI_DR2"}

        param_sets_policy = []
        fs8_gamma_fixed_by_group = {}

        for gi, obs_grp in enumerate(config[mod]["observations"]):
            grp_params = list(config[mod]["parameters"][gi])

            # r_d handling:
            #  - If early-time calibrators present: r_d is calibrated → do not sample it.
            #  - If singleton BAO/DESI: fix r_d (rd_policy["fixed_value"]).
            #  - If BAO/DESI + something: treat r_d as a free parameter.
            if _has_early_calibrator(obs_grp):
                if "r_d" in grp_params:
                    grp_params.remove("r_d")
                if logger:
                    early = {
                        "BBN_DH", "BBN_DH_AlterBBN", "BBN_PryMordial",
                        "CMB_hil", "CMB_hil_TT", "CMB_lowl",
                    }
                    calibs = [x for x in obs_grp if x in early]
                    logger.warning(
                        "r_d calibrated by early-time dataset(s) %s for %s (model %s)",
                        calibs, obs_grp, mod,
                    )

            elif _is_bao_desi_singleton(obs_grp):
                if "r_d" in grp_params:
                    grp_params.remove("r_d")
                if logger:
                    rd_fix = float(config[mod]["rd_policy"].get("fixed_value", K.R_D_SINGLETON))
                    logger.warning(
                        "BAO/DESI singleton: fixed r_d=%.1f Mpc and removed 'r_d' from parameters "
                        "for %s (model %s)",
                        rd_fix, obs_grp, mod,
                    )

            elif _has_bao_desi(obs_grp):
                if "r_d" not in grp_params:
                    grp_params.append("r_d")
                if logger:
                    logger.warning(
                        "BAO/DESI combo: r_d FREE for %s (model %s)", obs_grp, mod
                    )

            else:
                if "r_d" in grp_params:
                    grp_params.remove("r_d")

            # fσ8 singleton: fix gamma to GR-like value to avoid degeneracy
            if len(obs_grp) == 1 and obs_grp[0] == "f_sigma_8":
                if "gamma" in grp_params:
                    grp_params.remove("gamma")
                gamma_fix = float(true_values.get("gamma_fixed", K.GAMMA_FS8_SINGLETON))
                fs8_gamma_fixed_by_group[gi] = gamma_fix
                if logger:
                    logger.warning(
                        "fσ8 singleton: FIX gamma=%.3f for %s (model %s)",
                        gamma_fix, obs_grp, mod,
                    )

            param_sets_policy.append(grp_params)

        config[mod]["parameters"] = param_sets_policy
        config[mod]["fs8_gamma_fixed_by_group"] = fs8_gamma_fixed_by_group
        # After our policy choices, singleton_mode is effectively "fixed"
        config[mod]["rd_policy"]["singleton_mode"] = "fixed"
        config[mod]["prior_limits_global"] = dict(prior_limits)

        # ------------------------------------------------------------------
        # 4. PRUNE Planck nuisance to exactly what each clik requires
        # ------------------------------------------------------------------
        def _cmb_extra_names_for(obs: str) -> set[str]:
            """
            Ask clik what extra parameter names it exposes for a given CMB
            likelihood. This is used to intersect with our Planck nuisance
            allowlists to avoid sampling unused parameters.
            """
            def _decode_set(names):
                out = set()
                for n in names:
                    out.add(n.decode() if isinstance(n, (bytes, bytearray)) else str(n))
                return out
            try:
                from Kosmulator_main.Class_run import (
                    get_clik_hil,
                    get_clik_hilTT,
                    get_clik_lowl,
                    get_clik_lensing,
                )
                
                if obs == "CMB_hil":
                    return _decode_set(get_clik_hil().get_extra_parameter_names())
                if obs == "CMB_hil_TT":
                    return _decode_set(get_clik_hilTT().get_extra_parameter_names())
                if obs == "CMB_lowl":
                    return _decode_set(get_clik_lowl().get_extra_parameter_names())
                if obs == "CMB_lensing":
                    return _decode_set(get_clik_lensing().get_extra_parameter_names())
            except Exception:
                return set()
            return set()

        def _nuisance_allowlist_for_group(obs_grp: list[str]) -> set[str]:
            """
            For a given obs group, build the allowed set of Planck nuisance
            parameters, based on:
              - High-level policy (TT-only vs TTTEEE vs lowl vs lensing-only).
              - What clik *actually* exposes for that combination.
            """
            # Choose base allowlist by combo (policy-driven)
            if ("CMB_hil_TT" in obs_grp) and ("CMB_hil" not in obs_grp):
                base = set(PLANCK_TT_ONLY_NUISANCE)
                base.add("A_planck")  # ✅ ensure A_planck is never pruned for TT-only combos
            elif ("CMB_hil" in obs_grp):
                base = PLANCK_TTTEEE_NUISANCE   # TTTEEE: keep EE/TE & P-cal terms
            elif ("CMB_lowl" in obs_grp):
                base = _cmb_extra_names_for("CMB_lowl")  # usually empty/minimal
            elif ("CMB_lensing" in obs_grp):
                base = set()  # lensing 4pt exposes no sampler nuisances
            else:
                base = set()

            # Intersect with what clik reports for the group
            clik_reported = set()
            for o in obs_grp:
                clik_reported |= _cmb_extra_names_for(o)

            return (set(base) & clik_reported) if clik_reported else set(base)

        # Make a superset for safe membership tests
        ALL_PLANCK_NUISANCE = (
            set(PLANCK_NUISANCE_DEFAULTS.keys()) |
            PLANCK_TT_ONLY_NUISANCE |
            PLANCK_TTTEEE_NUISANCE
        )

        # Per-group pruning: remove any Planck nuisance not in the allowlist
        for gi, obs_grp in enumerate(config[mod]["observations"]):
            allow = _nuisance_allowlist_for_group(obs_grp)
            if not allow:
                continue
            keep = []
            for p in param_sets_policy[gi]:
                # Drop only if it's a Planck nuisance AND not allowed
                if (p in ALL_PLANCK_NUISANCE) and (p not in allow):
                    continue
                keep.append(p)
            param_sets_policy[gi] = keep

        # ------------------------------------------------------------------
        # 5. Rebuild tv/prior/ndim after pruning
        # ------------------------------------------------------------------
        tv_sets2, pl_sets2, ndim_sets2 = [], [], []
        for pset in config[mod]["parameters"]:
            tv_vec, pl_map = [], {}
            for p in pset:
                if p not in prior_limits:
                    raise ValueError(
                        f"Missing prior for '{p}' after rd/CMB policy. "
                        f"Add prior_limits['{p}'] = (low, high)."
                    )
                lo, hi = prior_limits[p]
                pl_map[p] = (lo, hi)
                tv_vec.append(true_values.get(p, 0.5 * (lo + hi)))
            tv_sets2.append(np.asarray(tv_vec, float))
            pl_sets2.append(pl_map)
            ndim_sets2.append(len(pset))

        config[mod]["true_values"]   = tv_sets2
        config[mod]["prior_limits"]  = pl_sets2
        config[mod]["ndim"]          = ndim_sets2

        # ------------------------------------------------------------------
        # 6. Compute safe walker counts AFTER all pruning
        # ------------------------------------------------------------------
        # Emcee/Zeus rule-of-thumb: nwalker >= 2 * ndim + 2
        min_per_group = [2 * d + 2 for d in ndim_sets2]
        safe_nwalkers = max(int(nwalkers), max(min_per_group))
        if safe_nwalkers > int(nwalkers) and logger:
            logger.warning(
                "[Config] Bumping nwalker from %d to %d for model '%s' "
                "(rule: 2*ndim+2; max ndim=%d; per-group mins=%s).",
                int(nwalkers), safe_nwalkers, mod, max(ndim_sets2), min_per_group,
            )
        config[mod]["nwalker"]        = safe_nwalkers
        config[mod]["nwalker_by_obs"] = [
            max(int(nwalkers), m) for m in min_per_group
        ]

        # ------------------------------------------------------------------
        # 7. Friendly warnings for weak singletons
        # ------------------------------------------------------------------
        for group in config[mod]["observations"]:
            if len(group) == 1 and group[0] in {
                "f", "BBN_DH", "BBN_DH_AlterBBN", "BBN_PryMordial", "BBN_prior"
            }:
                if logger:
                    logger.warning(
                        "Observation %s alone is not ideal for cosmology; "
                        "consider combining it with complementary data.",
                        group[0],
                    )

    # ----------------------------------------------------------------------
    # 8. Load data once (based on the observation list)
    # ----------------------------------------------------------------------
    data = load_all_data(
        {
            'observations': observation,
            'prior_limits': prior_limits,
        },
        prior_limits=prior_limits,
        logger=logger,
    )
    return config, data


# ---------------------------------------------------------------------------
# Parameter injection helper
# ---------------------------------------------------------------------------

def Add_required_parameters(
    models: Dict[str, Any],
    observations: List[List[str]],
) -> Dict[str, Any]:
    """
    Expand each model's core parameter list with those required by its
    observation groups.

    Policy summary:
      - Dataset-specific requirements are applied per group:
           BAO/DESI → H_0, r_d
           Pantheon/PantheonP → H_0 (+ M_abs for PantheonP)
           fσ8/f → gamma, sigma_8 as appropriate
           BBN → Omega_bh^2, and r_d is *not* sampled in BBN groups.
      - CMB-driven reparam is *per group*:
           If a group contains CMB data → we drop Omega_m and enforce
           {H_0, Omega_bh^2, Omega_dh^2} in that group only.
        Non-CMB groups keep the model's native parametrisation (e.g. Omega_m).
    """
    params_map = {
        'BAO': ['H_0', 'r_d'],
        'DESI_DR1': ['H_0', 'r_d'],
        'DESI_DR2': ['H_0', 'r_d'],
        'Pantheon': ['H_0'],
        'PantheonP': ['H_0', 'M_abs'],
        'PantheonPS':['H_0', 'M_abs'],
        'DESY5': ['H_0'],
        'Union3': ['H_0'],
        'f_sigma_8': ['Omega_m','sigma_8', 'gamma'],
        'f': ['Omega_m','gamma'],
        'JLA': ['H_0'],
        'OHD': ['H_0'],
        'CC': ['H_0'],
        'BBN_DH': ['Omega_bh^2'],
        'BBN_DH_AlterBBN': ['Omega_bh^2'],
        'BBN_PryMordial': ['Omega_bh^2'],

        # CMB core + nuisance to *sample* (we prune nuisances later)
        "CMB_hil": [
            "H_0", "Omega_bh^2", "Omega_dh^2",
            "ln10^10_As", "n_s", "tau_reio",
            *PLANCK_NUISANCE_DEFAULTS.keys(),
        ],
        "CMB_hil_TT": [
            "H_0", "Omega_bh^2", "Omega_dh^2",
            "ln10^10_As", "n_s", "tau_reio",
            "calib_100T", "calib_217T",
            *PLANCK_NUISANCE_DEFAULTS.keys(),  # pruned later by clik
        ],
        "CMB_lowl": [
            "H_0", "Omega_bh^2", "Omega_dh^2",
            "ln10^10_As", "n_s", "tau_reio",
            "A_planck",
        ],
        "CMB_lensing": [
            "H_0", "Omega_bh^2", "Omega_dh^2",
            "ln10^10_As", "n_s", "tau_reio",
            "A_planck",
        ],
    }

    # CMB datasets that trigger per-group reparametrisation
    cmb_names = {"CMB_hil", "CMB_hil_TT", "CMB_lowl", "CMB_lensing"}

    for mod, mod_data in models.items():
        # Core parameter list specified by the model (e.g. LCDM parameters)
        base_core = list(mod_data['parameters'])
        new_params: List[List[str]] = []

        for obs_grp in observations:
            grp_params = list(base_core)  # group-specific copy
            added: List[str] = []

            # 1) Dataset-required parameters for this group
            for o in obs_grp:
                for req in params_map.get(o, []):
                    if req not in grp_params:
                        grp_params.append(req)
                        added.append(req)

            if added:
                logger.warning(
                    "Added %s to parameters for %s in model %s",
                    added, obs_grp, mod,
                )

            # 2) Early-time calibration: any BBN in the group → do not sample r_d
            if any(str(o).startswith(("BBN",)) for o in obs_grp):
                if "r_d" in grp_params:
                    grp_params.remove("r_d")
                    logger.warning(
                        "Auto-calibrating r_d from BBN: removed 'r_d' from parameters "
                        "for %s (model %s)",
                        obs_grp, mod,
                    )

            # 3) CMB background handling is per group:
            #    If a group contains CMB data, we enforce {H_0, Omega_bh^2, Omega_dh^2}
            #    as the *sampling* parameters and derive Omega_m/Omega_b for reporting.
            grp_has_cmb = any(o in cmb_names for o in obs_grp)

            # --- NEW: CMB r_d advisory (primary CMB only; exclude lensing-only) ---
            has_primary_cmb = any(o in {"CMB_hil", "CMB_hil_TT", "CMB_lowl"} for o in obs_grp)
            if has_primary_cmb:
                if "r_d" in grp_params:
                    grp_params.remove("r_d")
                    logger.warning(
                        "Auto-calibrating r_d from CMB: removed 'r_d' from parameters for %s (model %s)",
                        obs_grp, mod,
                    )
                else:
                    # r_d may not have been present (pure CMB group), but we still want the note
                    logger.warning(
                        "Auto-calibrating r_d from CMB: r_d will be calibrated (not sampled) for %s (model %s)",
                        obs_grp, mod,
                    )

            if grp_has_cmb:
                # Ensure the physical-density parametrisation exists for CLASS/CMB
                for req in ("H_0", "Omega_bh^2", "Omega_dh^2"):
                    if req not in grp_params:
                        grp_params.append(req)

                # Do not sample Omega_b / Omega_m in CMB groups; derive them for reporting
                for redundant in ("Omega_b", "Omega_m"):
                    if redundant in grp_params:
                        grp_params.remove(redundant)

            # ✅ Append exactly once per observation group (CMB or not)
            new_params.append(grp_params)

        mod_data['parameters'] = new_params

    return models
