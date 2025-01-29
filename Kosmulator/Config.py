import numpy as np
import emcee
import os
import pandas as pd
import scipy.linalg as la
import inspect
import warnings
import re

def load_mcmc_results(output_path, file_name="tutorial.h5", CONFIG=None):
    """
    Load saved MCMC chains from an HDF5 backend.
    """
    chain_path = os.path.join(output_path, file_name)
    reader = emcee.backends.HDFBackend(chain_path)
    return reader.get_chain(discard=CONFIG['burn'], flat=True)
    
def load_data(file_path):
    """
    Load observational data from a file.
    """
    data = np.loadtxt(file_path)
    return {"redshift": data[:, 0], "type_data": data[:, 1], "type_data_error": data[:, 2]}

def load_all_data(config):
    """
    Load all datasets specified in the CONFIG dictionary.
    """
    observation_data = {}
    for obs_list in config["observations"]:
        for obs in obs_list:
            file_path = os.path.join("./Observations", f"{obs}.dat")
            if obs == "PantheonP":
                data = pd.read_csv(file_path, delim_whitespace=True)
                cov_matrix = la.cholesky(np.loadtxt("./Observations/PantheonP.cov")[1:].reshape(1701, 1701), lower=True)
                observation_data[obs] = {"zHD": data["zHD"].values,
                    "m_b_corr": data["m_b_corr"].values,
                    "m_b_corr_err_DIAG": data["m_b_corr_err_DIAG"].values,
                    "IS_CALIBRATOR": data["IS_CALIBRATOR"].values,
                    "CEPH_DIST": data["CEPH_DIST"].values,
                    "biasCor_m_b": data["biasCor_m_b"].values,
                    "cov": cov_matrix
                }
            elif obs == "BAO":
                observation_data[obs] = {"covd1": np.loadtxt(file_path)}
            else:
                observation_data[obs] = load_data(file_path)
    return observation_data

def create_config(models, true_values=None, prior_limits=None, observation=None, nwalkers=20, nsteps=200, burn=20, model_name=None):
    """
    Create a CONFIG dictionary dynamically.
    """
    observation = observation or ["SNe"]
    true_values = true_values or {}
    observation_types = [["SNe" if obs in ["Pantheon", "JLA", "PantheonP"] else obs for obs in obs_list] for obs_list in observation]
    model_name = model_name or "LCDM"
    config, data = {}, {}

    # Validate model configurations
    if not isinstance(models, dict) or not models:
        raise ValueError("Models must be provided as a non-empty dictionary.")

    for mod in model_name:
        parameters = models[mod]['parameters']
        ndim = len(parameters)
        adjusted_nwalkers = max(nwalkers, 2 * ndim + 2)
        if nwalkers < 2 * ndim:
            print(f"\033[4;31mSafeguard:\033[0m 'nwalkers' for model '{mod}' was too low ({nwalkers}). \033[34mAdjusted to {adjusted_nwalkers}\033[0m (2x+2 the number of parameters: {ndim}).")
        config[mod] = {
            "parameters": parameters,
            "true_values": [true_values.get(p, np.mean(prior_limits[p])) for p in parameters],
            "prior_limits": {p: prior_limits[p] for p in parameters},
            "observations": observation,
            "ndim": ndim,
            "nwalker": adjusted_nwalkers,
            "nsteps": nsteps,
            "burn": burn,
            "model_name": model_name,
            "observation_types": observation_types
        }
        data = load_all_data(config[mod])
    return config, data

def Add_required_parameters(models, observations):
    """
    Add required parameters like H_0 and r_d if missing, with warnings.
    """
    params_map = {"BAO": "r_d", "PantheonP": "M_abs", "f_sigma_8": ["sigma_8", "gamma"], "sigma_8": "gamma"}
    for mod, mod_data in models.items():
        parameters = mod_data["parameters"]
        if "H_0" not in parameters:
            parameters.append("H_0")
            print(f"\033[4;31mSafeguard:\033[0m Added 'H_0' to the parameter list of {mod} (Required for all models).")
        for obs in observations:
            for key, value in params_map.items():
                if key in obs:
                    if isinstance(value, list):
                        for v in value:
                            if v not in parameters:
                                parameters.append(v)
                                print(f"\033[4;31mSafeguard:\033[0m Added '{v}' to the parameter list of {mod} (Required for {key}).")
                    elif value not in parameters:
                        parameters.append(value)
                        print(f"\033[4;31mSafeguard:\033[0m Added '{value}' to the parameter list of {mod} (Required for {key}).")
    return models

def create_output_directory(model_name, observations):
    """
    Create output directories for saving MCMC results.
    """
    if not observations:
        raise ValueError("No observations provided.")
    save_dir = f"MCMC_Chains/{model_name}/"
    os.makedirs(save_dir, exist_ok=True)

    output_paths = {}
    for obs in observations:
        # Create subdirectory for each observation
        obs_dir = os.path.join(save_dir, obs)
        os.makedirs(obs_dir, exist_ok = True)
        output_paths[obs] = obs_dir
    return output_paths

def generate_label(obs):
    """Generate a label based on the observation list."""
    return obs[0] if len(obs) == 1 else '+'.join(obs)

def format_elapsed_time(seconds):
    """
    Format elapsed time into a human-readable string.
    """
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    
    if days:
        return f"{days}:{hours:02}:{minutes:02}:{seconds:02} days"
    if hours:
        return f"{hours}:{minutes:02}:{seconds:02} hours"
    if minutes:
        return f"{minutes}:{seconds:02} minutes"
    return f"{seconds} seconds"