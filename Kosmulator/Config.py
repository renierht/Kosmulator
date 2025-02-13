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
    Create a CONFIG dictionary dynamically, ensuring each observation has its own parameter set.
    """
    observation = observation or [["SNe"]]
    true_values = true_values or {}
    prior_limits = prior_limits or {}
    
    # Define observation type mappings
    obs_type_map = {
        "JLA": "SNe",
        "Pantheon": "SNe",
        "PantheonP": "SNe",
        "OHD": "OHD",
        "CC": "CC",
        "BAO": "BAO",
        "f_sigma_8": "f_sigma_8",
        "f": "f",
    }

    config, data = {}, {}

    for mod in model_name:
        parameters_list = models[mod]['parameters']  # Now a list of lists

        ndim_list = [len(params) for params in parameters_list]  # Compute dimensions per observation
        adjusted_nwalkers = max(nwalkers, 2 * max(ndim_list) + 2)
        
        # Generate observation types dynamically
        observation_types = [
            [obs_type_map[obs_item] for obs_item in obs_list if obs_item in obs_type_map] 
            for obs_list in observation
        ]

        config[mod] = {
            "parameters": parameters_list,  # List of lists now
            "true_values": [
                [
                    true_values[p] if p in true_values else sum(prior_limits[p]) / 2 if p in prior_limits else 0.5
                    for p in param_set
                ]
                for param_set in parameters_list
            ],
            "prior_limits": [
                {p: prior_limits[p] for p in param_set if p in prior_limits}
                for param_set in parameters_list
            ],
            "observations": observation,
            "observation_types": observation_types, 
            "ndim": ndim_list,
            "nwalker": adjusted_nwalkers,
            "nsteps": nsteps,
            "burn": burn,
            "model_name": model_name,
        }

        # Load the data
        data = load_all_data(config[mod])

    return config, data

def Add_required_parameters(models, observations):
    """
    Modify the parameter list structure: ensure each observation set has the correct parameters.
    Prints warnings when parameters are added automatically.
    """
    params_map = {
        "BAO": ["H_0","r_d"], 
        "PantheonP": ["H_0","M_abs"], 
        "f_sigma_8": ["sigma_8", "gamma"], 
        "f": ["gamma"],
        "JLA": ["H_0"],
        "OHD": ["H_0"],
        "CC": ["H_0"],
        "Pantheon": ["H_0"],
    }

    for mod, mod_data in models.items():
        core_parameters = mod_data["parameters"][:]  # Copy user-defined parameters
        new_param_list = []  # Create a fresh list for observations
        
        print(f"\nProcessing model: {mod}")
        
        # Loop through each observation and assign the correct parameters
        for obs in observations:
            obs_parameters = list(core_parameters)  # Start with core parameters
            added_params = []

            for key, value in params_map.items():
                if key in obs:
                    for v in value:
                        if v not in obs_parameters:
                            obs_parameters.append(v)
                            added_params.append(v)

            # Show a warning if parameters were added
            if added_params:
                print(f"\033[4;31mSafeguard:\033[0m Added {added_params} to the parameter list for {obs} (Required for this observation type).")

            new_param_list.append(obs_parameters)  # Append modified parameter list

        # Ensure we **replace** instead of appending
        mod_data["parameters"] = new_param_list  

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