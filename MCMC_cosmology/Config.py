import numpy as np
import os
import pandas as pd
import scipy.linalg as la
import inspect
import warnings

def load_data(file_path, observations):
    """
    Load observational data from a file and return a structured dictionary.

    Args:
        file_path (str)     : Path to the data file.
        observations (str)  : Type of observation (e.g., "OHD", "CC").

    Returns:
        dict                : A dictionary with keys 'redshift', 'type_data', and 'type_error',
                              corresponding to redshift values, observational data, and errors.
    """
    data = np.loadtxt(file_path)
    return {
        "redshift": data[:, 0],
        "type_data": data[:, 1],
        "type_data_error": data[:, 2]
    }

def load_all_data(config):
    """
    Load all datasets specified in the CONFIG dictionary.

    Args:
        config (dict): The CONFIG dictionary containing an "observations" key
                       with a list of observation types.

    Returns:
        dict: A dictionary where keys are observation types and values are the loaded data.
              For standard datasets, returns (redshift, observation value, observation error).
              For 'PantheonP', returns a dictionary with specific keys for the dataset.
    """
    observation_data = {}

    for observation in config["observations"]:
        # Construct file path based on observation name
        file_path = os.path.join("./Observations", f"{observation}.dat")

        if observation == "PantheonP":
            # Special handling for the Pantheon+ dataset
            data = pd.read_csv(file_path, delim_whitespace = True)

            # Load and process the covariance matrix
            cov_path = os.path.join("./Observations", "PantheonP.cov")
            C00 = np.loadtxt(cov_path)
            cov_array = C00[1:len(C00)]
            cov_matrix = cov_array.reshape(1701, 1701)
            cov = la.cholesky(cov_matrix, lower = True, overwrite_a = True)

            # Store dataset-specific values in a dictionary
            observation_data[observation] = {
                "zHD": data["zHD"].values,
                "m_b_corr": data["m_b_corr"].values,
                "m_b_corr_err_DIAG": data["m_b_corr_err_DIAG"].values,
                "IS_CALIBRATOR": data["IS_CALIBRATOR"].values,
                "CEPH_DIST": data["CEPH_DIST"].values,
                "biasCor_m_b": data["biasCor_m_b"].values,
                "cov_matrix": cov_matrix,
                "cov": cov
            }
        elif observation == "BAO":
            # Special handling for BAO data
            data = np.loadtxt(file_path)
            observation_data[observation] = {
                "covd1": np.array(data)
            }
        else:
            # General case for other datasets
            observation_data[observation] = load_data(file_path, observations=observation)

    return observation_data

def create_config(parameters, true_values=None, prior_limits=None, observation=None,
                  Type=None, nwalkers=20, nsteps=200, burn=20, model_name=None):
    """
    Dynamically creates a CONFIG dictionary for the user's model.

    Args:
        parameters (list): List of parameter names (e.g., ["Omega_m", "H0", "alpha"]).
        true_values (dict, optional): Dictionary of true values for parameters.
                                      Defaults to the midpoint of the prior limits.
        prior_limits (dict, optional): Dictionary of prior limits for parameters.
        observation (list, optional): List of observations (e.g., ["SNe", "OHD"]). Defaults to ["SNe"].
        nwalkers (int): Number of MCMC walkers. Default is 20 for testing; recommended 100 for full runs.
        nsteps (int): Number of iterations for the MCMC simulation. Default is 200; recommended 10,000.
        burn (int): Number of burn-in steps to exclude from the final results. Default is 20; recommended 500.
        model_name (str): Name of the model being constrained.

    Returns:
        tuple: CONFIG dictionary and loaded observational data.
    """
    # Default values for true_values and observations
    if true_values is None:
        true_values = {}
    if observation is None:
        observation = ["SNe"]

    # Map observations to their types
    observation_types = []
    for obs in observation:
        if obs in ["Pantheon", "JLA", "PantheonP"]:
            observation_types.append("SNe")
        else:
            observation_types.append(obs)

    # Default model name if not provided
    if model_name is None:
        model_name = "LCDM"

    # Validate prior limits
    if prior_limits is None:
        raise ValueError("Prior limits must be provided for all parameters.")
    missing_priors = [param for param in parameters if param not in prior_limits]
    if missing_priors:
        raise ValueError(f"Missing prior limits for parameters: {', '.join(missing_priors)}")

    # Build the CONFIG dictionary
    config = {
        "parameters": parameters,
        "true_values": [true_values.get(param, np.sum(prior_limits[param]) / 2) for param in parameters],
        "prior_limits": {param: prior_limits[param] for param in parameters},
        "observations": observation,
        "ndim": len(parameters),
        "nwalker": nwalkers,
        "nsteps": nsteps,
        "burn": burn,
        "model_name": model_name,
        "observation_types": observation_types
    }

    # Load observational data
    data = load_all_data(config)

    return config, data
    
def create_output_directory(model_name="LCDM", observations=None):
    """
    Creates an output directory for saving MCMC results.

    Args:
        model_name (str): Name of the model.
        observations (list): List of observation names.

    Returns:
        dict: A dictionary mapping observation names to their corresponding output paths.
    """
    if observations is None:
        raise ValueError("No observations provided.")

    # Base directory for saving results
    save_dir = f"MCMC_Chains/{model_name}/"
    os.makedirs(save_dir, exist_ok = True)

    output_paths = {}
    for obs in observations:
        # Create subdirectory for each observation
        obs_dir = os.path.join(save_dir, obs)
        os.makedirs(obs_dir, exist_ok = True)
        output_paths[obs] = obs_dir

    return output_paths
    
def Warn_unused_params(MODEL_funcs, likelihood_func, param_dict, model_name, obs_types):
    """
    Warn about unused parameters in `param_dict` for a given set of functions.

    Args:
        funcs (list): List of functions to analyze (can include functions from different modules).
        param_dict (dict): Dictionary of parameters.
        model_name (str): Name of the model being checked.
    """
    # Define parameter requirements for each observation type
    obs_type_param_map = {
        "SNe": [],  # SNe does not use gamma or sigma_8
        "fsigma8": ["gamma", "sigma_8"],
        "sigma8": ["gamma"],
        # Add more observation types and their parameters as needed
    }

    # Aggregate relevant parameters for all observation types
    relevant_params = set()
    for obs_type in obs_types:
        relevant_params.update(obs_type_param_map.get(obs_type, []))

    # Extract used parameters from all MODEL functions and the likelihood function
    used_params = set()
    for model_func in MODEL_funcs:
        used_params.update(inspect.signature(model_func).parameters.keys())
    used_params.update(inspect.signature(likelihood_func).parameters.keys())

    # Add the relevant params for the provided observation types
    used_params.update(relevant_params)

    # Calculate unused parameters
    unused_params = set(param_dict.keys()) - used_params

    # Warn about unused parameters
    if unused_params:
        warnings.warn(
            f"\n The following parameters are unused in the {model_name} model: {unused_params} !!!\n"
        )
        
