import numpy as np
import os
import pandas as pd
import scipy.linalg as la

def load_data(file_path, observations):
    """
    Load supernova data from a file and return a structured dictionary.

    Args:
        file_path (str): Path to the data file.
        observations (str): Type of observation (e.g., "OHD", "CC").

    Returns:
        dict: A dictionary with keys 'redshift', 'type_data', and 'type_error', corresponding to
              redshift values, observational data, and errors.
    """
    data = np.loadtxt(file_path)
    return {"redshift": data[:, 0], "type_data": data[:, 1], "type_data_error": data[:, 2]}

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
        file_path = os.path.join("./Observations", f"{observation}.dat")  # File path based on observation name

        if observation == "PantheonP":
            # Special handling for Pantheon+ dataset
            print(f"Loading Pantheon+ dataset from {file_path}")
            data = pd.read_csv(file_path, delim_whitespace=True)
            cov_path = os.path.join("./Observations", "PantheonP.cov")
            C00 = np.loadtxt(cov_path)
            cov_array = C00[1:len(C00)]  
            cov_matrix = cov_array.reshape(1701, 1701)
            cov = la.cholesky(cov_matrix, lower=True, overwrite_a=True) 
            
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
        else:
            # General case for other datasets
            print(f"Loading {observation} dataset from {file_path}")
            observation_data[observation]=load_data(file_path, observations=observation)
    return observation_data

def create_config(parameters, true_values=None, prior_limits=None, observation=None, Type=None, nwalkers=20, nsteps=200, burn=20, model_name=None):
    """
    Dynamically creates a CONFIG dictionary for the user's model.

    Args:
        parameters (list): List of parameter names (e.g., ["Omega_m", "H0", "alpha"]).
        true_values (dict, optional): Dictionary of true values for parameters (e.g., {"Omega_m": 0.315}).
                                      If not provided, defaults to the midpoint of the prior limits.
        prior_limits (dict, optional): Dictionary of prior limits for parameters
                                        (e.g., {"Omega_m": (0.0, 1.0), "alpha": (0.0, 2.0)}).
                                        This argument is mandatory.
        observation (list, optional): List of observations (e.g., ["SNe", "OHD"]).
                                        Defaults to ["SNe"] if not provided.
        nwalker (int): Number of MCMC walkers. Default of 20 setup for test run. Recommended 100 for full runs
        nsteps (int): Number of iterations the MCMC simulations must make before stopping. Default of 200 setup
                        for test run. Recommended 10 000 for full runs.
        burn (int): The first x amount of iterations will not be included in the final Gaussian distribution
                    plot. Improving the general statistics. Default of 20 setup for test runs. Recommended 500
                    for full runs.
        model_name (str): Name of the model you are constraining

    Returns:
        dict: CONFIG dictionary for the user's model.
    """
    # Default values
    if true_values is None:
        true_values = {}
        
    if observation is None:
        observations = ["SNe"]
    
    # Map each observation to its type
    observation_types = []
    for obs in observation:
        if obs in ["Pantheon", "JLA", "PantheonP"]:
            observation_types.append("SNe")
        else:
            observation_types.append(obs)
        
    if model_name is None:
        model_name = "LCDM"

    if prior_limits is None:
        raise ValueError("Prior limits must be provided for all parameters.")
    missing_priors = [param for param in parameters if param not in prior_limits]
    if missing_priors:
        raise ValueError(
            f"Missing prior limits for parameters: {', '.join(missing_priors)}"
        )

    # Build the CONFIG dictionary
    config = {
        "parameters": parameters,
        "true_values": [true_values.get(param, np.sum(prior_limits[param])/2) for param in parameters],
        "prior_limits": {param: prior_limits[param] for param in parameters},
        "observations": observation,
        "ndim": len(parameters),
        "nwalker": nwalkers,  # Number of walkers
        "nsteps": nsteps,  # Number of steps
        "burn": burn,     # Burn-in steps
        "model_name": model_name,
        "observation_types": observation_types
    }
    data = load_all_data(config)
    return config, data
    
def create_output_directory(model_name="LCDM", observations=None):
    """
    Creates an output directory for MCMC results.

    Args:
        base_dir (str): The base directory where outputs will be saved.
        observations (list): List of observation names.

    Returns:
        dict: A dictionary where keys are observation names, and values are the corresponding output paths.
    """
    save_dir = f"MCMC_Chains/{model_name}/"
    if observations is None:
        raise ValueError("No observations provided.")

    output_paths = {}
    os.makedirs(save_dir, exist_ok=True)  # Create the base directory if it doesn't exist

    for obs in observations:
        obs_dir = os.path.join(save_dir, obs)
        os.makedirs(obs_dir, exist_ok=True)  # Create a directory for each observation
        output_paths[obs] = obs_dir

    return output_paths
