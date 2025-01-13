import numpy as np
import os
import pandas as pd
import scipy.linalg as la
import inspect
import warnings
import re

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

    for i,observation in enumerate(config["observations"]):
        for a in range(0,len(observation)):
            # Construct file path based on observation name
            file_path = os.path.join("./Observations", f"{observation[a]}.dat")

            if observation[a] == "PantheonP":
                # Special handling for the Pantheon+ dataset
                data = pd.read_csv(file_path, delim_whitespace = True)

                # Load and process the covariance matrix
                cov_path = os.path.join("./Observations", "PantheonP.cov")
                C00 = np.loadtxt(cov_path)
                cov_array = C00[1:len(C00)]
                cov_matrix = cov_array.reshape(1701, 1701)
                cov = la.cholesky(cov_matrix, lower = True, overwrite_a = True)

                # Store dataset-specific values in a dictionary
                observation_data[observation[a]] = {
                    "zHD": data["zHD"].values,
                    "m_b_corr": data["m_b_corr"].values,
                    "m_b_corr_err_DIAG": data["m_b_corr_err_DIAG"].values,
                    "IS_CALIBRATOR": data["IS_CALIBRATOR"].values,
                    "CEPH_DIST": data["CEPH_DIST"].values,
                    "biasCor_m_b": data["biasCor_m_b"].values,
                    "cov_matrix": cov_matrix,
                    "cov": cov
                }
            elif observation[a] == "BAO":
                # Special handling for BAO data
                data = np.loadtxt(file_path)
                observation_data[observation[a]] = {
                    "covd1": np.array(data)
                }
            else:
                # General case for other datasets
                observation_data[observation[a]] = load_data(file_path, observations=observation[a])

    return observation_data

def create_config(models, true_values=None, prior_limits=None, observation=None,
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
    #for i,obs in enumerate(observation):
    for obs in observation:
        types = [ ]
        #for a in range(0,len(obs)):
        for a in obs:
            if a in ["Pantheon", "JLA", "PantheonP"]:
                types.append("SNe")
            else:
                types.append(a)
        observation_types.append(types)

    # Default model name if not provided
    if model_name is None:
        model_name = "LCDM"

    # Validate model configurations
    if not isinstance(models, dict) or not models:
        raise ValueError("Models must be provided as a non-empty dictionary.")
        
    #for model_name, model_config in models.items():
     #   if "parameters" not in model_config or not model_config["parameters"]:
      #      raise ValueError(f"Model '{model_name}' must define a non-empty 'parameters' list.")
       # for param in model_config["parameters"]:
        #    if param not in prior_limits:
         #       raise ValueError(f"Prior limits for parameter '{param}' are missing in model '{model_name}'.")

    # Build the CONFIG dictionary
    config = {}
    data = {}
    for mods in model_name:
        parameters = models[mods]['parameters']
        config[mods] = {
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
                
        #print (config[mods]['observations'])
        # Load observational data
        data = load_all_data(config[mods])
        #print (data)
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

def find_used_params(func, param_dict_keys, obs_type=None):
    '''
    Find used parameters in the source code of a function.

    Args:
        func (function): The function to analyze.
        param_dict_keys (iterable): The keys of the parameter dictionary to check.
        obs_type (str): Current observation type (optional).

    Returns:
        set: A set of parameters used in the function.
    '''
    try:
        source = inspect.getsource(func)
        used = set()
        for key in param_dict_keys:
            if re.search(rf"param_dict\[['\"]{key}['\"]\]", source):
                used.add(key)
        return used
    except OSError as e:
        print(f"Debug: Could not inspect {func.__name__}: {e}")
        return set()
            
def Warn_unused_params(MODEL_funcs, likelihood_func, param_dict, model_name, obs_type):
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
        "BAO": ["rd"],
        # Add more observation types and their parameters as needed
    }
    
    # Initialize sets for tracking unused parameters across all observation types
    total_used_params = set()
    param_dict_keys = set(param_dict.keys())

    # Process each observation
    for obs in obs_type:
        relevant_params = set(obs_type_param_map.get(obs_type, []))

        # Extract used parameters for each function
        used_params = set()
        for model_func in MODEL_funcs:
            used_in_model = find_used_params(model_func, param_dict_keys, obs_type)
            used_params.update(used_in_model)

        used_in_likelihood = find_used_params(likelihood_func, param_dict_keys, obs_type)
        used_params.update(used_in_likelihood)
        
         # Add explicitly relevant params for this observation
        used_params.update(relevant_params)

        # Accumulate total used params
        total_used_params.update(used_params)
    
    # Identify unused parameters across all observations
    unused_params = param_dict_keys - total_used_params

    # Debugging information
    # print("Debug: Total used parameters:", total_used_params)
    # print("Debug: Param dict keys:", param_dict_keys)
    
    # Warn about unused parameters
    if unused_params:
        warnings.warn(
            f"The following parameters are unused for the {model_name} model and the model_likelihood function: {unused_params} !!!"
        )
        print ("\n")
        
def generate_label(obs):
    """
    Generate a label based on the observation list.

    Parameters:
    - obs: List of observations (e.g., ['JLA'], ['OHD'], ['JLA', 'OHD'])

    Returns:
    - str: A label string for the plot
    """
    if len(obs) == 1:
        # Single entry: return the first item as the label
        return obs[0]
    else:
        # Multiple entries: join them with '+'
        return '+'.join(obs)  

def format_elapsed_time(seconds):
    """
    Format elapsed time into days, hours, minutes, and seconds.

    Parameters:
    - seconds (int or float): Total elapsed time in seconds.

    Returns:
    - str: A formatted string representing the elapsed time.
    """
    seconds = int(seconds)  # Ensure seconds is an integer
    days, seconds = divmod(seconds, 86400)  # 86400 seconds in a day
    hours, seconds = divmod(seconds, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(seconds, 60)  # 60 seconds in a minute

    if days > 0:
        return f"{days}:{hours:02}:{minutes:02}:{seconds:02} days"
    elif hours > 0:
        return f"{hours}:{minutes:02}:{seconds:02} hours"
    elif minutes > 0:
        return f"{minutes}:{seconds:02} minutes"
    else:
        return f"{seconds} seconds"