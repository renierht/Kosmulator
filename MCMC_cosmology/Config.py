import numpy as np
import os

# Load data once
def load_data(file_path):
    """Load supernova data from a file."""
    data = np.loadtxt(file_path)
    return data[:, 0], data[:, 1], data[:, 2]  # redshift, observational value, error

def load_all_data(config):
    """
    Load all datasets specified in the config.

    Args:
        config (dict): The CONFIG dictionary containing an "Observations" key
                       with a list of observation types.

    Returns:
        dict: A dictionary where keys are observation types and values are tuples
              of (redshift, observation value, observation error).
    """
    observation_data = {}
    for observation in config["observations"]:
        file_path = f"./Observations/{observation}.dat"  # Assume file name matches observation type
        observation_data[observation] = load_data(file_path)
    return observation_data

def create_config(parameters, true_values=None, prior_limits=None, observations=None, nwalkers=20, nsteps=200, burn=20, model_name=None):
    """
    Dynamically creates a CONFIG dictionary for the user's model.

    Args:
        parameters (list): List of parameter names (e.g., ["Omega_m", "H0", "alpha"]).
        true_values (dict, optional): Dictionary of true values for parameters (e.g., {"Omega_m": 0.315}).
                                      If not provided, defaults to the midpoint of the prior limits.
        prior_limits (dict, optional): Dictionary of prior limits for parameters
                                        (e.g., {"Omega_m": (0.0, 1.0), "alpha": (0.0, 2.0)}).
                                        This argument is mandatory.
        observations (list, optional): List of observations (e.g., ["SNe", "OHD"]).
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
    if observations is None:
        observations = ["SNe"]
        
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
        "observations": observations,
        "ndim": len(parameters),
        "nwalker": nwalkers,  # Number of walkers
        "nsteps": nsteps,  # Number of steps
        "burn": burn,     # Burn-in steps
        "model_name": model_name
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
