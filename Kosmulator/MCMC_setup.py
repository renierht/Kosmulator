import os
import sys
from Kosmulator import Config, EMCEE
import User_defined_modules as UDM

# Add the parent directory to the Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_mcmc_for_all_models(models, observations, CONFIG, data, overwrite, convergence, PLOT_SETTINGS, use_mpi, num_cores, suffix=""):
    """
    Run MCMC simulations for all models and observations.

    Args:
        models (dict): Dictionary of model names and parameters.
        observations (list): List of observation sets.
        CONFIG (dict): Configuration dictionary for all models.
        data (dict): Observational data.
        overwrite (bool): Whether to overwrite existing chains.
        convergence (float): Convergence threshold for MCMC.
        PLOT_SETTINGS (dict): Plotting configuration.

    Returns:
        dict: Dictionary of all MCMC samples for all models and observations.
    """
    All_Samples = {}

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except ImportError:
        rank = 0
    
    # Loop through each model in the configuration
    for j, model_name in enumerate(CONFIG[list(models.keys())[0]]['model_name']):
        
        if rank == 0:
            print(f"\n\033[33m{'#'*48}\033[0m")
            print(f"\033[33m####\033[0m Processing model: \033[4;31m{model_name}\033[0m")
            print(f"\033[33m{'#'*48}\033[0m")
        
        Samples = {}  # Dictionary to store samples for the current model

        # Loop through all observation sets for the current model
        for i, obs in enumerate(CONFIG[list(models.keys())[j]]['observations']):
            last_obs = i == (len(observations) - 1)

            # Generate a name for the observation set
            if len(obs) > 1:
                observations_name = '_'.join(CONFIG[list(models.keys())[j]]['observations'][i])
            else:
                observations_name = CONFIG[list(models.keys())[j]]['observations'][i][0]

            # Create the output directories and retrieve the model function
            #output_dirs = Config.create_output_directory(model_name=model_name, observations=[observations_name])
            base_dir, output_dirs = Config.create_output_directory(model_name, [observations_name], suffix=suffix)
            MODEL = UDM.Get_model_function(model_name)

            # Print observation details
            for a in range(len(obs)):
                obs_type = CONFIG[list(models.keys())[j]]['observation_types'][i][a]
                if rank == 0: 
                    print(f"Observations:             \033[34m{obs[a]}\033[0m data (aka \033[34m{obs_type}\033[0m data)")

            # Define the output directory and file name for the MCMC chain
            #output_dir = f"MCMC_Chains/{model_name}/{observations_name}"
            output_dir = output_dirs[observations_name]
            file_name = f"{observations_name}.h5"
            chain_path = os.path.join(output_dir, file_name)

            # Load existing MCMC chains if not overwriting
            if not overwrite and os.path.exists(chain_path):
                if rank == 0:
                    print(f"Loading existing MCMC chain from {chain_path}\n")
                    samples = EMCEE.load_mcmc_results(output_path=output_dir, file_name=file_name, CONFIG=CONFIG[list(models.keys())[j]])
                    observation_key = '+'.join(obs) if len(obs) > 1 else obs[0]
                    Samples[observation_key] = samples
            else:
                label = Config.generate_label(obs)
                Samples[label] = EMCEE.run_mcmc(
                    data=data,
                    saveChains=True,
                    chain_path=chain_path,
                    overwrite=overwrite,
                    MODEL_func=MODEL,
                    CONFIG=CONFIG[list(models.keys())[j]],
                    autoCorr=True,
                    parallel=True,
                    model_name=model_name,
                    obs=obs,
                    Type=CONFIG[list(models.keys())[j]]['observation_types'][i],
                    colors=PLOT_SETTINGS['color_schemes'][i],
                    convergence=convergence,
                    last_obs=last_obs,
                    PLOT_SETTINGS=PLOT_SETTINGS,
                    obs_index = i,
                    use_mpi=use_mpi,
                    num_cores = num_cores,
                )

        All_Samples[model_name] = Samples  # Store all samples for the current model

    return All_Samples
