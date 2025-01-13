import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from Kosmulator import Config, EMCEE, Statistic_packages
from Plots import Plots as MP                                                       # Custom module for creating plots (e.g., autocorrelation plot)
import User_defined_modules as UDM                      # Custom module with user-defined functions for cosmological calculations

if sys.version_info[0] == 2:
        print(f'\033[4;31mNote\033[0m: Your Python version {sys.version_info[0]}.{sys.version_info[1]} is outdated. Be carefull when executing the program')

'''
Create a configuration object and load observational data. 
Note: Current observations include JLA, Pantheon, OHD, CC, Pantheon+ (PantheonP), fsigma8, sigma8, BAO
Note: Auto-correlation checks are used to ensure convergence of the MCMC.
        The simulation stops early if the average parameter change across 100 iterations is less than 1.0% (changable).
'''
model_names = [ "LCDM"]
observations = [['JLA'],['OHD']]
nwalkers : int = 10
nsteps : int = 1000
burn : int = 10

overwrite = True
convergence = 0.01
color_schemes = ['r', 'b', 'green', 'cyan', 'yellow', 'grey', 'k', 'm']

prior_limits = {
    "Omega_m"   : (0.10, 0.4),  
    "H_0"           : (60.0, 80.0),
    "M_abs"       : (-22.0, -15.0),
    "q0"             : (-0.8, -0.01),
    "q1"             : (-0.75, 1.0),
    "beta"         : (0.01, 5.0),
    "n"               : (0.1, 1.28),
    "gamma"       : (0.43, 0.68),
    "sigma_8"   : (0.5, 1.0),
    "Omega_w"   : (0.0,1.0),
    "rd"             : (100.0, 200.0),
    }

true_values = {
    "Omega_m"   : 0.315, 
    "H_0"           : 67.4, 
    "gamma"       : 0.55, 
    "sigma_8"   : 0.8,
    "q0"             : -0.537,
    "ns"             : 0.96,
    "As"             : 3.1,
    "Omega_b"   : 0.045,
    "rd"             : 147.5
    }

models = UDM.Get_model_names(model_names)

CONFIG, data = Config.create_config(models                   = models,
                                                                     true_values         = true_values,
                                                                     prior_limits       = prior_limits,
                                                                     observation         = observations, #,['JLA', 'OHD','Pantheon']], 
                                                                     nwalkers               = nwalkers,              # Number of MCMC walkers
                                                                     nsteps                   = nsteps,             # Number of steps for each walker
                                                                     burn                       = burn,              # Burn-in steps to discard
                                                                     model_name           = model_names         # Specify your model name (e.g., Lambda-CDM)
)

# Dictionary to store MCMC samples for all models and observations
All_Samples = {}
if __name__ == "__main__":
    start = time.time()
    # Loop over all models
    for j, model_name in enumerate(CONFIG[list(models.keys())[0]]['model_name']):   # j loops over the models
        print (f"\n\033[33m################################################\033[0m\n\033[33m####\033[0m Processing model:         \033[4;31m{model_name}\033[0m")
        print ("\033[33m################################################\033[0m")

        # Dictionary to store samples for the current model
        Samples = {}
        # Loop through all observations defined in the CONFIG
        for i, obs in enumerate(CONFIG[list(models.keys())[j]]["observations"]):  # i loops over the observations
            if len(obs)>1:
                observations_name =  '_'.join(CONFIG[list(models.keys())[j]]["observations"][i])
            else:
                observations_name = CONFIG[list(models.keys())[j]]["observations"][i][0]
            observations_list = [ observations_name]
            # Set up directories for saving outputs for the current model
            output_dirs = Config.create_output_directory(model_name = model_name, observations = observations_list)

            # Extract the model function for the current model
            MODEL = UDM.Get_model_function(model_name)
            
            if len(obs)>1:
                for a in range(0,len(obs)):
                    print(f"Observations:             Combo \033[34m{obs[a]}\033[0m data (aka \033[34m{CONFIG[list(models.keys())[j]]['observation_types'][i][a]}\033[0m data)")
            else:
                print(f"Observations:             \033[34m{obs[0]}\033[0m data (aka \033[34m{CONFIG[list(models.keys())[j]]['observation_types'][i][0]}\033[0m data)")
            #Config.Warn_unused_params([MODEL, EMCEE.model_likelihood], EMCEE.model_likelihood, dict(zip(CONFIG["parameters"], CONFIG["true_values"])), 
            #                                                   CONFIG['model_name'], CONFIG['observation_types'][i],)

            # Define the output directory and file name for the MCMC chain
            output_dir = f"MCMC_Chains/{model_name}/{observations_list[0]}"
            file_name = f"{observations_list[0]}.h5"
            chain_path = os.path.join(output_dir, file_name)

            # Load existing chains if overwrite is disabled and the chain file exists
            if not overwrite and os.path.exists(chain_path):
                print(f"Loading existing MCMC chain from {chain_path}")
                samples = EMCEE.load_mcmc_results(output_path = output_dir, file_name = file_name, CONFIG = CONFIG[list(models.keys())[j]])
                Samples[obs[0]] = samples  # Store the loaded samples
            else:
                # Run MCMC simulation for the current observation
                Samples[Config.generate_label(obs)] = EMCEE.run_mcmc(data,
                                                saveChains  = True,
                                                chain_path  = chain_path,
                                                overwrite    = overwrite,
                                                MODEL_func  = MODEL,
                                                CONFIG          = CONFIG[list(models.keys())[j]],
                                                autoCorr      = True,  # Enable auto-correlation check
                                                parallel      = True,  # Run in parallel if possible (NB! Windows machines struggle with parallization)
                                                model_name  = model_name,
                                                obs                = obs,
                                                Type              = CONFIG[list(models.keys())[j]]['observation_types'][i], 
                                                colors           = color_schemes[i],
                                                convergence = convergence,
            )
        # Generate corner plots for the estimated parameters
        MP.make_CornerPlot(Samples, CONFIG = CONFIG[list(models.keys())[j]], model_name = model_name, color = color_schemes)
        
        # Store the samples for the current model
        All_Samples[model_name] = Samples
    end = time.time()
    formatted_time = Config.format_elapsed_time(end-start)
    print(f"\nAll models processed successfully in a total time of {formatted_time}!!!\n")
    
    #print (np.shape(All_Samples['LCDM']['JLA']))