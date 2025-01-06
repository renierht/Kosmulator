import numpy as np
import matplotlib.pyplot as plt
import os
from MCMC_cosmology import Config, EMCEE, Statistic_packages
import Plots as MP                                      # Custom module for creating plots (e.g., autocorrelation plot)
import User_defined_modules as UDM                      # Custom module with user-defined functions for cosmological calculations

# Centralized configuration of cosmological parameters and priors, additional parameters 
# can be uncommented or added as needed. Overwrite flag for MCMC chains; if True, rerun 
# and overwrite existing results,
overwrite = True
parameters = ["Omega_m", "H_0", "rd"]#, "ns", "As", "Omega_b"]#, "gamma", "sigma_8", "M_abs", "q0", "q1", "beta", "n"]  
true_values = {
    "Omega_m"  : 0.315, 
    "H_0"      : 67.4, 
    "gamma"    : 0.55, 
    "sigma_8"  : 0.8,
    "q0"       : -0.537,
    "ns"       : 0.96,
    "As"       : 3.1,
    "Omega_b"  : 0.045,
    "rd"       : 147.5
    }
  
prior_limits = {
    "Omega_m"  : (0.10, 0.4),  
    "H_0"      : (60.0, 80.0),
    "M_abs"    : (-22.0, -15.0),
    "q0"       : (-0.8, -0.01),
    "q1"       : (-0.75, 1.0),
    "beta"     : (0.01, 5.0),
    "n"        : (0.1, 1.28),
    "gamma"    : (0.43, 0.68),
    "sigma_8"  : (0.5, 1.0),
    "Omega_w"  : (0.0,1.0),
    "rd"       : (100.0, 200.0),
    }

'''
Create a configuration object and load observational data. 
Note: Current observations include JLA, Pantheon, OHD, CC, Pantheon+ (PantheonP), fsigma8, sigma8, BAO
'''
CONFIG, data = Config.create_config(parameters   = parameters,
                                    true_values  = true_values,
                                    prior_limits = prior_limits,
                                    observation  = ['BAO'],  
                                    nwalkers     = 100,              # Number of MCMC walkers
                                    nsteps       = 10000,             # Number of steps for each walker
                                    burn         = 1000,              # Burn-in steps to discard
                                    model_name   = "LCDM",          # Specify your model name (e.g., Lambda-CDM)
)

# Set up directories for saving outputs
output_dirs = Config.create_output_directory(model_name = CONFIG['model_name'], observations = CONFIG["observations"])


# Extracting your defined model in the User_defined_modules.py script!
MODEL = UDM.Get_model_function(CONFIG['model_name'])
#Config.Warn_unused_params([MODEL, EMCEE.model_likelihood], 
#                           EMCEE.model_likelihood, 
#                           dict(zip(CONFIG["parameters"], CONFIG["true_values"])), 
#                           CONFIG['model_name'], 
#                           CONFIG['observation_types'],
#)
'''
Note:
Auto-correlation checks are used to ensure convergence of the MCMC.
The simulation stops early if the average parameter change across 100 iterations is less than 1.0%.
'''

# Dictionary to store MCMC samples for different observations and execute the main program
Samples = {}
if __name__ == "__main__":
    # Loop through all observations defined in the CONFIG
    for i, obs in enumerate(CONFIG["observations"]):
        print(f"Running MCMC for the {CONFIG['model_name']} model on the {obs} data (aka {CONFIG['observation_types'][i]} data)...")
        
        
        # Special handling for Cosmic Chronometers (CC) data
        if obs == "CC":
            print("Running Covariance matrix for CC data")

        # Define the output directory and file name for the MCMC chain
        output_dir = f"MCMC_Chains/{CONFIG['model_name']}/{obs}"
        file_name = f"{obs}.h5"
        chain_path = os.path.join(output_dir, file_name)

        # Load existing chains if overwrite is disabled and the chain file exists
        if not overwrite and os.path.exists(chain_path):
            print(f"Loading existing MCMC chain from {chain_path}")
            samples = EMCEE.load_mcmc_results(output_path = output_dir, file_name = file_name, CONFIG = CONFIG)
            Samples[obs] = samples  # Store the loaded samples
        else:
            # Run MCMC simulation for the current observation
            Samples[obs] = EMCEE.run_mcmc(data[obs],
                                          saveChains = True,
                                          chain_path = chain_path,
                                          overwrite  = overwrite,
                                          MODEL_func = MODEL,
                                          CONFIG     = CONFIG,
                                          autoCorr   = True,  # Enable auto-correlation check
                                          parallel   = True,  # Run in parallel if possible (NB! Windows machines struggle with parallization)
                                          model_name = CONFIG['model_name'],
                                          obs        = obs,
            )

# Generate corner plots for the estimated parameters
MP.make_CornerPlot(Samples, CONFIG = CONFIG)