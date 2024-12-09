import numpy as np
import matplotlib.pyplot as plt
from MCMC_cosmology import Config, EMCEE, Statistic_packages
import Plots as MP
import os

def MODEL(z, param_dict, Type="SNe"):
    """General relativity model for the cosmological parameter."""
    model = np.sqrt(param_dict['Omega_m']*(1 + z)**3+1-param_dict['Omega_m'])#+0.5*param_dict['Omega_w'])
    return {"SNe": 1 / model, "OHD": model, "CC":model, "fsigma8":model}.get(Type, None)

# Centralized configuration parameters
parameters = ["Omega_m","H_0","gamma","sigma_8"]
true_values = {"Omega_m": 0.315, "H_0": 67.4, "gamma": 0.55, "sigma_8": 0.8}
prior_limits = {
    "Omega_m": (0.15, 0.4),
    "H_0": (60, 80),
    "gamma": (0.43, 0.68),
    "sigma_8": (0.5, 1.0),
    #"Omega_w": (0.0,1.0)
}
overwrite = False

CONFIG, data = Config.create_config(parameters, true_values, prior_limits, observation=['fsigma8'], nwalkers=20,nsteps=1000,burn=10, model_name="LCDM")
output_dirs = Config.create_output_directory(model_name=CONFIG['model_name'],observations=CONFIG["observations"])
'''
Auto Correlesation occurs when MCMC average changes between parameters is less that 0.5% over 100 iterations!!!
If true, the model is deemed converged and stops the MCMC simulation early.
'''

Samples={}
if __name__ == "__main__":
    for i, obs in  enumerate(CONFIG["observations"]):
        print(f"Running MCMC for the {CONFIG["model_name"]} model on the {obs} data (aka {CONFIG['observation_types'][i]} data)...")
        if obs=="CC":
            print ("Running Covariance matrix for CC data")
        # Correctly create the output directory and file name for each observation
        output_dir = f"MCMC_Chains/{CONFIG['model_name']}/{obs}"  # Use 'obs' here
        file_name = f"{obs}.h5"  # Use 'obs' here
        chain_path = os.path.join(output_dir, file_name)

        if not overwrite and os.path.exists(chain_path):
            print(f"Loading existing MCMC chain from {chain_path}")
            samples = EMCEE.load_mcmc_results(output_path=output_dir, file_name=file_name, CONFIG=CONFIG)
            Samples[obs] = samples  # Store the loaded samples
        else:
            Samples[obs] = EMCEE.run_mcmc(data[obs], saveChains=True, chain_path = chain_path, overwrite=overwrite, MODEL_func=MODEL, CONFIG=CONFIG, autoCorr=True, parallel=True, model_name=CONFIG['model_name'], obs=obs)


MP.make_CornerPlot(Samples, CONFIG=CONFIG)
