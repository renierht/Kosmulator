import numpy as np
import matplotlib.pyplot as plt
from MCMC_cosmology import EMCEE as MCMC
import Plots as MP
from MCMC_cosmology import Config 

def MODEL(z, param_dict, Type="SNe"):
    """General relativity model for the cosmological parameter."""
    model = np.sqrt(param_dict['Omega_m']*(1 + z)**3+1-param_dict['Omega_m'])#+0.5*param_dict['Omega_w'])
    return {"SNe": 1 / model, "OHD": model, "CC":model}.get(Type, None)

# Centralized configuration parameters
parameters = ["Omega_m","H_0"]#,"Omega_w"]
true_values = {"Omega_m": 0.315, "H_0": 67.4}
prior_limits = {
    "Omega_m": (0.15, 0.4),
    "H_0": (60, 80),
    #"n": (-0.2, 5.0),
    #"Omega_w": (0.0,1.0)
}

CONFIG, data = Config.create_config(parameters, true_values, prior_limits, observations=["SNe", "OHD", "CC"],nwalkers=10,nsteps=1000,burn=50, model_name="CC_test")
output_dirs = Config.create_output_directory(model_name=CONFIG['model_name'],observations=CONFIG["observations"])

'''
Auto Correlesation occurs when MCMC average changes between parameters is less that 0.5% over 100 iterations!!!
If true, the model is deemed converged and stops the MCMC simulation early.
'''

Samples={}
if __name__ == "__main__":
    for obs in CONFIG["observations"]:
        print(f"Running MCMC for the {CONFIG["model_name"]} model on the {obs} data...")
        if obs=="CC":
            print ("Running Covariance matrix for CC data")
        Samples[obs]= MCMC.run_mcmc(data[obs], saveChains=True, overwrite=True, Type=[obs], MODEL_func=MODEL, CONFIG=CONFIG, autoCorr=True, parallel=True, model_name=CONFIG['model_name'])


MP.make_CornerPlot(Samples, CONFIG=CONFIG)
