import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from Kosmulator import Config, EMCEE, Statistic_packages
from Plots import Plots as MP  # Custom module for creating plots (e.g., autocorrelation plot)
import User_defined_modules as UDM  # Custom module with user-defined functions for cosmological calculations
from Kosmulator.MCMC_setup import run_mcmc_for_all_models

# Safeguard: Check Python version and warn if outdated
if sys.version_info[0] == 2:
    print(f'\033[4;31mNote\033[0m: Your Python version {sys.version_info[0]}.{sys.version_info[1]} is outdated. Be careful when executing the program.')

#'OHD', 'JLA', 'Pantheon', 'PantheonP', 'CC', 'BAO', 'f_sigma_8', 'sigma_8'
# Constants for the simulation
model_names = ['LCDM']
observations = [['JLA'],['OHD','CC'],['OHD','JLA','Pantheon']]
nwalkers: int = 10
nsteps: int = 500
burn: int = 10

overwrite = False
convergence = 0.01

prior_limits = {
    "Omega_m": (0.10, 0.4),
    "H_0": (60.0, 80.0),
    "M_abs": (-22.0, -15.0),
    "q0": (-0.8, -0.01),
    "q1": (-0.75, 1.0),
    "beta": (0.01, 5.0),
    "n": (0.1, 1.28),
    "gamma": (0.43, 0.68),
    "sigma_8": (0.5, 1.0),
    "Omega_w": (0.0, 1.0),
    "r_d": (100.0, 200.0),
}

true_values = {
    "Omega_m": 0.315,
    "H_0": 67.4,
    "gamma": 0.55,
    "sigma_8": 0.8,
    "q0": -0.537,
    "ns": 0.96,
    "As": 3.1,
    "Omega_b": 0.045,
    "rd": 147.5,
    "M_abs": -19.2,
}

PLOT_SETTINGS = {
    "color_schemes": ['r', 'b', 'green', 'cyan', 'yellow', 'grey', 'k', 'm', 'purple', 'gray', 'orange', 'pink', 'crimson', 'darkred', 'salmon'],
    "line_styles": ["-", "--", ":", "-."],
    "marker_size": 4,
    "legend_font_size": 10,
    "title_font_size": 14,
    "label_font_size": 12,
    "latex_enabled": True,
    "dpi": 300,
    "autocorr_save_path": "./Plots/auto_corr/",
    "Table": True,
}

# Enable LaTeX settings for plots if configured
if PLOT_SETTINGS["latex_enabled"]:
    if sys.version_info >= (3, 10):
        import shutil
        if shutil.which("latex"):
            plt.rc("text", usetex=True)
            plt.rc("font", family="Arial")
            plt.rc("text.latex", preamble=r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}")
    elif sys.version_info < (3, 10):
        from distutils.spawn import find_executable
        if find_executable("latex"):
            plt.rc("text", usetex=True)
            plt.rc("font", family="Arial")

# Display safeguard warnings
print(f"\033[33m################################################\033[0m")
print(f"\033[33m####\033[0m Safeguards + Warnings")
print(f"\033[33m################################################\033[0m")

# Prepare models and configurations
models = UDM.Get_model_names(model_names)
models = Config.Add_required_parameters(models, observations)

CONFIG, data = Config.create_config(
    models=models,
    true_values=true_values,
    prior_limits=prior_limits,
    observation=observations,
    nwalkers=nwalkers,
    nsteps=nsteps,
    burn=burn,
    model_name=model_names,
)

# Main execution block
if __name__ == "__main__":
    start = time.time()

    # Run MCMC simulations
    All_Samples = run_mcmc_for_all_models(
        models=models,
        observations=observations,
        CONFIG=CONFIG,
        data=data,
        overwrite=overwrite,
        convergence=convergence,
        PLOT_SETTINGS=PLOT_SETTINGS,
    )

    # Generate plots
    print(f"\n\033[33m################################################\033[0m")
    print(f"\033[33m####\033[0m Generating Plots :)")
    print(f"\033[33m################################################\033[0m")
    MP.generate_plots(All_Samples, CONFIG, PLOT_SETTINGS, data)

    # Print execution time
    end = time.time()
    formatted_time = Config.format_elapsed_time(end - start)
    print(f"\nAll models processed successfully in a total time of {formatted_time}!!!\n")
