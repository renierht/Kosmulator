import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from Kosmulator import Config, EMCEE, Statistic_packages
from Plots import Plots as MP  # Custom module for creating plots (e.g., autocorrelation plot)
import User_defined_modules as UDM  # Custom module with user-defined functions for cosmological calculations
from Kosmulator.MCMC_setup import run_mcmc_for_all_models
from Plots.Plot_functions import print_aligned_latex_table

# Safeguard: Check Python version and warn if outdated
if sys.version_info[0] == 2:
    print(f'\033[4;31mNote\033[0m: Your Python version {sys.version_info[0]}.{sys.version_info[1]} is outdated. Be careful when executing the program.')

#'OHD', 'JLA', 'Pantheon', 'PantheonP', 'CC', 'BAO', 'f_sigma_8', 'f'
# Constants for the simulation
model_names = ["BetaRn"]
observations =  [['JLA','BAO','f_sigma_8'],['JLA'],['BAO'],['f'],['f_sigma_8'], ['OHD'],['CC']]
true_model = "LCDM" # True model will always run first irregardless of model names, due to the statistical analysis
nwalkers: int = 10
nsteps: int = 200
burn: int = 10

overwrite = False
convergence = 0.01

prior_limits = {
    "Omega_m": (0.10, 0.4),
    "H_0": (60.0, 80.0),
    "r_d": (100.0, 200.0),
    "M_abs": (-22.0, -15.0),
    "zeta": (0.0,0.3),
    "gamma": (0.4, 0.7),
    "sigma_8": (0.5, 1.0),
    "n": (0.15,1.0),
    "p": (0.0, 1.0),
    "Gamma": (2.0, 10.0),
    "q0": (-0.8, -0.01),
    "q1": (-0.75, 1.0),
    "beta": (0.01, 5.0),
    "Omega_w": (0.0, 1.0),
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
full_colors = ['r', 'b', 'green', 'cyan', 'purple', 'grey', 'yellow', 'm', 
               'k', 'gray', 'orange', 'pink', 'crimson', 'darkred', 'salmon']
               
PLOT_SETTINGS = {
    "color_schemes": full_colors[:len(observations)],
    "line_styles": ["-", "--", ":", "-."],
    "marker_size": 4,
    "legend_font_size": 12,
    "title_font_size": 12,
    "label_font_size": 12,
    "latex_enabled": True,
    "dpi": 300,
    "autocorr_save_path": "./Plots/auto_corr/",
    "Table": True,
    "table_anchor": (0.98, 1.0),  # pushes the table further up

    # Additional table-specific settings for finer control:
    "base_table_width": 0.4,
    "width_increment": 0.045,
    "base_table_height": 0.3,
    "height_increment": 0.04,
    "font_base": 14,
    "font_reduction": 0.7,
    "min_font_size": 10,
    "cell_height_base": 0.10,
    "cell_scaling": 3.8,
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

# Ensure the true_model is at the front of the model_names list
if true_model in model_names:
    model_names.remove(true_model)  # Remove it from its current position
model_names.insert(0, true_model)  # Insert it at the front

# Display safeguard warnings
print(f"\033[33m################################################\033[0m")
print(f"\033[33m####\033[0m Safeguards + Warnings")
print(f"\033[33m################################################\033[0m")

# Prepare models and configurations
models = UDM.Get_model_names(model_names)  # Get initial model names
models = Config.Add_required_parameters(models, observations)  # Ensure observations get correct parameters

CONFIG, data = Config.create_config(  # Do NOT call Add_required_parameters again inside create_config!
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

    best_fit_values, All_LaTeX_Tables, statistical_results = MP.generate_plots(All_Samples, CONFIG, PLOT_SETTINGS, data, true_model)

    # Print LaTeX tables for all models
    for model_name, (aligned_table, parameter_labels, observation_names) in All_LaTeX_Tables.items():
        print(f"\nModel: {model_name} Aligned LaTeX Table:")
        print_aligned_latex_table(aligned_table, parameter_labels, observation_names)

    # Print execution time
    end = time.time()
    formatted_time = Config.format_elapsed_time(end - start)
    print(f"\nAll models processed successfully in a total time of {formatted_time}!!!\n")