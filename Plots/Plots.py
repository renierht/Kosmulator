import matplotlib.pyplot as plt
from getdist import plots, MCSamples
from Kosmulator.Config import generate_label
import User_defined_modules as UDM                      # Custom module with user-defined functions for cosmological calculations
from scipy.interpolate import interp1d
from Plots.Plot_functions import *
from Kosmulator import Statistic_packages as SP
import numpy as np
import shutil
import sys
import os
import re

def generate_plots(All_Samples, CONFIG, PLOT_SETTINGS, data):
    """Generate corner plots and best-fit plots for all models."""
    All_best_fit_values = {}
    for model_name, Samples in All_Samples.items():
        print(f"\nCreating corner plot for model {model_name}...")
        All_best_fit_values[model_name] = make_CornerPlot(Samples, CONFIG[model_name], model_name, model_name, PLOT_SETTINGS)
    print("\nPlotting Best-Fit models onto its data source...")
    best_fit_plots(All_best_fit_values, CONFIG, data, PLOT_SETTINGS["color_schemes"])
    
def autocorrPlot(autocorr, index, model_name, color, obs, PLOT_SETTINGS, close_plot=False, nsteps=100):
    """
    Generate and save the autocorrelation plot dynamically during MCMC sampling.

    Args:
        autocorr (array-like): Autocorrelation values.
        index (int): Current iteration index.
        model_name (str): Name of the model being sampled.
        obs (str): Observation label.
        PLOT_SETTINGS (dict): Global plot configuration settings from Kosmulator.py.
        close_plot (bool, optional): Whether to close the plot after saving. Defaults to False.
        nsteps (int, optional): Total number of steps for the MCMC run. Defaults to 100.
    """
    if close_plot:
        plt.close()
        return
    
    # Ensure output directory exists
    folder_path = PLOT_SETTINGS.get("autocorr_save_path", "./Plots/auto_corr/")
    os.makedirs(folder_path, exist_ok=True)
    
    # Prepare data    
    n = 100 * np.arange(1, index + 1)
    y = autocorr[:index]
        
    # Plot setup
    plt.plot(n, n / 100.0, linestyle='--', color="k")
    if index == 1:
        plt.plot(n, y, label = f'{generate_label(obs)}', color = color)
    else:
        plt.plot(n, y, color = color)
    
    plt.ylim(0, max(100, autocorr[:index].max() + 10))
    plt.xlim(50, nsteps)
    plt.title(f"Auto-Correlator: Check for convergence - {model_name} model")
    plt.xlabel("Number of steps", fontsize=PLOT_SETTINGS.get("label_font_size", 12))
    plt.ylabel(r"Mean $\hat{\tau}$", fontsize=PLOT_SETTINGS.get("label_font_size", 12))
    plt.legend(fontsize=PLOT_SETTINGS.get("legend_font_size", 10))
    plt.savefig(f"./Plots/auto_corr/{model_name}.png", dpi=PLOT_SETTINGS.get("dpi", 200))
       
def make_CornerPlot(Samples, CONFIG, model_name, save_file_name, PLOT_SETTINGS):
    """Generate a corner plot from MCMC samples with optional LaTeX table."""
    print("\n\033[4;31mNote\033[0m: GetDist's read chains ignore burn-in, since EMCEE already applies a burn fraction.")

    # Ensure output folder exists
    folder_path = f"./Plots/corner_plots/{model_name}/"
    os.makedirs(folder_path, exist_ok=True)

    # Prepare parameter labels
    parameter_labels = greek_Symbols(CONFIG['parameters']) if PLOT_SETTINGS.get("latex_enabled", False) else CONFIG['parameters']

    # Initialize GetDist plotter
    g = plots.get_subplot_plotter(subplot_size=4, subplot_size_ratio=0.8)
    g.settings.figure_legend_frame = True
    g.settings.tight_layout = True
    g.settings.alpha_filled_add = 0.4
    g.settings.title_limit_fontsize = PLOT_SETTINGS.get("title_font_size", 12)
    g.settings.legend_fontsize = PLOT_SETTINGS.get("legend_font_size", 12)
    g.settings.fontsize = PLOT_SETTINGS.get("label_font_size", 12)

    # Extract plot settings
    line_styles = PLOT_SETTINGS.get("line_styles", ['-', '--', ':', '-.'])
    line_widths = [1.2, 1.5] * (len(line_styles) // 2 + 1)
    color_schemes = PLOT_SETTINGS.get("color_schemes", ['r', 'b', 'green', 'cyan'])

    # Verify Samples
    if not Samples:
        raise ValueError("Samples dictionary is empty or not properly set up.")

    # Prepare distributions and labels
    distributions = []
    labels = []
    for obs in Samples:
        if not Samples[obs].size:
            raise ValueError(f"Samples for '{obs}' are empty.")
        distribution = MCSamples(samples=Samples[obs], names=parameter_labels, labels=parameter_labels)
        distributions.append(distribution)
        labels.append(obs)

    # Calculate LaTeX table and structured values
    results, latex_table, structured_values = SP.calculate_asymmetric_from_samples(Samples, CONFIG['parameters'])

    # Plot corner plot
    num_cols = (len(labels) + 1) // 2  # Maximum 2 entries per column
    g.triangle_plot(
        distributions,
        filled=True,
        legend_labels=labels,
        legend_loc='upper right',
        legend_ncol=num_cols,
        line_args=[{'ls': ls, 'lw': lw, 'color': color_schemes[i % len(color_schemes)]}
                   for i, (ls, lw) in enumerate(zip(line_styles, line_widths))]
    )

    # Optionally add table to the plot
    if PLOT_SETTINGS and PLOT_SETTINGS.get("Table", False):
        add_corner_table(g, latex_table, labels, parameter_labels, PLOT_SETTINGS, len(CONFIG['parameters']))

    # Save the plot
    plt.savefig(f"{folder_path}/{save_file_name}.png", dpi=PLOT_SETTINGS.get("dpi", 300))
    plt.close()

    return structured_values
    
def best_fit_plots(All_best_fit_values, CONFIG, data, color_schemes):
    """
    Generate best-fit plots for each model and observation combination.
    """
    for model_name, model_best_fit in All_best_fit_values.items():
        observations = CONFIG[model_name]["observations"]
        observation_types = CONFIG[model_name]["observation_types"]

        for obs_index, obs_set in enumerate(observations):
            obs_name = "_".join(obs_set)
            folder_path = f"./Plots/Best_fits/{model_name}/{obs_name}/"
            setup_folder(folder_path)

            obs_types = observation_types[obs_index]
            if len(set(obs_types)) > 1 and not (set(obs_types) == {'OHD', 'CC'} or set(obs_types) == {'CC', 'OHD'}):
                print(f"Skipping combination plot for {obs_set} due to mixed observation types (excluding OHD+CC): {obs_types}")
                continue

            combined_best_fit = model_best_fit["+".join(obs_set)]
            params_combined_median, params_combined_upper, params_combined_lower = fetch_best_fit_values(combined_best_fit)

            combined_redshift, combined_type_data, combined_type_data_error = prepare_data(obs_set, data, params_combined_median)

            redshiftx = np.linspace(0.005, max(combined_redshift) + 0.05, 10000)
            obs_type = observation_types[obs_index][0]

            model_combined_median, y_label = compute_model(model_name, redshiftx, params_combined_median, obs_type=obs_type)
            model_combined_upper, _ = compute_model(model_name, redshiftx, params_combined_upper, obs_type=obs_type)
            model_combined_lower, _ = compute_model(model_name, redshiftx, params_combined_lower, obs_type=obs_type)

            # Create the plot
            fig, (ax_main, ax_residual) = plt.subplots(
                nrows=2, ncols=1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1], "hspace": 0}, sharex=True
            )

            ax_main.plot(
                redshiftx, model_combined_median,
                label=f"{model_name} (Combined: {'+'.join(obs_set)})",
                color=color_schemes[0], linestyle="-", zorder=3, linewidth=2
            )
            ax_main.fill_between(
                redshiftx, model_combined_lower, model_combined_upper,
                color="k", alpha=0.5, label="Model Uncertainty", zorder=2
            )

            for i, obs in enumerate(obs_set):
                obs_data = data[obs]
                if obs == 'PantheonP':
                    redshift = obs_data['zHD']
                    type_data = obs_data['m_b_corr'] - params_combined_median['M_abs']
                    type_data_error = np.zeros(len(type_data))
                else:
                    redshift = obs_data["redshift"]
                    type_data = obs_data["type_data"]
                    type_data_error = obs_data["type_data_error"]

                ax_main.errorbar(
                    redshift, type_data, yerr=type_data_error,
                    fmt="o", label=obs, color=color_schemes[(i + 1) % len(color_schemes)],
                    capsize=3, markersize=3, zorder=2
                )

            ax_main.set_ylabel(y_label)
            ax_main.set_xlim(0, max(combined_redshift) + 0.05)
            ax_main.legend(loc="lower right")
            ax_main.grid(False)

            interpolator = interp1d(redshiftx, model_combined_median, kind="linear", bounds_error=False, fill_value="extrapolate")
            model_at_combined_data = interpolator(combined_redshift)

            residual_start = 0
            for i, obs in enumerate(obs_set):
                obs_data = data[obs]
                if obs == 'PantheonP':
                    redshift = obs_data['zHD']
                    type_data = obs_data['m_b_corr'] - params_combined_median['M_abs']
                    type_data_error = np.zeros(len(type_data))
                else:
                    redshift = obs_data["redshift"]
                    type_data = obs_data["type_data"]
                    type_data_error = obs_data["type_data_error"]

                residuals = [type_data[j] - model_at_combined_data[residual_start + j] for j in range(len(type_data))]
                residual_start += len(type_data)
                ax_residual.errorbar(
                    redshift, residuals, yerr=type_data_error,
                    fmt="o", color=color_schemes[(i + 1) % len(color_schemes)],
                    capsize=3, markersize=3, zorder=2
                )

            ax_residual.axhline(0, color="black", linestyle="--", linewidth=2, zorder=3)
            ax_residual.set_xlabel("Redshift")
            ax_residual.set_ylabel("Residual: $Model - Data$ ($Mpc$)")
            ax_residual.grid()

            plt.tight_layout()
            plt.subplots_adjust(hspace=0)
            plt.savefig(f"{folder_path}/{model_name}_Combined.png", dpi=300)
            plt.close()