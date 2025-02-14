import matplotlib.pyplot as plt
from getdist import plots, MCSamples
from Kosmulator.Config import generate_label, save_stats_to_file, save_interpretations_to_file, save_latex_table_to_file, print_stats_table
import User_defined_modules as UDM                      # Custom module with user-defined functions for cosmological calculations
from scipy.interpolate import interp1d
from Plots.Plot_functions import *
from Kosmulator import Statistic_packages as SP
import numpy as np
import shutil
import sys
import os
import re
import pandas as pd

def generate_plots(All_Samples, CONFIG, PLOT_SETTINGS, data, true_model):
    """
    Generate corner plots, best-fit plots, statistical analysis, and save results to organized folders.
    """
    # Initialize outputs
    All_best_fit_values = {}
    All_LaTeX_Tables = {}
    reference_chi_squared = None  # To store the reduced chi-squared value of the true model

    # Dictionaries to store statistical results and interpretations
    stats_dict = {}
    interpretations_dict = {}

    # Create a main folder for saving tables
    main_folder = "saved_tables"
    os.makedirs(main_folder, exist_ok=True)

    # 1. Generate corner plots for all models
    for model_name, Samples in All_Samples.items():
        print(f"\nCreating corner plot for model: {model_name}...")
        structured_values, aligned_table, parameter_labels, observation_names = make_CornerPlot(
            Samples, CONFIG[model_name], model_name, model_name, PLOT_SETTINGS
        )
        All_best_fit_values[model_name] = structured_values
        All_LaTeX_Tables[model_name] = (aligned_table, parameter_labels, observation_names)

    # 2. Generate best-fit plots
    print("\nCreating best-fit plots...")
    best_fit_plots(All_best_fit_values, CONFIG, data, PLOT_SETTINGS["color_schemes"])

    # 3. Perform statistical analysis
    print("\nPerforming statistical analysis...")
    statistical_results = SP.statistical_analysis(All_best_fit_values, data, CONFIG, true_model)

    # 4. Process statistical results and interpretations
    for model, obs_results in statistical_results.items():
        stats_dict[model] = []  # Initialize statistical results list for the model
        interpretations_dict[model] = []  # Initialize interpretations list for the model

        for obs, stats in obs_results.items():
            # Interpret diagnostics
            diagnostics = SP.provide_model_diagnostics(
                reduced_chi_squared=stats["Reduced_Chi_squared"],
                model_name=model,
                reference_chi_squared=reference_chi_squared if model != true_model else None,
            )
            delta_aic_bic_feedback = SP.interpret_delta_aic_bic(stats["dAIC"], stats["dBIC"])

            # Store statistical results
            stats_row = {
                "Observation": obs,
                "Log-Likelihood": stats["Log-Likelihood"],
                "Chi_squared": stats["Chi_squared"],
                "Reduced_Chi_squared": stats["Reduced_Chi_squared"],
                "AIC": stats["AIC"],
                "BIC": stats["BIC"],
                "dAIC": stats["dAIC"],
                "dBIC": stats["dBIC"],
            }
            stats_dict[model].append(stats_row)

            # Store interpretations
            interpretations_row = {
                "Observation": obs,
                "Reduced Chi2 Diagnostics": diagnostics.strip(),
                "AIC Interpretation": delta_aic_bic_feedback.splitlines()[0].strip(),
                "BIC Interpretation": delta_aic_bic_feedback.splitlines()[1].strip(),
            }
            interpretations_dict[model].append(interpretations_row)

    # 5. Save results to files
    for model in stats_dict.keys():
        # Create a folder for the model
        model_folder = os.path.join(main_folder, model)
        os.makedirs(model_folder, exist_ok=True)

        # Save statistical results
        save_stats_to_file(model, model_folder, stats_dict[model])

        # Save interpretations
        save_interpretations_to_file(model, model_folder, interpretations_dict[model])

        # Save LaTeX tables
        save_latex_table_to_file(model, model_folder, All_LaTeX_Tables[model])

    # 6. Print final statistical tables to the console
    print("\nFinal Statistical Tables:\n")
    for model, stats_list in stats_dict.items():
        print_stats_table(model, stats_list)

    return All_best_fit_values, All_LaTeX_Tables, statistical_results

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
    
    # Extract plot settings
    line_styles = PLOT_SETTINGS.get("line_styles", ["-", "--", ":", "-."])
    line_widths = [1.2, 1.5]
    color_schemes = PLOT_SETTINGS.get("color_schemes", ['r', 'b', 'green', 'cyan', 'yellow'])

    # Initialize GetDist plotter
    g = plots.get_subplot_plotter(subplot_size=4, subplot_size_ratio=0.8)
    g.settings.figure_legend_frame = True
    g.settings.tight_layout = True
    g.settings.alpha_filled_add = 0.4
    g.settings.solid_colors = list(reversed(color_schemes))
    
    # Determine the number of parameters from the first sample
    if Samples:
        first_sample = next(iter(Samples.values()))
        num_params = first_sample.shape[1]
    else:
        raise ValueError("No samples found to determine the number of parameters.")
    
    # Dynamically calculate font size
    base_size = PLOT_SETTINGS.get("label_font_size", 12)
    max_size = 20
    font_size = base_size + ((num_params - 2) / (6 - 2)) * (max_size - base_size)
    font_size = max(font_size, base_size)
    g.settings.title_limit_fontsize = font_size
    g.settings.legend_fontsize = font_size
    g.settings.fontsize = font_size

    # Prepare distributions and labels
    distributions, labels = [], []
    for i, (obs, sample) in enumerate(Samples.items()):
        obs_list = next((obs_entry for obs_entry in CONFIG["observations"] if "+".join(obs_entry) == obs), None)
        if obs_list is None:
            raise ValueError(f"Observation '{obs}' not found in CONFIG['observations'].")

        param_index = CONFIG["observations"].index(obs_list)
        parameter_names = CONFIG["parameters"][param_index]
        parameter_labels = (
            greek_Symbols(parameter_names) if PLOT_SETTINGS.get("latex_enabled", False) else parameter_names
        )

        if not sample.size:
            raise ValueError(f"Samples for '{obs}' are empty.")

        distribution = MCSamples(samples=sample, names=parameter_names, labels=parameter_labels)
        distribution.plotColor = color_schemes[i % len(color_schemes)]
        distributions.append(distribution)
        labels.append(obs)

    formatted_labels = [
        obs.replace("PantheonP", "Pantheon+").replace("f_sigma_8", r"$f_{\sigma_8}$")
        for obs in labels
    ]
    
    results, latex_table, structured_values = SP.calculate_asymmetric_from_samples(
        Samples, CONFIG["parameters"], CONFIG["observations"]
    )
    
    aligned_latex_table = align_table_to_parameters(latex_table, CONFIG["parameters"])

    line_args = [
        {
            "ls": line_styles[i % len(line_styles)],
            "lw": line_widths[i % len(line_widths)],
            "color": color_schemes[i % len(color_schemes)]
        }
        for i in range(len(distributions))
    ]

    num_cols = (len(labels) + 1) // 2
    g.triangle_plot(
        distributions,
        filled=True,
        legend_labels=formatted_labels,
        legend_loc="upper right",
        legend_ncol=num_cols,
        line_args=line_args,
    )

    formatted_columns = format_for_latex(greek_Symbols(CONFIG["parameters"][0]))

    if PLOT_SETTINGS.get("Table", False):
        add_corner_table(
            g,
            aligned_latex_table,
            formatted_labels,
            PLOT_SETTINGS,
            formatted_columns,
            CONFIG["parameters"][0],
            len(CONFIG["parameters"][0]),
        )

    plt.savefig(f"{folder_path}/{save_file_name}.png", dpi=PLOT_SETTINGS.get("dpi", 300))
    plt.close()

    # Return the aligned LaTeX table, parameter labels, and observation names
    return structured_values, aligned_latex_table, CONFIG["parameters"][0], ["+".join(obs) for obs in CONFIG["observations"]]

def best_fit_plots(All_best_fit_values, CONFIG, data, color_schemes):
    """
    Generate best-fit plots for each model and observation combination.
    """
    red_start, reset_color = "\033[31m", "\033[0m"  # ANSI codes for warnings

    for model_name, model_best_fit in All_best_fit_values.items():
        observations = CONFIG[model_name]["observations"]
        observation_types = CONFIG[model_name]["observation_types"]

        for obs_index, obs_set in enumerate(observations):
            obs_name = "_".join(obs_set)
            folder_path = f"./Plots/Best_fits/{model_name}/{obs_name}/"
            obs_types = observation_types[obs_index]

            # Skip certain plots
            if len(set(obs_types)) > 1 and not (set(obs_types) == {'OHD', 'CC'} or set(obs_types) == {'CC', 'OHD'}):
                print(f"{red_start}Skipping{reset_color} combination plot for {obs_set} due to mixed observation types (excluding OHD+CC).")
                continue
            if "BAO" in obs_types:
                print(f"{red_start}Skipping{reset_color} the best-fit plot for {obs_set} data.")
                continue

            setup_folder(folder_path)

            # Extract best-fit values and prepare data
            combined_best_fit = model_best_fit["+".join(obs_set)]
            params_median, params_upper, params_lower = fetch_best_fit_values(combined_best_fit)
            combined_redshift, combined_data, combined_error = prepare_data(obs_set, data, params_median)

            redshiftx = np.linspace(0.005, max(combined_redshift) + 0.05, 10000)
            obs_type = observation_types[obs_index][0]

            # Compute models for median, upper, and lower bounds
            model_median, y_label = compute_model(model_name, redshiftx, params_median, obs_type=obs_type)
            model_upper, _ = compute_model(model_name, redshiftx, params_upper, obs_type=obs_type)
            model_lower, _ = compute_model(model_name, redshiftx, params_lower, obs_type=obs_type)

            # Create the plot
            fig, (ax_main, ax_residual) = plt.subplots(
                nrows=2, ncols=1, figsize=(8, 8),
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0}, sharex=True
            )

            # Plot the best-fit model and uncertainty
            ax_main.plot(
                redshiftx, model_median, label=f"{model_name} (Combined: {'+'.join(obs_set)})",
                color=color_schemes[0], linestyle="-", linewidth=2, zorder=3
            )
            ax_main.fill_between(
                redshiftx, model_lower, model_upper,
                color="k", alpha=0.5, label="Model Uncertainty", zorder=2
            )

            # Plot observational data
            for i, obs in enumerate(obs_set):
                obs_data = data[obs]
                redshift, type_data, type_data_error = extract_observation_data(obs, obs_data, params_median)
                ax_main.errorbar(
                    redshift, type_data, yerr=type_data_error,
                    fmt="o", label=obs, color=color_schemes[(i + 1) % len(color_schemes)],
                    capsize=3, markersize=3, zorder=2
                )

            ax_main.set_ylabel(y_label)
            ax_main.set_xlim(0, max(combined_redshift) + 0.05)
            ax_main.legend(loc="lower right")
            ax_main.grid(False)

            # Compute and plot residuals
            interpolator = interp1d(redshiftx, model_median, kind="linear", bounds_error=False, fill_value="extrapolate")
            model_at_combined_data = interpolator(combined_redshift)

            residual_start = 0
            for i, obs in enumerate(obs_set):
                obs_data = data[obs]
                redshift, type_data, type_data_error = extract_observation_data(obs, obs_data, params_median)

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

            # Save the plot
            plt.tight_layout()
            plt.subplots_adjust(hspace=0)
            plt.savefig(f"{folder_path}/{model_name}_Combined.png", dpi=300)
            plt.close()











