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
import pandas as pd


def generate_plots(All_Samples, CONFIG, PLOT_SETTINGS, data, true_model):
    """
    Generate corner plots, best-fit plots, statistical analysis, and save results to organized folders.
    """
    All_best_fit_values = {}
    All_LaTeX_Tables = {}
    reference_chi_squared = None  # To store the reduced chi-squared value of the true model

    # To store all statistical results and interpretations
    stats_dict = {}
    interpretations_dict = {}

    # Create a main folder for saving tables
    main_folder = "saved_tables"
    os.makedirs(main_folder, exist_ok=True)

    for model_name, Samples in All_Samples.items():
        print(f"\nCreating corner plot for model {model_name}...")
        structured_values, aligned_table, parameter_labels, observation_names = make_CornerPlot(
            Samples, CONFIG[model_name], model_name, model_name, PLOT_SETTINGS
        )
        All_best_fit_values[model_name] = structured_values
        All_LaTeX_Tables[model_name] = (aligned_table, parameter_labels, observation_names)

    print("\nCreating best-fit plots...\n")
    best_fit_plots(All_best_fit_values, CONFIG, data, PLOT_SETTINGS["color_schemes"])
    
    print("\nPerforming statistical analysis...\n")
    # Pass `true_model` to the statistical analysis function
    statistical_results = statistical_analysis(All_best_fit_values, data, CONFIG, true_model)

    # Store results in stats_dict and interpretations_dict
    for model, obs_results in statistical_results.items():
        stats_dict[model] = []  # Create an empty list for statistical results
        interpretations_dict[model] = []  # Create an empty list for interpretations

        for obs, stats in obs_results.items():
            # Interpret diagnostics and comparisons
            diagnostics = provide_model_diagnostics(
                reduced_chi_squared=stats["Reduced_Chi_squared"],
                model_name=model,
                reference_chi_squared=reference_chi_squared if model != true_model else None,
            )
            delta_aic_bic_feedback = interpret_delta_aic_bic(stats["dAIC"], stats["dBIC"])

            # Save statistical results
            stats_row = {
                "Observation": obs,
                "Log-Likelihood": stats["Log-Likelihood"],
                "Chi_squared": stats["Chi_squared"],
                "Reduced_Chi_squared": stats["Reduced_Chi_squared"],
                "AIC": stats["AIC"],
                "BIC": stats["BIC"],
                "dAIC": stats["dAIC"],
                "dBIC": stats["dBIC"]
            }
            stats_dict[model].append(stats_row)

            # Save interpretations
            interpretations_row = {
                "Observation": obs,
                "Reduced Chi2 Diagnostics": diagnostics.strip(),
                "AIC Interpretation": delta_aic_bic_feedback.splitlines()[0].strip(),
                "BIC Interpretation": delta_aic_bic_feedback.splitlines()[1].strip(),
            }
            interpretations_dict[model].append(interpretations_row)

    # Save results in organized folders
    for model in stats_dict.keys():
        # Create a folder for the model inside the main folder
        model_folder = os.path.join(main_folder, model)
        os.makedirs(model_folder, exist_ok=True)

        # Save statistical results in improved table format
        with open(os.path.join(model_folder, "stats_summary.txt"), "w") as f:
            f.write(f"Statistical Results for Model: {model}\n\n")
            f.write("Observation            | Log-Likelihood | Chi-Squared | Reduced Chi-Squared | AIC     | BIC     | dAIC   | dBIC\n")
            f.write("-" * 120 + "\n")  # Horizontal line for clarity

            # Iterate through observations and stats
            for stats in stats_dict[model]:
                f.write(
                    f"{stats['Observation']:<22} | "
                    f"{stats['Log-Likelihood']:<15.4f} | "
                    f"{stats['Chi_squared']:<12.4f} | "
                    f"{stats['Reduced_Chi_squared']:<20.4f} | "
                    f"{stats['AIC']:<8.3f} | {stats['BIC']:<8.3f} | "
                    f"{stats['dAIC']:<8.3f} | {stats['dBIC']:<8.3f}\n"
                )

        # Save interpretations in an improved table format
        with open(os.path.join(model_folder, "interpretations_summary.txt"), "w") as f:
            f.write(f"Interpretations for Model: {model}\n\n")
            f.write("Observation            | Reduced Chi2 Diagnostics                              | AIC Interpretation               | BIC Interpretation\n")
            f.write("-" * 140 + "\n")  # Horizontal line for clarity

            # Iterate through observations and interpretations
            for interpretation in interpretations_dict[model]:
                f.write(
                    f"{interpretation['Observation']:<22} | "
                    f"{interpretation['Reduced Chi2 Diagnostics']:<50} | "
                    f"{interpretation['AIC Interpretation']:<30} | "
                    f"{interpretation['BIC Interpretation']:<30}\n"
                )

        # Save LaTeX tables as .txt files
        aligned_table, parameter_labels, observation_names = All_LaTeX_Tables[model]
        with open(os.path.join(model_folder, "aligned_table.txt"), "w") as f:
            f.write(f"Aligned Table for Model: {model}\n")
            f.write("Observation            | " + " | ".join(parameter_labels) + "\n")
            f.write("-" * (20 + 25 * len(parameter_labels)) + "\n")
            for obs, row in zip(observation_names, aligned_table):
                f.write(obs + " | " + " | ".join(row) + "\n")

    # Print the stats table for each model
    print("\nFinal Statistical Tables:\n")
    for model, stats_list in stats_dict.items():
        print(f"Statistical Results for Model: {model}")
        print("Observation            | Log-Likelihood | Chi-Squared | Reduced Chi-Squared | AIC     | BIC     | dAIC   | dBIC")
        print("-" * 120)
        for stats in stats_list:
            print(
                f"{stats['Observation']:<22} | "
                f"{stats['Log-Likelihood']:<15.4f} | "
                f"{stats['Chi_squared']:<12.4f} | "
                f"{stats['Reduced_Chi_squared']:<20.4f} | "
                f"{stats['AIC']:<8.3f} | {stats['BIC']:<8.3f} | "
                f"{stats['dAIC']:<8.3f} | {stats['dBIC']:<8.3f}"
            )
        print("\n")

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

def statistical_analysis(best_fit_values, data, CONFIG, true_model):
    """
    Perform statistical analysis for all models and observation combinations,
    and calculate delta AIC/BIC relative to the true model.

    Args:
        best_fit_values (dict): Best-fit parameter values for each model and observation.
        data (dict): Observational data.
        CONFIG (dict): Configuration dictionary.
        true_model (str): Name of the true model to use as the reference.

    Returns:
        dict: Statistical analysis results (Log-Likelihood, Chi-squared, Reduced Chi-squared, AIC, BIC, dAIC, dBIC).
    """
    results = {}
    reference_aic = {}
    reference_bic = {}

    for model_name, observations in best_fit_values.items():
        results[model_name] = {}
        for obs_name, params in observations.items():
            # Extract best-fit parameter values (median values)
            param_dict = {param: values[0] for param, values in params.items()}
            num_params = len(param_dict)  # Number of free parameters for this model

            # Handle combined datasets (e.g., 'BAO+f_sigma_8')
            obs_list = obs_name.split("+")
            chi_squared_total = 0
            num_data_points_total = 0

            for obs in obs_list:
                # Extract data for the individual observation
                obs_data = data.get(obs)
                if not obs_data:
                    raise ValueError(f"Observation data for {obs} not found.")

                # Determine model function
                MODEL_func = UDM.Get_model_function(model_name)

                # Dynamically find `obs_type` from CONFIG
                try:
                    obs_index = CONFIG[model_name]["observations"].index([obs])
                    obs_type = CONFIG[model_name]["observation_types"][obs_index][0]
                except ValueError:
                    raise ValueError(f"Observation {obs} not found in CONFIG for model {model_name}.")

                # Handle JLA and Pantheon datasets
                if obs_type == "SNe" and obs != "PantheonP":
                    redshift = obs_data["redshift"]
                    type_data = obs_data["type_data"]
                    type_data_error = obs_data["type_data_error"]

                    # Compute comoving distances and distance modulus
                    comoving_distances = UDM.Comoving_distance_vectorized(MODEL_func, redshift, param_dict, "SNe")
                    model = 25 + 5 * np.log10(comoving_distances * (1 + redshift))

                    # Calculate chi-squared for these datasets
                    chi_squared = SP.Calc_chi(obs_type, type_data, type_data_error, model)
                    num_data_points_total += len(type_data)

                # Handle PantheonP dataset
                elif obs == "PantheonP":
                    zHD = obs_data["zHD"]
                    m_b_corr = obs_data["m_b_corr"]
                    IS_CALIBRATOR = obs_data["IS_CALIBRATOR"]
                    CEPH_DIST = obs_data["CEPH_DIST"]
                    cov = obs_data["cov"]

                    # Compute comoving distances and distance modulus
                    comoving_distances = UDM.Comoving_distance_vectorized(MODEL_func, zHD, param_dict, "SNe")
                    distance_modulus = 25 + 5 * np.log10(comoving_distances * (1 + zHD))

                    # Use PantheonP chi calculation
                    chi_squared = SP.Calc_PantP_chi(m_b_corr, IS_CALIBRATOR, CEPH_DIST, cov, distance_modulus, param_dict)
                    num_data_points_total += len(m_b_corr)

                # Handle BAO datasets
                elif obs_type == "BAO":
                    chi_squared = SP.Calc_BAO_chi(obs_data, MODEL_func, param_dict, "BAO")
                    num_data_points_total += len(obs_data["covd1"])  # Number of points from the covariance matrix

                # Handle growth rate and f_sigma_8 datasets
                elif obs_type in ["f", "f_sigma_8"]:
                    redshift = obs_data["redshift"]
                    type_data = obs_data["type_data"]
                    type_data_error = obs_data["type_data_error"]

                    Omega_zeta = UDM.matter_density_z(redshift, MODEL_func, param_dict, obs_type)

                    if obs_type == "f":
                        model = Omega_zeta ** param_dict["gamma"]
                    elif obs_type == "f_sigma_8":
                        integral_term = UDM.integral_term(redshift, MODEL_func, param_dict, obs_type)
                        model = param_dict["sigma_8"] * Omega_zeta ** param_dict["gamma"] * np.exp(-1 * integral_term)

                    chi_squared = SP.Calc_chi(obs_type, type_data, type_data_error, model)
                    num_data_points_total += len(type_data)

                # Handle OHD and CC datasets
                elif obs_type in ["OHD", "CC"]:
                    redshift = obs_data["redshift"]
                    type_data = obs_data["type_data"]
                    type_data_error = obs_data["type_data_error"]

                    model = param_dict["H_0"] * np.array([MODEL_func(z, param_dict, obs_type) for z in redshift])
                    chi_squared = SP.Calc_chi(obs_type, type_data, type_data_error, model)
                    num_data_points_total += len(type_data)

                else:
                    raise ValueError(f"Unsupported observation type: {obs_type}")

                # Update chi-squared total
                chi_squared_total += chi_squared

            # Compute log-likelihood
            log_likelihood = -0.5 * chi_squared_total

            # Calculate degrees of freedom
            dof = num_data_points_total - num_params
            if dof <= 0:
                raise ValueError(f"Degrees of freedom (DOF) is zero or negative. Check your model or dataset.")

            # Calculate reduced chi-squared
            reduced_chi_squared = chi_squared_total / dof

            # Calculate AIC
            aic = 2 * num_params - 2 * log_likelihood

            # Calculate BIC
            bic = num_params * np.log(num_data_points_total) - 2 * log_likelihood

            # Store results
            results[model_name][obs_name] = {
                "Log-Likelihood": log_likelihood,
                "Chi_squared": chi_squared_total,
                "Reduced_Chi_squared": reduced_chi_squared,
                "AIC": aic,
                "BIC": bic,
            }

            # Save AIC/BIC for the true model
            if model_name == true_model:
                reference_aic[obs_name] = aic
                reference_bic[obs_name] = bic

    # Calculate delta AIC/BIC for all models relative to the true model
    for model_name, obs_results in results.items():
        for obs_name, stats in obs_results.items():
            stats["dAIC"] = stats["AIC"] - reference_aic.get(obs_name, stats["AIC"])
            stats["dBIC"] = stats["BIC"] - reference_bic.get(obs_name, stats["BIC"])

    return results


def provide_model_diagnostics(reduced_chi_squared, model_name="", reference_chi_squared=None):
    """
    Provide a quick diagnostic description of the model's performance.

    Args:
        reduced_chi_squared (float): Reduced chi-squared value.
        model_name (str): Name of the model (e.g., "LCDM") for context.
        reference_chi_squared (float, optional): Reduced chi-squared value of a reference model (e.g., LCDM).

    Returns:
        str: Diagnostic feedback for the user.
    """
    feedback = ""

    # Statistical Approach
    feedback += "Statistical Interpretation:\n"
    if 0.9 <= reduced_chi_squared <= 1.1:
        feedback += "  - The model appears to fit the data very well. The reduced chi-squared is close to 1, " \
                    "indicating the residuals are consistent with the uncertainties.\n"
    elif 0.5 <= reduced_chi_squared < 0.9:
        feedback += "  - The reduced chi-squared is slightly below 1. This could indicate overfitting, " \
                    "or that the data uncertainties may be overestimated.\n"
    elif reduced_chi_squared < 0.5:
        feedback += "  - The reduced chi-squared is significantly below 1. This suggests possible overfitting or overly conservative error bars.\n"
    elif 1.1 < reduced_chi_squared <= 3.0:
        feedback += "  - The reduced chi-squared is above 1, but within an acceptable range. This indicates a reasonable fit, though there " \
                    "might be room for improvement in the model or data uncertainties.\n"
    else:
        feedback += "  - The reduced chi-squared is significantly above 3. This suggests the model does not fit the data well. " \
                    "Consider revising your model or checking for systematic errors in the data.\n"

    # Benchmark Approach: Only applies to non-LCDM models
    if reference_chi_squared is not None and model_name.lower() != "lcdm":
        feedback += "\nBenchmark Comparison (Relative to LCDM):\n"
        if reduced_chi_squared < reference_chi_squared:
            feedback += f"  - This model's reduced chi-squared ({reduced_chi_squared:.2f}) is lower than the benchmark LCDM value ({reference_chi_squared:.2f}).\n"
            feedback += "    This could indicate overfitting or that uncertainties are playing a significant role.\n"
        elif reduced_chi_squared > reference_chi_squared:
            feedback += f"  - This model's reduced chi-squared ({reduced_chi_squared:.2f}) is higher than the benchmark LCDM value ({reference_chi_squared:.2f}).\n"
            feedback += "    This may suggest underfitting or that the model does not capture the data as well as LCDM.\n"
        else:
            feedback += f"  - This model's reduced chi-squared matches the benchmark LCDM value ({reference_chi_squared:.2f}), suggesting a comparable fit.\n"

    # Special case for LCDM
    if model_name.lower() == "lcdm":
        feedback += "\nThe LCDM model is widely regarded as a robust and well-tested benchmark model. It is recommended when comparing to other models to compare their reduced chi-squared values to the LCDM model's to determine whether over- or under-fitting happened irregardless of the uncertainties in the observations themselves.\n"

    return feedback

def interpret_delta_aic_bic(delta_aic, delta_bic):
    """
    Interpret delta AIC and BIC values using Jeffreys scale.

    Args:
        delta_aic (float): Delta AIC relative to the true model.
        delta_bic (float): Delta BIC relative to the true model.

    Returns:
        str: Interpretation of the delta AIC and BIC values.
    """
    feedback = ""

    # Interpret delta AIC
    if delta_aic < 0:
        feedback += f"  - Delta AIC: The model has a better fit than the reference model (Delta AIC = {delta_aic:.2f}).\n"
    elif delta_aic <= 2:
        feedback += "  - Delta AIC: The models are equally plausible.\n"
    elif 4 <= delta_aic < 7:
        feedback += "  - Delta AIC: Moderate evidence against this model.\n"
    elif delta_aic >= 10:
        feedback += "  - Delta AIC: Strong evidence against this model.\n"

    # Interpret delta BIC
    if delta_bic < 0:
        feedback += f"  - Delta BIC: The model is strongly preferred over the reference model (Delta BIC = {delta_bic:.2f}).\n"
    elif delta_bic <= 2:
        feedback += "  - Delta BIC: The models are equally plausible.\n"
    elif 4 <= delta_bic < 7:
        feedback += "  - Delta BIC: Moderate evidence against this model.\n"
    elif delta_bic >= 10:
        feedback += "  - Delta BIC: Strong evidence against this model.\n"

    return feedback









