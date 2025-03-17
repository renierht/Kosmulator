import numpy as np
import scipy.linalg as la 
from Plots.Plots import autocorrPlot  # Custom module for autocorrelation plotting
import User_defined_modules as UDM   # Custom user-defined cosmology functions

def Calc_chi(Type, type_data, type_data_error, model):
    """
    Calculate the chi-squared value for a given dataset and model.

    Args:
        Type (str): Type of observation (e.g., "CC" for chronometers).
        type_data (array-like): Observational data values.
        type_data_error (array-like): Associated errors for the data.
        model (array-like): Model predictions.

    Returns:
        float: Chi-squared value.
    """
    result = Covariance_matrix(model, type_data, type_data_error) if Type == "CC" else np.sum(((type_data - model) ** 2) / (type_data_error ** 2))
    return result


def Calc_PantP_chi(mb, trig, cepheid, cov, model, param_dict):
    """
    Calculate chi-squared for Pantheon+ data using a covariance matrix.
    """
    M = param_dict.get('M_abs', -19.20)  # Default absolute magnitude
    meub = mb - M  # Distance modulus
    moduli = np.where(trig == 1, cepheid, model)
    delta = meub - moduli
    residuals = la.solve_triangular(cov, delta, lower=True, check_finite=False)
    return (residuals ** 2).sum()

def Calc_BAO_chi(data, Model_func, param_dict, Type):
    """
    Compute the BAO chi-square value based on the input data.

    Args:
        data (dict): BAO observational data, including covariance matrix `covd1`.
        Model_func (callable): Cosmological model function.
        param_dict (dict): Cosmological parameters.
        Type (str): Type of observation.

    Returns:
        float: Chi-square value for the BAO dataset.
    """
    covd1 = data["covd1"]
    redshifts = [0.295, 0.510, 0.510, 0.706, 0.706, 0.930, 0.930, 1.317, 1.317, 1.491, 2.330, 2.330]

    # Compute BAO distances for all redshifts
    dmrd_vals = UDM.dmrd(redshifts, Model_func, param_dict, Type)
    dhrd_vals = UDM.dhrd(redshifts, Model_func, param_dict, Type)
    dvrd_vals = UDM.dvrd([0.295, 1.491], Model_func, param_dict, Type)

    # Assemble residual vector zz12
    zz12 = [
        dvrd_vals[0] - 7.925129270,
        dmrd_vals[1] - 13.62003080,
        dhrd_vals[1] - 20.98334647,
        dmrd_vals[3] - 16.84645313,
        dhrd_vals[3] - 20.07872919,
        dmrd_vals[5] - 21.70841761,
        dhrd_vals[5] - 17.87612922,
        dmrd_vals[7] - 27.78720817,
        dhrd_vals[7] - 13.82372285,
        dvrd_vals[1] - 26.07217182,
        dmrd_vals[10] - 39.70838281,
        dhrd_vals[10] - 8.522565830,
    ]

    # Compute chi-square
    return np.dot(zz12, np.dot(np.linalg.inv(covd1), zz12))

def Covariance_matrix(model, type_data, type_data_error):
    """
    Compute chi-squared using a full covariance matrix approach.

    Optimized for diagonal covariance matrices.
    """
    
    # Calculate delta_H (residuals)
    delta_H = type_data - model
    
    # Diagonal covariance matrix and its inverse
    Cov_diag = type_data_error ** 2
    Cov_inv_diag = 1.0 / Cov_diag  # Inverse of diagonal elements
    
    # Efficient chi-squared calculation
    result = np.sum(delta_H ** 2 * Cov_inv_diag)  # Avoid explicit matrix inversion
    
    return result
  
def AutoCorr(pos, iterations, sampler, model_name, color, obs, PLOT_SETTINGS, convergence=0.01, last_obs=False):
    """
    Compute and monitor the autocorrelation time during MCMC sampling.
    """
    index, autocorr, old_tau = 0, np.empty(iterations), np.inf
    
    for sample in sampler.sample(pos, iterations=iterations, progress=True):
        if sampler.iteration % 100:
            continue
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1
        converged = np.all(tau * 100 < sampler.iteration) and np.all(np.abs(old_tau - tau) / tau < convergence)
        n = 100 * np.arange(1, index + 1)
        if converged: 
            if last_obs:
                autocorrPlot(autocorr, index,  model_name, color, obs, PLOT_SETTINGS, close_plot = True, nsteps = iterations)
            break # Stop sampling if convergence criteria are met
        elif  n[-1]== iterations:
             autocorrPlot(autocorr, index,  model_name, color, obs, PLOT_SETTINGS, close_plot = False, nsteps = iterations)
             if  last_obs:
                 autocorrPlot(autocorr, index,  model_name, color, obs, PLOT_SETTINGS, close_plot = True, nsteps = iterations)
             break
        else:
             old_tau = tau
             autocorrPlot(autocorr, index,  model_name, color, obs, PLOT_SETTINGS, close_plot = False, nsteps = iterations)   

def calculate_asymmetric_from_samples(samples, parameters, observations):
    """
    Calculate median, upper, and lower uncertainties from MCMC samples.
    """
    results, latex_table, structured_values = {}, [], {}

    for obs, obs_samples in samples.items():
        results[obs], structured_values[obs], row = {}, {}, []

        # Match the observation key to its corresponding parameter list
        obs_list = next(
            (obs_entry for obs_entry in observations if "+".join(obs_entry) == obs),
            None,
        )
        if obs_list is None:
            raise ValueError(f"Observation '{obs}' not found in observations.")

        # Fetch the corresponding parameter list
        param_index = observations.index(obs_list)
        obs_param_names = parameters[param_index]

        # Iterate over the parameters for this observation
        for param_index, param in enumerate(obs_param_names):
            # Check that the parameter index is within bounds of obs_samples
            if param_index < obs_samples.shape[1]:
                param_samples = obs_samples[:, param_index]

                # Calculate percentiles
                percentiles = np.percentile(param_samples, [16, 50, 84])
                median, lower_error, upper_error = percentiles[1], percentiles[1] - percentiles[0], percentiles[2] - percentiles[1]

                # Add to results
                results[obs][param] = {
                    "median": round(median, 3),
                    "lower_error": round(lower_error, 3),
                    "upper_error": round(upper_error, 3)
                }

                # Add LaTeX-formatted string for the table
                row.append(f"${median:.3f}^{{+{upper_error:.3f}}}_{{-{lower_error:.3f}}}$")

                # Add structured values for the parameter
                structured_values[obs][param] = [
                    round(median, 3),
                    round(median + upper_error, 3),
                    round(median - lower_error, 3)
                ]
            else:
                print(f"Warning: Parameter '{param}' index ({param_index}) exceeds sample dimensions.")

        # Add the row to the LaTeX table
        latex_table.append(row)

    return results, latex_table, structured_values

def statistical_analysis(best_fit_values, data, CONFIG, true_model):
    """
    Perform statistical analysis for all models and observation combinations,
    and calculate delta AIC/BIC relative to the true model.
    
    This updated version processes each observation individually by pairing it with
    its corresponding observation type from CONFIG.
    """
    results = {}
    reference_aic = {}
    reference_bic = {}

    for model_name, obs_results in best_fit_values.items():
        results[model_name] = {}
        for obs_name, params in obs_results.items():
            # Extract best-fit (median) values into a dictionary.
            param_dict = {param: values[0] for param, values in params.items()}
            num_params = len(param_dict)

            # Recover the full observation list that corresponds to this best-fit key.
            obs_entry = next(
                (obs_list for obs_list in CONFIG[model_name]["observations"] if "+".join(obs_list) == obs_name),
                None
            )
            if obs_entry is None:
                raise ValueError(f"Observation {obs_name} not found in CONFIG for model {model_name}.")
            obs_index = CONFIG[model_name]["observations"].index(obs_entry)

            chi_squared_total = 0
            num_data_points_total = 0

            # Get the model function once for this model.
            MODEL_func = UDM.Get_model_function(model_name)
            # Get the list of observation types for this observation set.
            obs_types = CONFIG[model_name]["observation_types"][obs_index]

            # Loop over each individual observation in the set, using its corresponding type.
            for i, obs in enumerate(obs_entry):
                obs_type = obs_types[i]
                obs_data = data.get(obs)
                if not obs_data:
                    raise ValueError(f"Observation data for {obs} not found.")

                if obs == "PantheonP":
                    zHD = obs_data["zHD"]
                    m_b_corr = obs_data["m_b_corr"]
                    IS_CALIBRATOR = obs_data["IS_CALIBRATOR"]
                    CEPH_DIST = obs_data["CEPH_DIST"]
                    cov = obs_data["cov"]
                    # For PantheonP, use the "SNe" branch.
                    comoving_distances = UDM.Comoving_distance_vectorized(MODEL_func, zHD, param_dict, "SNe")
                    distance_modulus = 25 + 5 * np.log10(comoving_distances * (1 + zHD))
                    chi_squared = Calc_PantP_chi(m_b_corr, IS_CALIBRATOR, CEPH_DIST, cov, distance_modulus, param_dict)
                    num_data_points_total += len(m_b_corr)
                elif obs == "BAO":
                    chi_squared = Calc_BAO_chi(obs_data, MODEL_func, param_dict, "BAO")
                    num_data_points_total += len(obs_data["covd1"])
                elif obs_type == "SNe":
                    redshift = obs_data["redshift"]
                    type_data = obs_data["type_data"]
                    type_data_error = obs_data["type_data_error"]
                    comoving_distances = UDM.Comoving_distance_vectorized(MODEL_func, redshift, param_dict, "SNe")
                    model_val = 25 + 5 * np.log10(comoving_distances * (1 + redshift))
                    chi_squared = Calc_chi(obs_type, type_data, type_data_error, model_val)
                    num_data_points_total += len(type_data)
                elif obs_type in ["OHD", "CC"]:
                    redshift = obs_data["redshift"]
                    type_data = obs_data["type_data"]
                    type_data_error = obs_data["type_data_error"]
                    model_val = param_dict["H_0"] * np.array([MODEL_func(z, param_dict, obs_type) for z in redshift])
                    chi_squared = Calc_chi(obs_type, type_data, type_data_error, model_val)
                    num_data_points_total += len(type_data)
                elif obs_type in ["f", "f_sigma_8"]:
                    redshift = obs_data["redshift"]
                    type_data = obs_data["type_data"]
                    type_data_error = obs_data["type_data_error"]
                    Omega_zeta = UDM.matter_density_z(redshift, MODEL_func, param_dict, obs_type)
                    if obs_type == "f_sigma_8":
                        integral_term = UDM.integral_term(redshift, MODEL_func, param_dict, obs_type)
                        model_val = param_dict["sigma_8"] * Omega_zeta ** param_dict["gamma"] * np.exp(-1 * integral_term)
                    else:
                        model_val = Omega_zeta ** param_dict["gamma"]
                    chi_squared = Calc_chi(obs_type, type_data, type_data_error, model_val)
                    num_data_points_total += len(type_data)
                else:
                    raise ValueError(f"Unsupported observation type: {obs_type}")
                
                chi_squared_total += chi_squared

            log_likelihood = -0.5 * chi_squared_total
            dof = num_data_points_total - num_params
            if dof <= 0:
                raise ValueError("Degrees of freedom (DOF) is zero or negative. Check your model or dataset.")
            reduced_chi_squared = chi_squared_total / dof
            aic = 2 * num_params - 2 * log_likelihood
            bic = num_params * np.log(num_data_points_total) - 2 * log_likelihood

            results[model_name][obs_name] = {
                "Log-Likelihood": log_likelihood,
                "Chi_squared": chi_squared_total,
                "Reduced_Chi_squared": reduced_chi_squared,
                "AIC": aic,
                "BIC": bic,
            }
            if model_name == true_model:
                reference_aic[obs_name] = aic
                reference_bic[obs_name] = bic
    
    # Calculate delta AIC and delta BIC relative to the true model.
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




