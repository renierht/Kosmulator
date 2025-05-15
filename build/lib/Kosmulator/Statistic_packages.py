import numpy as np
import numba as nb
import scipy.linalg as la 
from Plots.Plots import autocorrPlot  # Custom module for autocorrelation plotting
import User_defined_modules as UDM   # Custom user-defined cosmology functions
import numexpr as ne
##############################Robert added packages#############################
from classy import Class
import os
from Kosmulator import Class_run as CR
import Kosmulator
import User_defined_modules as UDM
from clik.lkl import clik as PlanckLikelihood
#from Kosmulator.MCMC_setup import global_model_name
################################################################################

##############################Robert added likelihood###########################
def Calc_cmbtt_chi(param_dict,l, Dl, Dl_minus, Dl_plus,model_name):
    model_dir = os.path.join("./Class", model_name)
    like = PlanckLikelihood("./Observations/camspec_10.7HM_1400_TT_small.clik")
    
    l_theory = np.arange(2, 2501)
    cl = UDM.LCDM_MODEL(0, param_dict, Type="CMB")  # 'Type' doesn't matter for Cls
    
    #Dl_theory = cl['tt'][2:2501] * l_theory * (l_theory + 1) * 1e12

    lmax = 2500
    cl_array = np.zeros((2, lmax + 1))
    cl_array[0, 2:lmax+1] = cl['tt'][2:lmax+1]
    cl_array[1, 2:] = 0
    cl_array[2, 2:] = 0
    cl_array[3, 2:] = 0 

                               
    chi2 = like(cl_array)
    return chi2

def Calc_cmbee_chi(param_dict,l, Dl, Dl_minus, Dl_plus,model_name):
    model_dir = os.path.join("./Class", model_name)
    
    l_theory = np.arange(2, 1997)
    cl = UDM.LCDM_MODEL(0, param_dict, Type="CMB")  # 'Type' doesn't matter for Cls
    
    Dl_theory = cl['ee'][2:1997] * l_theory * (l_theory + 1) * 1e12

    chi2 = np.sum(((Dl - Dl_theory)**2) / (Dl_minus**2 + Dl_plus**2))
    return chi2

def Calc_cmbte_chi(param_dict,l, Dl, Dl_minus, Dl_plus,model_name):
    model_dir = os.path.join("./Class", model_name)
    
    l_theory = np.arange(2, 1997)
    cl = UDM.LCDM_MODEL(0, param_dict, Type="CMB")  # 'Type' doesn't matter for Cls
    
    Dl_theory = cl['te'][2:1997] * l_theory * (l_theory + 1) * 1e12

    chi2 = np.sum(((Dl - Dl_theory)**2) / (Dl_minus**2 + Dl_plus**2))
    return chi2
################################################################################

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
    Calculate chi-squared for Pantheon+ data using a covariance matrix,
    with numexpr used to speed up the elementwise arithmetic.

    Args:
        mb: Array of corrected apparent magnitudes.
        trig: Indicator array (e.g., 1 for calibrators).
        cepheid: Cepheid distances for calibrators.
        cov: Precomputed Cholesky-decomposed covariance matrix.
        model: Model predictions (for non-calibrators).
        param_dict: Dictionary containing parameters (e.g. 'M_abs').

    Returns:
        The chi-square value as a float.
    """
    # Retrieve the absolute magnitude.
    M = param_dict.get('M_abs', -19.20)
    # Compute moduli: if trig equals 1, use cepheid distances; otherwise, use the model.
    moduli = np.where(trig == 1, cepheid, model)
    # Compute delta = (mb - M) - moduli using numexpr for fast elementwise arithmetic.
    delta = ne.evaluate("mb - M - moduli")
    # Solve the triangular system quickly, disabling finite checks.
    residuals = la.solve_triangular(cov, delta, lower=True, check_finite=False)
    # Return the sum of squared residuals.
    return np.sum(residuals**2)

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

def Calc_DESI_chi(data, Model_func, param_dict, Type):
    """
    Compute the DESI log-likelihood for various data types, following MontePython's implementation.
    
    data: dict with keys 'redshift', 'measurement', 'measurement_error', 'type', and 'inv_cov' (also available as 'cov')
    Model_func: cosmological model function (e.g., LCDM_MODEL)
    param_dict: dictionary of cosmological parameters (must include "H_0" and "r_d")
    Type: not used here (for interface consistency)
    """
    # Extract arrays
    z = data["redshift"]
    meas = data["measurement"]
    types = data["type"]
    inv_cov = data["inv_cov"]
    #print(f"z: {z}, meas: {meas}, types: {types}, inv_cov: {inv_cov}")

    
    # Compute comoving distances using the "SNe" branch (which integrates 1/E(z))
    comoving = UDM.Comoving_distance_vectorized(Model_func, z, param_dict, "DESI")
    # Compute Hubble parameter at each redshift using the "OHD" branch and multiply by H_0:
    H_vals = param_dict["H_0"] * np.array([Model_func(zi, param_dict, "OHD") for zi in z])
    #comoving = UDM.Comoving_distance_vectorized(Model_func, z, param_dict, Type)
    #H_vals = np.array([Model_func(zi, param_dict, Type) for zi in z])
    DA = comoving / (1 + z)
    dr = z / H_vals
    dv = (DA**2 * (1 + z)**2 * dr)**(1. / 3.)
    
    print (param_dict)

    param_dict2 = {'Omega_m': 0.29, 'H_0': 68, 'r_d': 150.0}
    comoving2 = UDM.Comoving_distance_vectorized(Model_func, z, param_dict2, "DESI")
    # Compute Hubble parameter at each redshift using the "OHD" branch and multiply by H_0:
    H_vals2 = param_dict2["H_0"] * np.array([Model_func(zi, param_dict2, "OHD") for zi in z])
    #comoving = UDM.Comoving_distance_vectorized(Model_func, z, param_dict, Type)
    #H_vals = np.array([Model_func(zi, param_dict, Type) for zi in z])
    DA2 = comoving2 
    DM2 = DA2*(1+z)
    dr2 = z / H_vals2
    dv2 = (DA2**2 * (1 + z)**2 * dr2)**(1. / 3.)
    print (f"comoving: {comoving2}, H_vals2: {H_vals}, DA2: {DA2}, DM2: {DM2}, dr2: {dr2}, dv2: {dv2}")
    

    rs = param_dict["r_d"]

    # Compute theoretical prediction for each data point based on type:
    theo = np.zeros_like(z)
    for i in range(len(z)):
        if types[i] == 3:
            theo[i] = dv[i] / rs
        elif types[i] == 4:
            theo[i] = dv[i]
        elif types[i] == 5:
            theo[i] = DA[i] / rs
        elif types[i] == 6:
            theo[i] = 1.0 / H_vals[i] / rs
        elif types[i] == 7:
            theo[i] = rs / dv[i]
        elif types[i] == 8:
            theo[i] = DA[i] * (1 + z[i]) / rs
        else:
            raise ValueError(f"DESI data type {types[i]} not understood.")

    # Compute residuals (the difference between theoretical prediction and measurement)
    diff = theo - meas

    # Compute chi-squared using the inverse covariance matrix:
    chi2 = np.dot(np.dot(diff, inv_cov), diff)

    # Return the log-likelihood (as in MontePython)
    return -0.5 * chi2

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
                elif obs == "DESI":
                    chi_squared = Calc_DESI_chi(obs_data, MODEL_func, param_dict, "DESI")
                    num_data_points_total += len(obs_data["cov"])
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
                ###################################Robert added code####################
                elif obs == "CMB_TT" or obs == "CMB_EE" or obs == "CMB_TE":
                    # Handle CMB data (TT, EE, TE)
                    l = obs_data["l"]  # The multipoles (l)
                    Dl = obs_data["Dl"]  # The power spectrum values (Cl in mK^2)
                    Dl_minus = obs_data["Dl_minus"]  # The error bars for Cl
                    Dl_plus = obs_data["Dl_plus"]  # The error bars for Cl

                    # Now calculate the chi-squared for CMB data
                    chi_squared = 0
                    if obs == "CMB_TT":
                        chi_squared = Calc_cmbtt_chi(param_dict, l, Dl, Dl_minus, Dl_plus, model_name)
                    elif obs == "CMB_EE":
                        chi_squared = Calc_cmbee_chi(param_dict, l, Dl, Dl_minus, Dl_plus, model_name)
                    elif obs == "CMB_TE":
                        chi_squared = Calc_cmbte_chi(param_dict, l, Dl, Dl_minus, Dl_plus, model_name)
                    
                    chi_squared_total += chi_squared
                    num_data_points_total += len(Dl)
                #######################################################################
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
        feedback += f"Delta AIC: Model strongly preferred (ΔAIC = {delta_aic:.2f}).\n"
    elif delta_aic <= 2:
        feedback += "Delta AIC: Models equally plausible.\n"
    elif 4 <= delta_aic < 7:
        feedback += "Delta AIC: Moderate evidence against the model.\n"
    elif delta_aic >= 10:
        feedback += "Delta AIC: Strong evidence against the model.\n"
    else:
        feedback += f"Delta AIC: Weak evidence against the model (ΔAIC = {delta_aic:.2f}).\n"

    # Interpret delta BIC
    if delta_bic < 0:
        feedback += f"Delta BIC: Model strongly preferred (ΔBIC = {delta_bic:.2f}).\n"
    elif delta_bic <= 2:
        feedback += "Delta BIC: Models equally plausible.\n"
    elif 4 <= delta_bic < 7:
        feedback += "Delta BIC: Moderate evidence against the model.\n"
    elif delta_bic >= 10:
        feedback += "Delta BIC: Strong evidence against the model.\n"
    else:
        feedback += f"Delta BIC: Weak evidence against the model (ΔBIC = {delta_bic:.2f}).\n"

    return feedback


