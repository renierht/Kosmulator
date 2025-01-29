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
    return Covariance_matrix(model, type_data, type_data_error) if Type == "CC" else np.sum(((type_data - model) ** 2) / (type_data_error ** 2))
    
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
    
def Calc_BAO_chi(covd1, Model_func, param_dict, Type):
    """
    Calculate chi-squared for BAO data using a covariance matrix.
    """
    zz12 = [
        UDM.dvrd(0.295, Model_func, param_dict, Type) - 7.925129270,
        UDM.dmrd(0.510, Model_func, param_dict, Type) - 13.62003080,
        UDM.dhrd(0.510, Model_func, param_dict, Type) - 20.98334647,
        UDM.dmrd(0.706, Model_func, param_dict, Type) - 16.84645313,
        UDM.dhrd(0.706, Model_func, param_dict, Type) - 20.07872919,
        UDM.dmrd(0.930, Model_func, param_dict, Type) - 21.70841761,
        UDM.dhrd(0.930, Model_func, param_dict, Type) - 17.87612922,
        UDM.dmrd(1.317, Model_func, param_dict, Type) - 27.78720817,
        UDM.dhrd(1.317, Model_func, param_dict, Type) - 13.82372285,
        UDM.dvrd(1.491, Model_func, param_dict, Type) - 26.07217182,
        UDM.dmrd(2.330, Model_func, param_dict, Type) - 39.70838281,
        UDM.dhrd(2.330, Model_func, param_dict, Type) - 8.522565830
    ]
    return np.dot(zz12, np.dot(np.linalg.inv(covd1), zz12))
    
def Covariance_matrix(model, type_data, type_data_error):
    """
    Compute chi-squared using a full covariance matrix approach.
    """
    Cov_matrix = np.diag(type_data_error ** 2)
    delta_H = type_data - model
    return np.dot(delta_H, np.dot(np.linalg.inv(Cov_matrix), delta_H))
    
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

def calculate_asymmetric_from_samples(samples, parameters):
    """
    Calculate median, upper, and lower uncertainties from MCMC samples.
    """
    results, latex_table, structured_values = {}, [], {}
    
    for obs, obs_samples in samples.items():
        results[obs], structured_values[obs], row = {}, {}, []
        
        for i, param in enumerate(parameters):
            param_samples = obs_samples[:, i]
            percentiles = np.percentile(param_samples, [16, 50, 84])
            median, lower_error, upper_error = percentiles[1], percentiles[1] - percentiles[0], percentiles[2] - percentiles[1]
            results[obs][param] = {"median": round(median, 3), "lower_error": round(lower_error, 3), "upper_error": round(upper_error, 3)}
            row.append(f"${median:.3f}^{{+{upper_error:.3f}}}_{{-{lower_error:.3f}}}$")
            structured_values[obs][param] = [round(median, 3), round(median + upper_error, 3), round(median - lower_error, 3)]
        latex_table.append(row)
    
    return results, latex_table, structured_values