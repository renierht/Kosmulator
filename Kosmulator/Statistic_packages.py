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

def Covariance_matrix_old(model, type_data, type_data_error):
    """
    Compute chi-squared using a full covariance matrix approach.
    """
    print (f"Covariance matrix - type_data: {type_data}, type_data_error: {type_data_error}, model: {model}")
    Cov_matrix = np.diag(type_data_error ** 2)
    delta_H = type_data - model
    print (f"Cov_matrix: {Cov_matrix}, delta_H: {delta_H}")
    result = np.dot(delta_H, np.dot(np.linalg.inv(Cov_matrix), delta_H))
    print (f"Cov_matrix result: {result}")
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






