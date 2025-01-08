import numpy as np
from Plots.Plots import autocorrPlot                 # Custom module for creating plots (e.g., autocorrelation plot)
import User_defined_modules as UDM     # Custom module with user-defined functions for cosmological calculations
import scipy.linalg as la 

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
    if Type == "CC":
        # Use covariance matrix-based calculation for CC data
        chi = Covariance_matrix(model, type_data, type_data_error)
    else:
        # Standard chi-squared calculation for other data types
        chi = np.sum(((type_data - model) ** 2) / (type_data_error ** 2))
    return chi
    
def Calc_PantP_chi(mb, trig, cepheid, cov, model, param_dict):
    """
    Calculate chi-squared for Pantheon+ data using a covariance matrix.

    Args:
        mb (array-like): Corrected apparent magnitudes.
        trig (array-like): Boolean array indicating if a point is a calibrator.
        cepheid (array-like): Cepheid distance moduli for calibrator points.
        cov (ndarray): Lower triangular Cholesky factor of the covariance matrix for `mb`.
        model (array-like): Model distance moduli predictions.

    Returns:
        float: Chi-squared value.
    """
    # Check if 'M_abs' is in param_dict and is not None
    if 'M_abs' in param_dict and param_dict['M_abs'] is not None:
        M = param_dict['M_abs']  # Use M as a free parameter
    else:
        M = -19.20  # Default absolute magnitude for SNe standard candles  

    meub = mb - M                                   # Distance modulus from apparent magnitude
    moduli = np.where(trig == 1, cepheid, model)    # Use calibrators or model predictions
    delta = meub - moduli                           # Residuals
    
    # Solve for residuals using the Cholesky decomposition of the covariance matrix
    residuals = la.solve_triangular(cov, delta, lower = True, check_finite = False)
    chi = (residuals ** 2).sum()                    # Sum of squared residuals
    return chi
    
def Calc_BAO_chi(covd1, Model_func, param_dict, Type):
    """
    Calculate chi-squared for BAO data using a covariance matrix.

    Args:
        covd1 (ndarray)        : Covariance matrix for BAO data.
        Model_func (callable)  : Function that computes model predictions for given parameters.
        param_dict (dict)      : Dictionary of model parameters.
        Type (str)             : Model type (e.g., "LCDM").
        rd (float)             : Sound horizon at the drag epoch.

    Returns:
        float                  : Chi-squared value.
    """
    zz12 = [UDM.dvrd(0.295, Model_func, param_dict, Type) - 7.925129270, UDM.dmrd(0.510, Model_func, param_dict, Type) - 13.62003080, 
            UDM.dhrd(0.510, Model_func, param_dict, Type) - 20.98334647, UDM.dmrd(0.706, Model_func, param_dict, Type) - 16.84645313, 
            UDM.dhrd(0.706, Model_func, param_dict, Type) - 20.07872919, UDM.dmrd(0.930, Model_func, param_dict, Type) - 21.70841761,
            UDM.dhrd(0.930, Model_func, param_dict, Type) - 17.87612922, UDM.dmrd(1.317, Model_func, param_dict, Type) - 27.78720817, 
            UDM.dhrd(1.317, Model_func, param_dict, Type) - 13.82372285, UDM.dvrd(1.491, Model_func, param_dict, Type) - 26.07217182, 
            UDM.dmrd(2.330, Model_func, param_dict, Type) - 39.70838281, UDM.dhrd(2.330, Model_func, param_dict, Type) - 8.522565830]

    covinvd1 = np.linalg.inv(covd1)            # Inverse of the covariance matrix
    chi = np.dot(zz12, np.dot(covinvd1, zz12)) # Compute chi-squared using matrix operations
    return chi
    
def Covariance_matrix(model, type_data, type_data_error):
    """
    Compute chi-squared using a full covariance matrix approach.

    Args:
        model (array-like)           : Model predictions.
        type_data (array-like)       : Observational data values.
        type_data_error (array-like) : Associated errors for the data.

    Returns:
        float                        : Chi-squared value.
    """

    Cov_matrix = np.diag(type_data_error ** 2)      # Construct diagonal covariance matrix
    C_inv = np.linalg.inv(Cov_matrix)               # Invert the covariance matrix
    delta_H = type_data - model                     # Residuals
    chi = np.dot(delta_H, np.dot(C_inv, delta_H))   # Compute chi-squared
    return chi
    
def AutoCorr(pos, iterations, sampler,  model_name, color, obs, convergence = 0.01):
    """
    Compute and monitor the autocorrelation time during MCMC sampling.

    Args:
        pos (array-like)    : Initial positions of MCMC walkers.
        iterations (int)    : Number of iterations to sample.
        sampler (object)    : EMCEE sampler object.

    Returns:
        None                : Updates plots and checks for convergence.
    """
    index = 0
    autocorr = np.empty(iterations)
    old_tau = np.inf
    
    for sample in sampler.sample(pos, iterations = iterations, progress = True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
        
        # Estimate autocorrelation time
        tau = sampler.get_autocorr_time(tol = 0)
        autocorr[index] = np.mean(tau)
        index += 1

         # Check for convergence
        converged  = np.all(tau * 100 < sampler.iteration) 
        converged &= np.all(np.abs(old_tau - tau) / tau < convergence)
        if converged: 
            break # Stop sampling if convergence criteria are met
        
        old_tau = tau
        autocorrPlot(autocorr, index,  model_name, color, obs)   