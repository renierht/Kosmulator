import numpy as np
import emcee
import time
import os
import platform
import matplotlib.pyplot as plt
import h5py
import scipy.optimize as op
from scipy.optimize import fsolve
from Kosmulator import Statistic_packages as SP  # Statistical functions for cosmological calculations
from Kosmulator.Config import format_elapsed_time, load_mcmc_results
import User_defined_modules as UDM  # Custom user-defined cosmology functions
from joblib import Parallel, delayed
            
def model_likelihood(theta, obs_data, obs_type, CONFIG, MODEL_func, obs, obs_index):
    """
    Compute the log-likelihood for a given model and observation.
    """
    # Ensure obs_type is a single string (if needed)
    if isinstance(obs_type, list):
        obs_type = obs_type[0]

    # Extract parameters for the current observation from CONFIG
    required_params = CONFIG["parameters"][obs_index]
    param_dict = {param: value for param, value in zip(required_params, theta)}

    # Special handling for BAO:
    if obs == "BAO":
        chi = SP.Calc_BAO_chi(obs_data, MODEL_func, param_dict, obs_type)
        return -0.5 * chi
        
    # Handle specific cases for observations
    if obs == "PantheonP":
        redshift = obs_data['zHD']
        mb = obs_data['m_b_corr']
        trig = obs_data['IS_CALIBRATOR']
        cepheid = obs_data['CEPH_DIST']
        cov = obs_data['cov']
    else:
        redshift = obs_data["redshift"]
        type_data = obs_data["type_data"]
        type_data_error = obs_data["type_data_error"]

    # Compute the model prediction based on the observation type
    if obs_type == "SNe":
        # Vectorized luminosity distance computation
        comoving_distances = UDM.Comoving_distance_vectorized(MODEL_func, redshift, param_dict, obs_type)
        y_dl = comoving_distances * (1 + redshift)
        model = 25 + 5 * np.log10(y_dl)
    elif obs_type in ["OHD", "CC"]:
        # Vectorized Hubble parameter computation
        model = param_dict["H_0"] * np.array([MODEL_func(z, param_dict, obs_type) for z in redshift])
    elif obs_type in ["f_sigma_8", "f"]:
        # Compute growth rate model
        Omega_zeta = UDM.matter_density_z(redshift, MODEL_func, param_dict, obs_type)
        if obs_type == "f_sigma_8":
            integral_term = UDM.integral_term(redshift, MODEL_func, param_dict, obs_type)
            model = (
                param_dict["sigma_8"]
                * Omega_zeta ** param_dict["gamma"]
                * np.exp(-1 * integral_term)
            )
        else:
            model = Omega_zeta ** param_dict["gamma"]
    else:
        print(f"ERROR: Unknown obs_type: {obs_type}. Unable to compute model.")
        return -np.inf

    # Compute chi-square likelihood
    if obs == "PantheonP":
        chi = SP.Calc_PantP_chi(mb, trig, cepheid, cov, model, param_dict)
    else:
        chi = SP.Calc_chi(obs_type, type_data, type_data_error, model)
    return -0.5 * chi

def lnprior(theta, CONFIG, obs_index=0):
    """
    Compute the log-prior for the given parameters.
    """
    prior_limits = np.array([
        CONFIG["prior_limits"][obs_index][param]
        for param in CONFIG["parameters"][obs_index]
    ])
    if np.any((theta <= prior_limits[:, 0]) | (theta >= prior_limits[:, 1])):
        return -np.inf
    return 0.0

def lnprob(theta, data, Type, CONFIG, MODEL_func, obs, obs_index):
    """
    Compute the log-probability (prior + likelihood) for MCMC sampling.
    Args:
        theta (array): Parameter values.
        data (dict): Observational data.
        Type (list): List of observation types.
        CONFIG (dict): Configuration dictionary.
        MODEL_func (callable): Cosmological model function.
        obs (list): List of observation names.
        obs_index (int): Index of the observation in CONFIG.

    Returns:
        float: Log-probability (prior + likelihood).
    """
    # Ensure obs and Type are lists
    obs = [obs] if isinstance(obs, str) else obs
    Type = [Type] if isinstance(Type, str) else Type

    # Compute the log-prior
    lp = lnprior(theta, CONFIG, obs_index)
    if not np.isfinite(lp):  # Skip likelihood if prior is invalid
        return -np.inf

    # Calculate total likelihood efficiently
    total_likelihood = sum(
        model_likelihood(theta, data[obs_name], obs_type, CONFIG, MODEL_func, obs_name, obs_index)
        for obs_name, obs_type in zip(obs, Type)
    )

    # Combine prior and likelihood
    return lp + total_likelihood

def run_mcmc(data, model_name="LCDM", chain_path=None, MODEL_func=None, parallel=True, saveChains=False, 
    overwrite=False, autoCorr=True, CONFIG=None, obs=None, Type=None, colors='r', convergence=0.01, last_obs=False
    , PLOT_SETTINGS=None, obs_index=0):
    """
    Run MCMC sampler using `emcee` with parallelization, saving, and autocorrelation-based convergence.
    """
    # Determine multiprocessing settings
    if parallel:
        if platform.system() == "Darwin":  # macOS
            print ("OS:                       Mac-based system")
            from multiprocess import Pool, get_context
        elif platform.system() == "Linux":
            print ("OS:                       Linux-based system")
            from multiprocessing import Pool, get_context
        elif platform.system() == "Windows":
            print ("OS:                       Windows-based system")
            print ('Note: Windows crashes with the multiprocessing tool. Switching to serial calculation')
            parallel = False  # Disable parallelization for Windows
        else:
            raise ImportError("Unsupported operating system for parallization in this script.")
            
    # Check inputs
    if CONFIG is None or MODEL_func is None:
        raise ValueError("CONFIG and MODEL_func must be provided.")
    
    # Optimize starting point for MCMC
    # For optimization, use only the first observation and its type.
    if isinstance(obs, list):
        obs_str = obs[0]
    else:
        obs_str = obs

    if isinstance(Type, list):
        obs_type_for_opt = Type[0]
    else:
        obs_type_for_opt = Type
    print("\nFinding optimized initial parameter positions with Scipy...")
    bnds = np.array([
            (CONFIG["prior_limits"][obs_index][param][0], CONFIG["prior_limits"][obs_index][param][1])
            for param in CONFIG["parameters"][obs_index]
    ])
    theta_init = np.array([CONFIG["true_values"][obs_index][i] for i in range(len(CONFIG["parameters"][obs_index]))])
    nll = lambda theta, *args: -model_likelihood(theta, *args)  # Negative log-likelihood

    result = op.minimize(
        nll, theta_init,
        args=(data[obs_str], obs_type_for_opt, CONFIG, MODEL_func, obs_str, obs_index),
        bounds=bnds,
        method='L-BFGS-B'  # Use L-BFGS-B for better handling of bounds
    )
    pos = result['x'] + 1e-4 * np.random.randn(CONFIG["nwalker"], len(CONFIG["parameters"][obs_index]))

    print(f"SciPy's optimized IC: {result['x']}")
    
    # Setting up the MCMC ensamble that will be used to run the MCMC simulation. It can be done either in parallel
    # or series. You also have the choice of enabling saving the chains for later usage, as well as if you want to 
    # and Auto Correlation to stop the MCMC simulation when it has converged. 
    
    print (f"\nRunning the MCMC simulation for the \033[34m{model_name}\033[0m model on these data sets: \033[34m{obs}\033[0m...")
    if parallel:
        with Pool() as pool:
            if saveChains:
                backend = emcee.backends.HDFBackend(chain_path)
                backend.reset(CONFIG["nwalker"], CONFIG["ndim"][obs_index])
                sampler = emcee.EnsembleSampler(
                    CONFIG["nwalker"], CONFIG["ndim"][obs_index], lnprob, args = (data, Type, CONFIG, MODEL_func, obs, obs_index),
                    backend = backend, pool = pool,
                )
            else:
                sampler = emcee.EnsembleSampler(
                    CONFIG["nwalker"],CONFIG["ndim"][obs_index], lnprob, args=(data, Type, CONFIG, MODEL_func, obs, obs_index), 
                    pool = pool,
            )
            
            start = time.time()
            if autoCorr:    # AutoCorrelation to speed up calculation. Stops the MCMC when convergence occured
                SP.AutoCorr(pos, iterations = CONFIG['nsteps'], sampler = sampler, model_name = model_name, 
                                        color = colors, obs = obs, PLOT_SETTINGS = PLOT_SETTINGS, convergence = convergence, last_obs = last_obs)
                                        
            else:
                sampler.run_mcmc(pos, CONFIG["nsteps"], progress = True)                
            end = time.time()
            formatted_time = format_elapsed_time(end-start)
            print(f"Multiprocessing took {formatted_time}\n")
            
    else: # Calculating in series
        if saveChains:
            backend = emcee.backends.HDFBackend(chain_path)
            backend.reset(CONFIG["nwalker"], CONFIG["ndim"][obs_index])
            sampler = emcee.EnsembleSampler(
                CONFIG["nwalker"], CONFIG["ndim"][obs_index],  lnprob, args = (data, Type, CONFIG, MODEL_func, obs, obs_index), backend = backend,
            )
        else:
            sampler = emcee.EnsembleSampler(
                CONFIG["nwalker"], CONFIG["ndim"][obs_index],  lnprob, args = (data, Type, CONFIG, MODEL_func, obs, obs_index),
            ) 
                
        start = time.time()
        if autoCorr: # AutoCorrelation to speed up calculation. Stops the MCMC when convergence occured
            SP.AutoCorr(pos, iterations=CONFIG['nsteps'], sampler = sampler, model_name = model_name, 
                                    color = colors, obs = obs[0], PLOT_SETTINGS = PLOT_SETTINGS, convergence = convergence, last_obs = last_obs)
        else:
            sampler.run_mcmc(pos, CONFIG["nsteps"], progress = True)   
        end = time.time()
        formatted_time = format_elapsed_time(end-start)
        print(f"Series processing took {formatted_time}\n")
        
    samples = sampler.chain[:, CONFIG["burn"] :, :].reshape((-1, CONFIG["ndim"][obs_index]))
    return samples


