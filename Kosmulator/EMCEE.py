import numpy as np
import emcee
import time
import os
import platform
import matplotlib.pyplot as plt
import h5py
import scipy.optimize as op
from Kosmulator import Statistic_packages as SP  # Custom module for statitical functions for cosmological calculations
from Kosmulator.Config import format_elapsed_time
import User_defined_modules as UDM                   # Custom module with user-defined functions for cosmological calculations

# Load previous MCMC results from an HDF5 file
def load_mcmc_results(output_path, file_name = "tutorial.h5", CONFIG = None):
    """
    Load saved MCMC chains from an HDF5 backend.

    Args:
        output_path (str)   : Directory where the MCMC chain file is stored.
        file_name (str)     : Name of the HDF5 file.
        CONFIG (dict)       : Configuration dictionary containing burn-in steps.

    Returns:
        samples (array-like): MCMC samples after burn-in.
    """
    chain_path = os.path.join(output_path, file_name)
    reader = emcee.backends.HDFBackend(chain_path)
    samples = reader.get_chain(discard = CONFIG['burn'], flat = True)
    return samples
 
# Log-likelihood function for the model
def model_likelihood(theta, data, Type, CONFIG, MODEL_func, obs):
    """
    Computes the log-likelihood for the given parameters and observational data.

    Args:
        theta (array-like)    : Current parameter values (e.g., [Omega_m, H0]).
        data (tuple)          : Observational data and associated errors.
        Type (str)            : Observation type (e.g., "SNe", "OHD").
        CONFIG (dict)         : Configuration dictionary for MCMC setup.
        MODEL_func (function) : Function to compute the model values.
        obs (str)             : Specific observation being analyzed.

    Returns:
        float                 : Log-likelihood value (-0.5 * chi-square).
    """
    # Initialize variables
    model = None  
    if isinstance(Type, list):  # Handle cases where Type is provided as a list
        Type = Type[0]
    
    # Extract relevant data based on the observation type
    if obs == "PantheonP":
        redshift = data['zHD']
        mb = data['m_b_corr']
        trig = data['IS_CALIBRATOR']
        cepheid = data['CEPH_DIST']
        cov = data['cov']
    elif obs == "BAO":
        covd1 = data['covd1']
    else:
        redshift = data["redshift"]
        type_data = data["type_data"]
        type_data_error = data["type_data_error"]
    
    # Map parameter values
    param_dict = {param: value for param, value in zip(CONFIG["parameters"], theta)}

    # Compute the model based on the observation type
    if obs != 'BAO':
        model = np.zeros(len(redshift))
        
    if Type == "SNe":
        y_dl = np.zeros(len(redshift))
        for i in range(len(redshift)):
            y_dl[i] = UDM.Comoving_distance(MODEL_func, redshift[i], param_dict, Type) * (1 + redshift[i])
            model[i] = 25 + 5 * np.log10(y_dl[i])
            
    elif Type in ["OHD", "CC"]:
        for i, z in enumerate(redshift):
            model[i] = param_dict["H_0"] * MODEL_func(z, param_dict, Type)
            
    elif Type in ["fsigma8", "sigma8"]:
        E_value = MODEL_func(redshift, param_dict, Type )
        #E_value = UDM.nonLinear_Hubble_parameter(redshift, param_dict, Type)
        Omega_zeta = UDM.matter_density_z(redshift, MODEL_func, param_dict, Type)
        if Type == "fsigma8":
            # Compute the growth factor including sigma_8 as a parameter (fsigma8).
            model = param_dict['sigma_8'] * (Omega_zeta / E_value**2) ** param_dict['gamma']
        else:
            # Compute the growth factor (sigma8)
            model = (Omega_zeta / E_value**2) ** param_dict['gamma']
            
    elif Type == "BAO":
        pass
        
    else:
        print(f"Unknown Type: {Type}. Unable to compute model.")
        return -np.inf

    # Compute likelihoods for the model based on the observation
    if obs == "PantheonP":
        chi = SP.Calc_PantP_chi(mb, trig, cepheid, cov, model, param_dict)
    elif obs == "BAO":
        chi = SP.Calc_BAO_chi(covd1, MODEL_func, param_dict, Type)
    else:
        chi = SP.Calc_chi(Type, type_data, type_data_error, model)

    return -0.5 * chi

# Log-prior function to constrain parameters
def lnprior(theta, CONFIG):
    """
    Compute the log-prior for the given parameters.

    Args:
        theta (array-like) : Current parameter values.
        CONFIG (dict)      : Configuration dictionary containing prior limits.

    Returns:
        float              : 0.0 if within prior bounds, -np.inf otherwise.
    """
    for param, value in zip(CONFIG["parameters"], theta):
        lower, upper = CONFIG["prior_limits"][param]
        if not (lower < value < upper):
            return -np.inf
    return 0.0
    
# Combined log-probability function
def lnprob(theta, data, Type, CONFIG, MODEL_func, obs):
    """
    Combine log-prior and log-likelihood to compute log-probability.

    Args:
        theta (array-like)    : Current parameter values.
        data (array-like)     : Observational data.
        Type (str)            : Observation type.
        CONFIG (dict)         : Configuration dictionary.
        MODEL_func (function) : Model function.

    Returns:
        float                 : Log-probability value.
    """
    lp = lnprior(theta, CONFIG)
    if not np.isfinite(lp):
        return -np.inf
        
    total_likelihood = 0
    for i in range(0,len(Type)):
        total_likelihood += model_likelihood(theta, data[obs[i]], Type[i], CONFIG, MODEL_func, obs[i])
    return lp + total_likelihood

def run_mcmc(data, model_name = "LCDM", chain_path = None, MODEL_func = None, parallel = True, 
             saveChains = False, overwrite = False, autoCorr = True, CONFIG = None, obs = None,  
             Type  = None, colors = 'r', convergence = 0.01):
    """
    Runs an MCMC sampler using the emcee library with optional parallelization, 
    chain saving, and autocorrelation-based convergence checking.

    Args:
        data (dict, array-like)        : The observational data to be used in the MCMC analysis.
                                      
        MODEL (callable)            : The model function that computes the likelihood or other 
                                      relevant quantities. N.B. Must be provided by the user.
                                      
        parallel (bool, optional)   : Whether to run the MCMC process in parallel using multiple cores. 
                                      Default is True. Use False for Windows if parallization fails.
                                      
        saveChains (bool, optional) : Whether to save the MCMC chains to an HDF5 file. 
                                      Default is False. N.B. Slows down the code.
                                      
        model_name (str, optional)  : The name of the model's directory where MCMC chain will be saved. 
                                      Default is "LCDM".
                        
        overwrite (bool, optional)  : Whether to overwrite existing saved chains. 
                                      If False and a saved chain exists, the function will load it 
                                      instead of rerunning the MCMC. Default is False.

        autoCorr (bool, optional)   : Enables the use of autocorrelation checks to assess convergence. 
                                      If True, the sampler periodically computes the autocorrelation time  
                                      and stops early if the chain has converged. Default is True.

        CONFIG (dict)               : The configuration dictionary containing key parameters for the MCMC setup. 
                                      Required keys include:
                                        - "parameters" (list): List of parameter names.
                                        - "prior_limits" (dict): Dictionary of prior bounds for parameters.
                                        - "true_values" (list): List of initial guesses for parameters.
                                        - "nwalker" (int): Number of walkers in the ensemble.
                                        - "ndim" (int): Number of dimensions (parameters) in the model.
                                        - "nsteps" (int): Total number of MCMC steps.
                                        - "burn" (int): Number of burn-in steps to discard.

    Returns:
        samples (array-like)        : The MCMC samples after processing (e.g., burn-in removal). 
    """
    # Importing the correct multiprocessing package based on OS system. Works for Linux and Mac
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
    print ("\nFinding optimized initial parameter positions with Scipy...")
    bnds = [(CONFIG["prior_limits"][param][0], CONFIG["prior_limits"][param][1]) for param in CONFIG["parameters"]]
    theta = [CONFIG["true_values"][i] for i, param in enumerate(CONFIG["parameters"])]
    nll = lambda *args: -model_likelihood(*args)
    result = op.minimize(nll, theta, args=(data[obs[0]], Type, CONFIG, MODEL_func, obs),bounds =bnds)
    print (f"SciPy's optimized IC:     {result['x']}")
    #print ("")
    pos = [result['x'] + 1e-4 * np.random.randn(CONFIG["ndim"]) for _ in range(CONFIG["nwalker"])]
    
    # Setting up the MCMC ensamble that will be used to run the MCMC simulation. It can be done either in parallel
    # or series. You also have the choice of enabling saving the chains for later usage, as well as if you want to 
    # and Auto Correlation to stop the MCMC simulation when it has converged. 
    
    print (f"\nRunning the MCMC simulation for the \033[34m{model_name}\033[0m model on these data sets: \033[34m{obs}\033[0m...")
    if parallel:
        with Pool() as pool:
            if saveChains:
                backend = emcee.backends.HDFBackend(chain_path)
                backend.reset(CONFIG["nwalker"], CONFIG["ndim"])
                sampler = emcee.EnsembleSampler(
                    CONFIG["nwalker"], CONFIG["ndim"], lnprob, args = (data, Type, CONFIG, MODEL_func, obs),
                    backend = backend, pool = pool,
            )
            else:
                sampler = emcee.EnsembleSampler(
                    CONFIG["nwalker"], CONFIG["ndim"], lnprob, args=(data, Type, CONFIG, MODEL_func, obs), 
                    pool = pool,
            )
            
            start = time.time()
            if autoCorr:    # AutoCorrelation to speed up calculation. Stops the MCMC when convergence occured
                SP.AutoCorr(pos, iterations = CONFIG['nsteps'], sampler = sampler, model_name = model_name, 
                                        color = colors, obs = obs, convergence = convergence)
            else:
                sampler.run_mcmc(pos, CONFIG["nsteps"], progress = True)                
            end = time.time()
            formatted_time = format_elapsed_time(end-start)
            print(f"Multiprocessing took {formatted_time}\n")
            
    else: # Calculating in series
        if saveChains:
            backend = emcee.backends.HDFBackend(chain_path)
            backend.reset(CONFIG["nwalker"], CONFIG["ndim"])
            sampler = emcee.EnsembleSampler(
                CONFIG["nwalker"], CONFIG["ndim"],  lnprob, args = (data, Type, CONFIG, MODEL_func, obs), 
                backend = backend,
            )
        else:
            sampler = emcee.EnsembleSampler(
                CONFIG["nwalker"], CONFIG["ndim"],  lnprob, args = (data, Type, CONFIG, MODEL_func, obs),
            ) 
                
        start = time.time()
        if autoCorr: # AutoCorrelation to speed up calculation. Stops the MCMC when convergence occured
            
            SP.AutoCorr(pos, iterations=CONFIG['nsteps'], sampler = sampler, model_name = model_name, 
                                    color = colors, obs = obs[0], convergence = convergence)
        else:
            sampler.run_mcmc(pos, CONFIG["nsteps"], progress = True)   
        end = time.time()
        formatted_time = format_elapsed_time(end-start)
        print(f"Series processing took {formatted_time}\n")
        
    samples = sampler.chain[:, CONFIG["burn"] :, :].reshape((-1, CONFIG["ndim"]))
    return samples


