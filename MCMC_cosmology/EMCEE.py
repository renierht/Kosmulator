import numpy as np
import emcee
#from scipy import integrate
from multiprocessing import Pool
import time
import os
import matplotlib.pyplot as plt
import h5py
import Plots as MP
import scipy.optimize as op
from scipy import integrate
from MCMC_cosmology import Statistic_packages as SP

def load_mcmc_results(output_path, file_name="tutorial.h5",CONFIG=None):
    chain_path = os.path.join(output_path, file_name)
    reader = emcee.backends.HDFBackend(chain_path)
    samples = reader.get_chain(discard=CONFIG['burn'],flat=True)
    print (np.shape(samples))
    return samples
 
def model_likelihood(theta, data, Type, CONFIG, MODEL_func):
    """
    Computes the log-likelihood of a cosmological model given parameters, observational data, 
    and a model type (SNe, OHD, or CC).

    Arguments:
    theta (array-like): Current parameter values (e.g., [Omega_m, H0]).
    data (tuple): (redshift, type_data, type_data_error) - observational data and errors.
    Type (str, optional): The category or label for the observation type (e.g., "SNe"). 
    CONFIG (dict): The configuration dictionary containing key parameters for the MCMC setup.
    MODEL_func (function): User-defined model function for integration or direct computation.

    Returns:
    - log_likelihood (float): -0.5 * chi-square value, or -np.inf for invalid cases.

    Behaviour:
    - "SNe": Integrates MODEL_func to compute luminosity distances and magnitudes.
    - "OHD"/"CC": Computes Hubble parameters using MODEL_func.
    - Returns -np.inf if Type is invalid.
    """
    
    model = None  # Default initialization
    if isinstance(Type, list):
        Type = Type[0] 
    
    redshift, type_data, type_data_error = data
    param_dict = {param: value for param, value in zip(CONFIG["parameters"], theta)}

    model = np.zeros(len(redshift))
    if Type=="SNe":
        y_dl = np.zeros(len(redshift))
        for i in range(0,len(redshift)):
            int_val = integrate.quad(MODEL_func,0,redshift[i],args=(param_dict,Type))            #intergrating over our model
            y_dl[i] = (300000/param_dict['H_0'])*(1+redshift[i])*int_val[0]                          #calculating the luminosity distance
            model[i] = 25+ 5*np.log10(y_dl[i])
    elif Type=="OHD" or Type=="CC":
        for i, z in enumerate(redshift):
             model[i] = param_dict["H_0"]*MODEL_func(z,param_dict,Type)  
    else:
        print(f"Unknown Type: {Type}. Unable to compute model.")
        return -np.inf  # Return invalid log-likelihood for unknown type

    if model is None:
        raise ValueError("Model computation failed. Model is None.")
    
    chi = SP.Calc_chi(Type, type_data, type_data_error,model)
    return -0.5*chi
    
def lnprior(theta, CONFIG):
    """
    Log-prior function to ensure parameters are within defined limits.
    Args:
        theta (array-like): Current parameter values (e.g., [Omega_m, H0]).
        
        CONFIG (dict): The configuration dictionary containing key parameters for the MCMC setup. 

    Returns:
        float: 0.0 if within prior limits, -np.inf otherwise.
    """
    
    for param, value in zip(CONFIG["parameters"], theta):
        lower, upper = CONFIG["prior_limits"][param]
        if not (lower < value < upper):
            return -np.inf  # Parameter out of bounds
    return 0.0  #
    
def lnprob(theta, data, Type, CONFIG, MODEL_func):
    """
    Log-probability function combining prior and likelihood.
    Args:
        theta (array-like): Current parameter values (e.g., [Omega_m, H0]).
        
        data (array-like): The observational data to be used in the MCMC analysis (redshift, data, data_err).
        
        CONFIG (dict): The configuration dictionary containing key parameters for the MCMC setup. 
        
        MODEL_funct (callable): Function to compute the cosmological Friedmann equation.

    Returns:
        float: Log-probability value.
    """
    lp = lnprior(theta, CONFIG)
    if not np.isfinite(lp):
        return -np.inf  # Invalid prior
    return lp + model_likelihood(theta, data, Type, CONFIG, MODEL_func)

def run_mcmc(data, Type="SNe", model_name="LCDM", MODEL_func=None, parallel=True, saveChains =False, 
             overwrite=False, autoCorr = True, CONFIG=None):
    """
    Runs an MCMC sampler using the emcee library with optional parallelization, 
    chain saving, and autocorrelation-based convergence checking.

    Args:
        data (array-like): The observational data to be used in the MCMC analysis (redshift, data, data_err).

        Type (str, optional): The category or label for the observation type (e.g., "SNe"). 
                        Default is "SNe".

        MODEL (callable): The model function that computes the likelihood or other relevant quantities. 
                        N.B Must be provided by the user.

        parallel (bool, optional): Whether to run the MCMC process in parallel using multiple cores. 
                        Default is True. Use False for Windows if parallization fails

        saveChains (bool, optional): Whether to save the MCMC chains to an HDF5 file. 
                        Default is False. N.B. Slows down the code

        model_name (str, optional): The name of the model's directory where MCMC chain will be saved. 
                        Default is "LCDM".
                        
        overwrite (bool, optional): Whether to overwrite existing saved chains. 
                        If False and a saved chain exists, the function will load it 
                        instead of rerunning the MCMC. Default is False.

        autoCorr (bool, optional): Enables the use of autocorrelation checks to assess convergence. 
                        If True, the sampler periodically computes the autocorrelation time and 
                        stops early if the chain has converged. Default is True.

        CONFIG (dict): The configuration dictionary containing key parameters for the MCMC setup. 
                    Required keys include:
                        - "parameters" (list): List of parameter names.
                        - "prior_limits" (dict): Dictionary of prior bounds for parameters.
                        - "true_values" (list): List of initial guesses for parameters.
                        - "nwalker" (int): Number of walkers in the ensemble.
                        - "ndim" (int): Number of dimensions (parameters) in the model.
                        - "nsteps" (int): Total number of MCMC steps.
                        - "burn" (int): Number of burn-in steps to discard.

    Returns:
        samples (array-like): The MCMC samples after processing (e.g., burn-in removal). 
    """
    
    # Ensure that CONFIG and MODEL are provided; these are essential for the MCMC setup
    if CONFIG is None or MODEL_func is None:
        raise ValueError("CONFIG and Model must be provided from the main script")
    
    # Path to the MCMC chain file to check if the chain file already exists and whether to overwrite it
    output_dir = f"MCMC_Chains/{model_name}/{Type[0]}"
    file_name = f"{Type[0]}.h5"
    chain_path = os.path.join(output_dir, file_name)
    if not overwrite and os.path.exists(chain_path):
        print(f"Loading existing MCMC chain from {chain_path}")
        samples = load_mcmc_results(output_path=output_dir, file_name=file_name, CONFIG=CONFIG)
        return samples  # Exit early if not overwriting and chain is already available
    
    # Setup of theta and bound for an arbitrary amount of parameters and run Scip.optimize.minimize to find a good
    # starting point for the MCMC simulation.
    bnds = [(CONFIG["prior_limits"][param][0], CONFIG["prior_limits"][param][1]) for param in CONFIG["parameters"]]
    theta = [CONFIG["true_values"][i] for i, param in enumerate(CONFIG["parameters"])]
    nll = lambda *args: -model_likelihood(*args)
    result = op.minimize(nll, theta, args=(data, Type, CONFIG, MODEL_func),bounds =bnds)
    print ("SciPy's optimized start point for MCMC: ",result['x'])
    pos = [result['x'] + 1e-4 * np.random.randn(CONFIG["ndim"]) for _ in range(CONFIG["nwalker"])]
    
    # Setting up the MCMC ensamble that will be used to run the MCMC simulation. It can be done either in parallel
    # or series. You also have the choice of enabling saving the chains for later usage, as well as if you want to 
    # and Auto Correlation to stop the MCMC simulation when it has converged. 
    if parallel:
        with Pool() as pool:
            if saveChains:
                backend = emcee.backends.HDFBackend(chain_path)
                backend.reset(CONFIG["nwalker"], CONFIG["ndim"])
                sampler = emcee.EnsembleSampler(
                    CONFIG["nwalker"], CONFIG["ndim"], lnprob, args=(data, Type, CONFIG, MODEL_func),backend=backend, pool=pool
            )
            else:
                sampler = emcee.EnsembleSampler(
                    CONFIG["nwalker"], CONFIG["ndim"], lnprob, args=(data, Type, CONFIG, MODEL_func), pool=pool
            )
            start = time.time()
            
            if autoCorr:# AutoCorrelation to speed up calculation. Stops the MCMC when convergence occured
                SP.AutoCorr(pos, iterations=CONFIG['nsteps'], sampler=sampler)
            else:
                sampler.run_mcmc(pos, CONFIG["nsteps"], progress=True)                
            end = time.time()
            print(f"Multiprocessing took {end - start:.1f} seconds")
    else: # Calculating in series
        if saveChains:
            backend = emcee.backends.HDFBackend(chain_path)
            backend.reset(CONFIG["nwalker"], CONFIG["ndim"])
            sampler = emcee.EnsembleSampler(
                CONFIG["nwalker"], CONFIG["ndim"],  lnprob, args=(data, Type, CONFIG, MODEL_func), backend=backend
            )
        else:
            sampler = emcee.EnsembleSampler(
                CONFIG["nwalker"], CONFIG["ndim"],  lnprob, args=(data, Type, CONFIG, MODEL_func), backend=backend
            ) 
                
        if autoCorr: # AutoCorrelation to speed up calculation. Stops the MCMC when convergence occured
            SP.AutoCorr(pos, iterations=CONFIG['nsteps'], sampler=sampler)
        else:
            sampler.run_mcmc(pos, CONFIG["nsteps"], progress=True)          
    samples = sampler.chain[:, CONFIG["burn"] :, :].reshape((-1, CONFIG["ndim"]))
    #print (np.shape(samples))
    return samples


