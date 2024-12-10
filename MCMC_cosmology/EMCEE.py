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
import User_defined_modules as UDM

def load_mcmc_results(output_path, file_name="tutorial.h5",CONFIG=None):
    chain_path = os.path.join(output_path, file_name)
    reader = emcee.backends.HDFBackend(chain_path)
    samples = reader.get_chain(discard=CONFIG['burn'],flat=True)
    print (np.shape(samples))
    return samples
 
def model_likelihood(theta, data, Type, CONFIG, MODEL_func, obs):
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
    
    if obs == "PantheonP":
        #print("Matched PantheonP")
        redshift = data['zHD']
        mb = data['m_b_corr']
        trig = data['IS_CALIBRATOR']
        cepheid = data['CEPH_DIST']
        cov = data['cov']
    elif obs == "BAO":
        covd1 = data['covd1']
    else:
        #print(f"Matched {obs} but not PantheonP")
        redshift = data["redshift"]
        type_data = data["type_data"]
        type_data_error = data["type_data_error"]
        
    param_dict = {param: value for param, value in zip(CONFIG["parameters"], theta)}
    if obs != 'BAO':
        model = np.zeros(len(redshift))
        
    if Type=="SNe":
        y_dl = np.zeros(len(redshift))
        for i in range(0,len(redshift)):
            #int_val = integrate.quad(MODEL_func,0,redshift[i],args=(param_dict,Type))            #intergrating over our model
            #y_dl[i] = (300000/param_dict['H_0'])*(1+redshift[i])*int_val[0]                          #calculating the luminosity distance
            y_dl[i] = UDM.comoving_distance(MODEL_func, redshift[i], param_dict, Type)*(1+redshift[i])
            model[i] = 25+ 5*np.log10(y_dl[i])
    elif Type=="OHD" or Type=="CC":
        for i, z in enumerate(redshift):
             model[i] = param_dict["H_0"]*MODEL_func(z,param_dict,Type) 
    elif Type=="fsigma8" or Type=="sigma8":   
        E_value = UDM.nonLinear_Hubble_parameter(redshift,param_dict,Type)
        Omega_zeta = UDM.matter_density_z(redshift,param_dict,Type)
        if Type=="fsigma8":
            model  = param_dict['sigma_8'] * (Omega_zeta / E_value**2) ** param_dict['gamma']
        else:
            model = (Omega_zeta / E_value**2) ** param_dict['gamma'] 
    elif Type=="BAO":
        pass
    else:
        print(f"Unknown Type: {Type}. Unable to compute model.")
        return -np.inf  # Return invalid log-likelihood for unknown type

    if model is None and obs !='BAO':
        raise ValueError("Model computation failed. Model is None.")
    
    if obs == "PantheonP":
        chi = SP.Calc_PantP_chi(mb, trig, cepheid, cov, model)
    if obs == "BAO":
        chi = SP.Calc_BAO_chi(covd1, MODEL_func, param_dict, Type, rd)
    else:
        chi = SP.Calc_chi(Type, type_data, type_data_error, model)
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
    
def lnprob(theta, data, Type, CONFIG, MODEL_func, obs):
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
    return lp + model_likelihood(theta, data, Type, CONFIG, MODEL_func, obs)

def run_mcmc(data, model_name="LCDM", chain_path =None, MODEL_func=None, parallel=True, saveChains =False, 
             overwrite=False, autoCorr = True, CONFIG=None, obs=None):
    """
    Runs an MCMC sampler using the emcee library with optional parallelization, 
    chain saving, and autocorrelation-based convergence checking.

    Args:
        data (array-like): The observational data to be used in the MCMC analysis (redshift, data, data_err).

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
    Type = CONFIG['observation_types']
    # Ensure that CONFIG and MODEL are provided; these are essential for the MCMC setup
    if CONFIG is None or MODEL_func is None:
        raise ValueError("CONFIG and Model must be provided from the main script")
    
    ## Path to the MCMC chain file to check if the chain file already exists and whether to overwrite it
    #output_dir = f"MCMC_Chains/{model_name}/{CONFIG['observations'][0]}"
    #file_name = f"{CONFIG['observations'][0]}.h5"
    #chain_path = os.path.join(output_dir, file_name)
    #if not overwrite and os.path.exists(chain_path):
        #print(f"Loading existing MCMC chain from {chain_path}")
        #samples = load_mcmc_results(output_path=output_dir, file_name=file_name, CONFIG=CONFIG)
        #return samples  # Exit early if not overwriting and chain is already available
    
    
    # Setup of theta and bound for an arbitrary amount of parameters and run Scip.optimize.minimize to find a good
    # starting point for the MCMC simulation.
    bnds = [(CONFIG["prior_limits"][param][0], CONFIG["prior_limits"][param][1]) for param in CONFIG["parameters"]]
    theta = [CONFIG["true_values"][i] for i, param in enumerate(CONFIG["parameters"])]
    nll = lambda *args: -model_likelihood(*args)
    result = op.minimize(nll, theta, args=(data, Type, CONFIG, MODEL_func, obs),bounds =bnds)
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
                    CONFIG["nwalker"], CONFIG["ndim"], lnprob, args=(data, Type, CONFIG, MODEL_func, obs),backend=backend, pool=pool
            )
            else:
                sampler = emcee.EnsembleSampler(
                    CONFIG["nwalker"], CONFIG["ndim"], lnprob, args=(data, Type, CONFIG, MODEL_func, obs), pool=pool
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
                CONFIG["nwalker"], CONFIG["ndim"],  lnprob, args=(data, Type, CONFIG, MODEL_func, obs), backend=backend
            )
        else:
            sampler = emcee.EnsembleSampler(
                CONFIG["nwalker"], CONFIG["ndim"],  lnprob, args=(data, Type, CONFIG, MODEL_func, obs), backend=backend
            ) 
                
        if autoCorr: # AutoCorrelation to speed up calculation. Stops the MCMC when convergence occured
            SP.AutoCorr(pos, iterations=CONFIG['nsteps'], sampler=sampler)
        else:
            sampler.run_mcmc(pos, CONFIG["nsteps"], progress=True)          
    samples = sampler.chain[:, CONFIG["burn"] :, :].reshape((-1, CONFIG["ndim"]))
    #print (np.shape(samples))
    return samples


