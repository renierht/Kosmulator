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
            
def model_likelihood(theta, data, Type, CONFIG, MODEL_func, obs):
    """
    Compute the log-likelihood for given parameters and observational data.
    
    Args:
        theta (array-like): Current parameter values (e.g., [Omega_m, H0]).
        data (dict): Observational data and associated errors.
        Type (str): Observation type (e.g., "SNe", "OHD").
        CONFIG (dict): Configuration dictionary.
        MODEL_func (function): Function to compute the model values.
        obs (str): Specific observation being analyzed.
    
    Returns:
        float: Log-likelihood value (-0.5 * chi-square).
    """
    model = None
    if isinstance(Type, list):
        Type = Type[0]
    
    # Extract data based on observation type
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
    
    if obs != 'BAO':
        model = np.zeros(len(redshift))
    
    if Type == "SNe":
            y_dl = np.zeros(len(redshift))
            for i in range(len(redshift)):
                y_dl[i] = UDM.Comoving_distance(MODEL_func, redshift[i], param_dict, Type) * (1 + redshift[i])
                model[i] = 25 + 5 * np.log10(y_dl[i])
    elif Type in ["OHD", "CC"]:
        model = [param_dict["H_0"] * MODEL_func(z, param_dict, Type) for z in redshift]
    elif Type in ["f_sigma_8", "f"]:
        E_value = MODEL_func(redshift, param_dict, Type)
        Omega_zeta = UDM.matter_density_z(redshift, MODEL_func, param_dict, Type)
        if Type == "f_sigma_8":
            model = param_dict['sigma_8'] * (Omega_zeta / E_value**2) ** param_dict['gamma'] 
        elif "f":
            model = (Omega_zeta / E_value**2) ** param_dict['gamma']
    elif Type == "BAO":
        pass
    else:
        print(f"Unknown Type: {Type}. Unable to compute model.")
        return -np.inf

    # Compute chi-square and likelihood
    if obs == "PantheonP":
        chi = SP.Calc_PantP_chi(mb, trig, cepheid, cov, model, param_dict)
    elif obs == "BAO":
        chi = SP.Calc_BAO_chi(covd1, MODEL_func, param_dict, Type)
    else:
        chi = SP.Calc_chi(Type, type_data, type_data_error, model)

    return -0.5 * chi

def lnprior(theta, CONFIG):
    """
    Compute the log-prior for the given parameters.
    """
    for param, value in zip(CONFIG["parameters"], theta):
        lower, upper = CONFIG["prior_limits"][param]
        if not (lower < value < upper):
            return -np.inf
    return 0.0
    
def lnprob(theta, data, Type, CONFIG, MODEL_func, obs):
    """
    Compute the combined log-prior and log-likelihood.
    """
    lp = lnprior(theta, CONFIG)
    if not np.isfinite(lp):
        return -np.inf
    return lp + sum(model_likelihood(theta, data[obs[i]], Type[i], CONFIG, MODEL_func, obs[i],) for i in range(len(Type)))

def run_mcmc(data, model_name="LCDM", chain_path=None, MODEL_func=None, parallel=True, saveChains=False, 
    overwrite=False, autoCorr=True, CONFIG=None, obs=None, Type=None, colors='r', convergence=0.01, last_obs=False
    , PLOT_SETTINGS=None):
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
    print ("\nFinding optimized initial parameter positions with Scipy...")
    bnds = [(CONFIG["prior_limits"][param][0], CONFIG["prior_limits"][param][1]) for param in CONFIG["parameters"]]
    theta_init = [CONFIG["true_values"][i] for i in range(len(CONFIG["parameters"]))]
    nll = lambda *args: -model_likelihood(*args)
    result = op.minimize(nll, theta_init, args=(data[obs[0]], Type, CONFIG, MODEL_func, obs[0]), bounds=bnds)
    pos = [result['x'] + 1e-4 * np.random.randn(CONFIG["ndim"]) for _ in range(CONFIG["nwalker"])]
    print (f"SciPy's optimized IC:     {result['x']}")
    
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
                                        color = colors, obs = obs, PLOT_SETTINGS = PLOT_SETTINGS, convergence = convergence, last_obs = last_obs)
                                        
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
                CONFIG["nwalker"], CONFIG["ndim"],  lnprob, args = (data, Type, CONFIG, MODEL_func, obs), backend = backend,
            )
        else:
            sampler = emcee.EnsembleSampler(
                CONFIG["nwalker"], CONFIG["ndim"],  lnprob, args = (data, Type, CONFIG, MODEL_func, obs),
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
        
    samples = sampler.chain[:, CONFIG["burn"] :, :].reshape((-1, CONFIG["ndim"]))
    return samples


