import numpy as np
from scipy import integrate
from scipy.optimize import fsolve
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root

##############################################################
# Define your cosmological model used for MCMC analysis
##############################################################
def LCDM_MODEL(z, param_dict, Type="SNe"):
    """
    Lambda Cold Dark Matter (LCDM) model.
    Args:
        z (float or np.ndarray): Redshift value(s).
        param_dict (dict): Dictionary containing cosmological parameters.
        Type (str): Type of observation ('SNe', 'OHD', 'CC', f_sigma_8', or 'BAO').

    Returns:
        float or np.ndarray: Theoretical model prediction based on the observation type.
    """
    z = np.atleast_1d(z)  # Ensure z is an array for consistent operations
    model = np.sqrt(param_dict['Omega_m'] * (1 + z)**3 + 1 - param_dict['Omega_m'])
    result = Calculate_return_values(model, Type)
    return result if len(result) > 1 else result[0]  # Return scalar if input was scalar
    
def LCDM_v_MODEL(z, param_dict, Type="SNe"):
    z = np.atleast_1d(z)  # Ensure z is an array for consistent operations
    model = np.sqrt(param_dict['Omega_m'] * (1 + z)**(3-3*param_dict['zeta']) + 1 - param_dict['Omega_m'])
    result = Calculate_return_values(model, Type)
    return result if len(result) > 1 else result[0]  # Return scalar if input was scalar

def f1CDM_MODEL(z, param_dict, Type="SNe"):
    """
    Solve the nonlinear algebraic equation for E(z).
    """
    
    z = np.atleast_1d(z)  # Ensure z is treated as an array
    def hubble_nonlinear(E, z, Omega_m, n):
        return E**2 - (Omega_m * (1 + z)**3 + (1 - Omega_m) * E**(2 * n))

    # Extract model parameters
    Omega_m = param_dict["Omega_m"]
    n = param_dict["n"]

    # Solve for E(z) (vectorized)
    E_solution = np.array([fsolve(hubble_nonlinear, x0=1.0, args=(zi, Omega_m, n))[0] for zi in z])

    # Return processed results based on Type
    result = Calculate_return_values(E_solution, Type)
    return result if len(result) > 1 else result[0]

def f1CDM_v_MODEL(z, param_dict, Type="SNe"):
    z = np.atleast_1d(z)  # Ensure z is treated as an array
    def hubble_nonlinear(E, z, Omega_m, n, zeta):
        return E**2 - (Omega_m * (1 + z)**(3-3*zeta) + (1 - Omega_m) * E**(2 * n))

    Omega_m = param_dict["Omega_m"]
    n = param_dict["n"]
    zeta = param_dict["zeta"]

    # Solve for E(z) (vectorized)
    E_solution = np.array([fsolve(hubble_nonlinear, x0=1.0, args=(zi, Omega_m, n, zeta))[0] for zi in z])

    result = Calculate_return_values(E_solution, Type)
    return result if len(result) > 1 else result[0]
    
def f2CDM_MODEL(z, param_dict, Type="SNe"):
    z = np.atleast_1d(z)  # Ensure z is treated as an array
    def hubble_nonlinear(E, z, Omega_m, p):
        return E**2 - Omega_m * (1 + z)**3 -((1-Omega_m)/(1-(1+p)* np.exp(-p)))*(1-(1+p)*E*np.exp(-p*E))

    Omega_m = param_dict["Omega_m"]
    p = param_dict["p"]

    # Solve for E(z) (vectorized)
    E_solution = np.array([fsolve(hubble_nonlinear, x0=1.0, args=(zi, Omega_m, p))[0] for zi in z])

    result = Calculate_return_values(E_solution, Type)
    return result if len(result) > 1 else result[0]
    
def f2CDM_v_MODEL(z, param_dict, Type="SNe"):
    z = np.atleast_1d(z)  # Ensure z is treated as an array
    def hubble_nonlinear(E, z, Omega_m, p, zeta):
        return E**2 - Omega_m * (1 + z)**(3-3*zeta) -((1-Omega_m)/(1-(1+p)* np.exp(-p)))*(1-(1+p)*E*np.exp(-p*E))

    Omega_m = param_dict["Omega_m"]
    p = param_dict["p"]
    zeta = param_dict["zeta"]

    # Solve for E(z) (vectorized)
    E_solution = np.array([fsolve(hubble_nonlinear, x0=1.0, args=(zi, Omega_m, p, zeta))[0] for zi in z])

    result = Calculate_return_values(E_solution, Type)
    return result if len(result) > 1 else result[0]
    
def f3CDM_MODEL(z, param_dict, Type="SNe"):
    z = np.atleast_1d(z)  # Ensure z is treated as an array
    def hubble_nonlinear(E, z, Omega_m, Gamma):
        return E**2 - Omega_m * (1 + z)**3 - ((1 - Omega_m)/(2-np.log(Gamma)))*(2-np.log(Gamma*E))

    Omega_m = param_dict["Omega_m"]
    Gamma = param_dict["Gamma"]

    # Solve for E(z) (vectorized)
    E_solution = np.array([fsolve(hubble_nonlinear, x0=1.0, args=(zi, Omega_m, Gamma))[0] for zi in z])

    result = Calculate_return_values(E_solution, Type)
    return result if len(result) > 1 else result[0]
    
def f3CDM_v_MODEL(z, param_dict, Type="SNe"):
    z = np.atleast_1d(z)  # Ensure z is treated as an array
    def hubble_nonlinear(E, z, Omega_m, Gamma,zeta):
        return E**2 - Omega_m * (1 + z)**(3-3*zeta) - ((1 - Omega_m)/(2-np.log(Gamma)))*(2-np.log(Gamma*E))

    Omega_m = param_dict["Omega_m"]
    Gamma = param_dict["Gamma"]
    zeta = param_dict["zeta"]

    # Solve for E(z) (vectorized)
    E_solution = np.array([fsolve(hubble_nonlinear, x0=1.0, args=(zi, Omega_m, Gamma, zeta))[0] for zi in z])

    result = Calculate_return_values(E_solution, Type)
    return result if len(result) > 1 else result[0]

def BetaRn_MODEL(z, param_dict, Type = "SNe"):
    '''
    Beta R^n model for cosmology.
    '''
    z = np.atleast_1d(z)  # Ensure z is an array for consistent operations
    qz = param_dict["q0"]+param_dict["q1"]*((z)/(z+1))
    jz = qz*(2*qz+1)+((param_dict["q1"])/(z+1))
    term1 = (param_dict["Omega_m"]*(1+z)**3)
    term2bottom = qz*param_dict["n"]*param_dict["beta"]*(6**(param_dict["n"]-1))*((1-qz)**(param_dict["n"]-1))
    term3bottom = (1+((1-qz)/(param_dict["n"]*qz)) - ((param_dict["n"]-1)/(qz-(qz**2)))*(2+qz-jz))
    model = ((term1)/(term2bottom*term3bottom))**((1)/(2*param_dict["n"]))
    result = Calculate_return_values(model, Type)
    return result if len(result) > 1 else result[0]
    
def BetaR_alphaR_MODEL(z, param_dict, Type="SNe"):
    """
    Lambda Cold Dark Matter (LCDM) model.
    Args:
        z (float or np.ndarray): Redshift value(s).
        param_dict (dict): Dictionary containing cosmological parameters.
        Type (str): Type of observation ('SNe', 'OHD', 'CC', f_sigma_8', or 'BAO').

    Returns:
        float or np.ndarray: Theoretical model prediction based on the observation type.
    """
    z = np.atleast_1d(z)  # Ensure z is an array for consistent operations
    model = np.sqrt((1/param_dict["alpha"])*(param_dict["Omega_m"]*(1+z)**3-(param_dict["beta"]/6)))
    result = Calculate_return_values(model, Type)
    return result if len(result) > 1 else result[0]  # Return scalar if input was scalar
    
##############################################################
# Functions to control Models for the entire program
##############################################################
def Get_model_function(model_name):
    '''
    Retrieve the correct model function based on the model name.

    Args:
        model_name (str)    : Name of the cosmological model (e.g., 'LCDM', 'betaRn').

    Returns:
        function            : The corresponding model function.

    Raises:
        ValueError          : If the model name is not recognized.
    '''
    models = {
        "LCDM": LCDM_MODEL,
        "f1CDM": f1CDM_MODEL,
        "f2CDM": f2CDM_MODEL,
        "f3CDM": f3CDM_MODEL,
        "LCDM_v": LCDM_v_MODEL,
        "f1CDM_v": f1CDM_v_MODEL,
        "f2CDM_v": f2CDM_v_MODEL,
        "f3CDM_v": f3CDM_v_MODEL,
        "BetaRn": BetaRn_MODEL,
        "aRBR": BetaR_alphaR_MODEL,
    }
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not recognized. Available models: {list(models.keys())}")
    
    return models[model_name]    
    
def Get_model_names(model_name):
    # Set up free parameter list for your new models. N.B. Only include the parameters that you have in your model
    all_models = {
        "LCDM"     : {"parameters": ["Omega_m"]},
        "f1CDM"  : {"parameters": ["Omega_m", "n"]},
        "f2CDM"  : {"parameters": ["Omega_m", "p"]},
        "f3CDM"  : {"parameters": ["Omega_m", "Gamma"]},
        
        "LCDM_v"     : {"parameters": ["Omega_m","zeta"]},
        "f1CDM_v"  : {"parameters": ["Omega_m", "n","zeta"]},
        "f2CDM_v"  : {"parameters": ["Omega_m", "p","zeta"]},
        "f3CDM_v"  : {"parameters": ["Omega_m", "Gamma","zeta"]},
        
        "BetaRn" : {"parameters": ["Omega_m", "q0", "q1", "beta", "n"]},
        "aRBR" : {"parameters": ["Omega_m", "alpha", "beta"]},
    }
    models = {name: all_models[name] for name in model_name if name in all_models}
    return models

def Calculate_return_values(model, Type):
    '''
    Shared logic for computing return values based on the observation type.

    Args:
        model (float): Calculated model value.
        Type (str): Type of observation ('SNe', 'OHD', 'CC', 'f_sigma_8', or 'BAO').

    Returns:
        float or None: Theoretical model prediction based on the observation type.
    '''
    return {"SNe": 1 / model,  # Supernovae (inverse of model)
                   "OHD": model,      # Observational Hubble Data
                   "CC": model,       # Cosmic Chronometers
                   "f_sigma_8": model,  # Growth rate of structure including sigma_8 as a parameter
                   "f": model,   # Growth rate of structure
                   "BAO": 1 / model,  # Baryon Acoustic Oscillations (inverse of model)}
                   }.get(Type, None)

##############################################################
# General used functions
##############################################################

def Hubble(param_dict):
    '''
    Calculate the Hubble distance in units of km/s/Mpc.

    Args:
        param_dict (dict): Dictionary containing cosmological parameters,
                           specifically 'H_0' (Hubble constant in km/s/Mpc).

    Returns:
        float: Hubble distance in km/s/Mpc.
    '''
    return 300000 / param_dict['H_0']

def Comoving_distance_vectorized(MODEL_func, redshifts, param_dict, Type):
    """
    Compute the comoving distances for an array of redshifts (vectorized version).
    Args:
        MODEL_func (callable): Function that calculates the Hubble parameter at a given redshift.
        redshifts (array-like): Array of redshifts to compute the comoving distance for.
        param_dict (dict): Dictionary containing cosmological parameters.
        Type (str): Type of observation.

    Returns:
        np.ndarray: Array of comoving distances in Mpc.
    """
    comoving_distances = np.zeros_like(redshifts)
    for i, z in enumerate(redshifts):
        comoving_distances[i] = integrate.quad(MODEL_func, 0, z, args=(param_dict, Type))[0]

    return comoving_distances * Hubble(param_dict)
    
##############################################################
# Functions for cosmological calculations, specifically used
# for observations involving sigma8 or fsigma8.
##############################################################  
# Define the integral term for f_sigma8(z)
def integral_term(z, MODEL_func, param_dict, Type="f_sigma_8"):
    """
    Compute the integral term in the f_sigma_8 definition:
    Integral = ∫ (Omega_zeta^gamma / (1 + z)) dz from 0 to z.

    Args:
        z (float or array-like): Upper limit of the integral (current redshift).
        MODEL_func (function): Function to compute H(z) or related model values.
        param_dict (dict): Dictionary containing cosmological parameters.
        Type (str, optional): Type of observation. Defaults to "f_sigma_8".

    Returns:
        float or array-like: Value of the integral term for scalar or array input.
    """
    gamma = param_dict["gamma"]

    def integrand(z_prime):
        Omega_zeta = matter_density_z(z_prime, MODEL_func, param_dict, Type)
        return (Omega_zeta ** gamma) / (1 + z_prime)

    # Handle scalar input
    if np.isscalar(z):
        integral_value, _ = integrate.quad(integrand, 0, z)
        return integral_value

    # Handle array input
    elif isinstance(z, (list, np.ndarray)):
        results = []
        for z_val in z:
            integral_value, _ = integrate.quad(integrand, 0, z_val)
            results.append(integral_value)
        return np.array(results)
    
    else:
        raise ValueError("Input z must be a scalar or array-like.")
    
def matter_density_z(z, MODEL_func, param_dict, Type = "f_sigma_8"):
    '''
    Compute the matter density parameter Omega(z) at redshift z.

    Args:
        z (float): Redshift.
        param_dict (dict)       : Dictionary containing cosmological parameters.
        Type (str, optional)    : Type of observation. Defaults to "f_sigma_8".

    Returns:
        float                   : Omega(z), the ratio of matter density to critical density at z.
    '''
    #Model_value = MODEL_func(z, param_dict, Type)
    matter_density_z = (param_dict['Omega_m'] * (1 + z)**3) / (MODEL_func(z, param_dict, Type)**2)
    return matter_density_z

##############################################################
# Functions for cosmological calculations, specifically used
# for observations involving BAO.
##############################################################
def dmrd(redshifts, MODEL_func, param_dict, Type):
    """
    Compute the dimensionless comoving distance D_M / r_d for one or more redshifts.

    Args:
        redshifts (float or array-like): Redshift(s).
        MODEL_func (callable): Function for Hubble parameter or related quantities.
        param_dict (dict): Dictionary of cosmological parameters.
        Type (str): Type of observation.

    Returns:
        float or np.ndarray: D_M / r_d for the given redshift(s).
    """
    #dmrd = Comoving_distance(MODEL_func, redshift, param_dict, Type) / param_dict['r_d']
    comoving_distances = Comoving_distance_vectorized(MODEL_func, np.atleast_1d(redshifts), param_dict, Type)
    return comoving_distances / param_dict['r_d']

def dhrd(redshift, MODEL_func, param_dict, Type):
    '''
    Compute the dimensionless Hubble distance D_H / r_d.

    Args:
        redshift (float): Redshift.
        MODEL_func (callable): Function for Hubble parameter or related quantities.
        param_dict (dict): Dictionary of cosmological parameters.
        Type (str): Type of observation.

    Returns:
        float: D_H / r_d, dimensionless Hubble distance.
    '''
    dhrd = Hubble(param_dict) * MODEL_func(redshift, param_dict, Type) / param_dict['r_d']
    return dhrd
    
def dvrd(redshifts, MODEL_func, param_dict, Type):
    """
    Compute the dimensionless volume-averaged distance D_V / r_d for one or more redshifts.

    Args:
        redshifts (float or array-like): Redshift(s).
        MODEL_func (callable): Function for Hubble parameter or related quantities.
        param_dict (dict): Dictionary of cosmological parameters.
        Type (str): Type of observation.

    Returns:
        float or np.ndarray: D_V / r_d for the given redshift(s).
    """
    #dvrd = (Hubble(param_dict) * ((redshift * (Comoving_distance(MODEL_func, redshift, param_dict, Type) / Hubble(param_dict))**2)/ (1 / MODEL_func(redshift, param_dict, Type))) ** (1 / 3)) / param_dict['r_d']
    redshifts = np.atleast_1d(redshifts)
    
    # Compute comoving distances and Hubble parameter for all redshifts
    comoving_distances = Comoving_distance_vectorized(MODEL_func, redshifts, param_dict, Type)
    H_vals = np.array([MODEL_func(z, param_dict, Type) for z in redshifts])

    # Calculate D_V for each redshift
    dvrd_vals = (
        Hubble(param_dict)
        * ((redshifts * (comoving_distances / Hubble(param_dict))**2) / (1 / H_vals)) ** (1 / 3)
    ) / param_dict['r_d']

    return dvrd_vals
    
def dArd(redshifts, MODEL_func, param_dict, Type):
    """
    Compute the dimensionless angular diameter distance D_A / r_d for one or more redshifts.

    Args:
        redshifts (float or array-like): Redshift(s).
        MODEL_func (callable): Function for Hubble parameter or related quantities.
        param_dict (dict): Dictionary of cosmological parameters.
        Type (str): Type of observation.

    Returns:
        float or np.ndarray: D_A / r_d for the given redshift(s).
    """
    #dArd = (Comoving_distance(MODEL_func, redshift, param_dict, Type) / (1 + redshift)) / param_dict['r_d']
    redshifts = np.atleast_1d(redshifts)
    
    # Compute comoving distances
    comoving_distances = Comoving_distance_vectorized(MODEL_func, redshifts, param_dict, Type)

    # Calculate D_A / r_d
    dArd_vals = (comoving_distances / (1 + redshifts)) / param_dict['r_d']

    return dArd_vals

    