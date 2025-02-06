import numpy as np
from scipy import integrate
from scipy.optimize import fsolve
from scipy.interpolate import RegularGridInterpolator

##############################################################
# Define your cosmological model used for MCMC analysis
##############################################################
def LCDM_MODEL(z, param_dict, Type = "SNe"):
    '''
    Lambda Cold Dark Matter (LCDM) model.
    Args:
        z (float)           : Redshift value.
        param_dict (dict)   : Dictionary containing cosmological parameters.
        Type (str)          : Type of observation ('SNe', 'OHD', 'CC', f_sigma_8', or 'BAO').

    Returns:
        float or None       : Theoretical model prediction based on the observation type.
    '''
    model = np.sqrt(param_dict['Omega_m'] * (1 + z)**3 + 1 - param_dict['Omega_m'])
    return Calculate_return_values(model, Type)
    
def BetaRn_MODEL(z, param_dict, Type = "SNe"):
    '''
    Beta R^n model for cosmology.
    '''
    qz = param_dict["q0"]+param_dict["q1"]*((z)/(z+1))
    jz = qz*(2*qz+1)+((param_dict["q1"])/(z+1))
    term1 = (param_dict["Omega_m"]*(1+z)**3)
    term2bottom = qz*param_dict["n"]*param_dict["beta"]*(6**(param_dict["n"]-1))*((1-qz)**(param_dict["n"]-1))
    term3bottom = (1+((1-qz)/(param_dict["n"]*qz)) - ((param_dict["n"]-1)/(qz-(qz**2)))*(2+qz-jz))
    model = ((term1)/(term2bottom*term3bottom))**((1)/(2*param_dict["n"]))
    return Calculate_return_values(model, Type)
    
def NonLinear_fsolve_MODEL(z, param_dict, Type="SNe"):
    """
    Solve the nonlinear algebraic equation for E(z).

    Args:
        z (float): Redshift.
        param_dict (dict): Dictionary containing cosmological parameters.
        Type (str): Type of observation.

    Returns:
        float: Normalized Hubble parameter E(z).
    """

    def hubble_nonlinear(E, z, Omega_m, n):
        """ The nonlinear equation to be solved for E(z). """
        return E**2 - (Omega_m * (1 + z)**3 + (1 - Omega_m) * E**(2 * n))

    # Extract model parameters
    Omega_m = param_dict["Omega_m"]
    n = param_dict["n"]

    # Solve for E(z) using fsolve, starting with an initial guess of 1.0
    E_solution = fsolve(hubble_nonlinear, 1.0, args=(z, Omega_m, n))[0]

    return Calculate_return_values(E_solution, Type)

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
        "BetaRn": BetaRn_MODEL,
        "NonLinear": NonLinear_fsolve_MODEL,
        # Add new model names to the model dict
    }
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not recognized. Available models: {list(models.keys())}")
    
    return models[model_name]    
    
def Get_model_names(model_name):
    # Set up free parameter list for your new models
    all_models = {
        "LCDM"     : {"parameters": ["Omega_m", "H_0"]}, #, "M_abs", "rd", "ns", "As", "Omega_b", "gamma", "sigma_8",]
        "BetaRn" : {"parameters": ["Omega_m", "H_0", "q0", "q1", "beta", "n"]},
        "NonLinear"  : {"parameters": ["Omega_m", "H_0", "n"]},
        #"BetaR2n" : {"parameters": ["Omega_m", "H_0", "q0"]},
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
                   "sigma_8": model,   # Growth rate of structure
                   "BAO": 1 / model,  # Baryon Acoustic Oscillations (inverse of model)}
                   }.get(Type, None)


    """
    Compute E(z) dynamically using a precomputed interpolation table for any user-defined model.

    Args:
        z (float): Redshift.
        param_dict (dict): Cosmological parameters.
        Type (str): Type of observation.
        CONFIG (dict): Configuration dictionary.
        mod (str): Model name.

    Returns:
        float: Normalized Hubble parameter E(z).
    """

    global E_interpolator

    if mod is None:
        raise ValueError("Model name (mod) must be specified.")

    # Ensure the interpolation table is created only ONCE per model
    if mod not in E_interpolator:
        if CONFIG is None:
            raise ValueError("CONFIG must be provided to initialize the interpolation table.")
        print(f"Initializing precomputed E(z) table for model {mod} dynamically...")

        # Get the user-defined model function from User_defined_modules
        MODEL_func = Get_model_function(mod)

        # Precompute the interpolation table for the selected model
        E_interpolator[mod], interpolator_param_names = precompute_interpolation_table(CONFIG, mod, MODEL_func)

    # Extract only the parameters needed for interpolation
    param_values = [param_dict[param] for param in interpolator_param_names]

    # Use the interpolator to get E(z) for the current parameters
    E_z = E_interpolator[mod](param_values + [z])[0]

    return Calculate_return_values(E_z, Type)

##############################################################
# General used functions
##############################################################

def Comoving_distance(MODEL_func, redshift, param_dict, Type):
    '''
    Compute the comoving distance to a given redshift.

    Args:
        MODEL_func (callable): Function that calculates the Hubble parameter at a given redshift.
        redshift (float): Redshift to compute the comoving distance to.
        param_dict (dict): Dictionary containing cosmological parameters.
        Type (str): Type of observation.

    Returns:
        float: Comoving distance in Mpc.
    '''
    comoving_distance = Hubble(param_dict)*integrate.quad(MODEL_func, 0, redshift, args = (param_dict, Type))[0]
    return comoving_distance

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

##############################################################
# Functions for cosmological calculations, specifically used
# for observations involving sigma8 or fsigma8.
##############################################################  
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
def dmrd(redshift, MODEL_func, param_dict, Type):
    '''
    Compute the dimensionless comoving distance D_M / r_d.

    Args:
        redshift (float): Redshift.
        MODEL_func (callable): Function for Hubble parameter or related quantities.
        param_dict (dict): Dictionary of cosmological parameters.
        Type (str): Type of observation.
        rd (float): Sound horizon at the drag epoch (r_d).

    Returns:
        float: D_M / r_d, dimensionless comoving distance.
    '''
    dmrd = Comoving_distance(MODEL_func, redshift, param_dict, Type) / param_dict['r_d']
    return dmrd

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
    
def dvrd(redshift, MODEL_func, param_dict, Type):
    '''
    Compute the dimensionless volume-averaged distance D_V / r_d.

    Args:
        redshift (float): Redshift.
        MODEL_func (callable): Function for Hubble parameter or related quantities.
        param_dict (dict): Dictionary of cosmological parameters.
        Type (str): Type of observation.

    Returns:
        float: D_V / r_d, dimensionless volume-averaged distance.
    '''
    dvrd = (Hubble(param_dict) * ((redshift * (Comoving_distance(MODEL_func, redshift, param_dict, Type) / Hubble(param_dict))**2)
                                  / (1 / MODEL_func(redshift, param_dict, Type))) ** (1 / 3)) / param_dict['r_d']
    return dvrd
    
def dArd(redshift, MODEL_func, param_dict, Type):
    '''
    Compute the dimensionless angular diameter distance D_A / r_d.

    Args:
        redshift (float): Redshift.
        MODEL_func (callable): Function for Hubble parameter or related quantities.
        param_dict (dict): Dictionary of cosmological parameters.
        Type (str): Type of observation.

    Returns:
        float: D_A / r_d, dimensionless angular diameter distance.
    '''
    dArd = (Comoving_distance(MODEL_func, redshift, param_dict, Type) / (1 + redshift)) / param_dict['r_d']
    return dArd