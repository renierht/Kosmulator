import numpy as np
from scipy import integrate

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
    comoving_distance = Hubble(param_dict) * integrate.quad(MODEL_func, 0, redshift, args = (param_dict, Type))[0]
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