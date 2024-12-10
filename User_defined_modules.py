import numpy as np

##############################################################
#### Only when using sigma8 or fsigma8 observations
##############################################################
def nonLinear_Hubble_parameter(z, param_dict, Type="fsigma8"):
    """Calculate the nonlinear Hubble parameter E(z)."""
    model = np.sqrt(param_dict['Omega_m']*(1 + z)**3+1-param_dict['Omega_m'])
    return model
    
def matter_density_z(z, param_dict, Type="fsigma8"):
    """Compute Omega(z), the matter density at redshift z."""
    matter_density_z = param_dict['Omega_m'] * (1 + z)**3 / (nonLinear_Hubble_parameter(z, param_dict, Type="fsigma8")**2)
    return matter_density_z
    
def Hubble(param_dict):
    return 300000/param_dict['H_0']

def comoving_distance(Model_func, redshift, param_dict, Type):
    comoving_distance = Hubble(param_dict)*integrate.quad(MODEL_func,0,redshift,args=(param_dict,Type))
    return comoving_distance

def dmrd(redshift, Model_func, param_dict, Type, rd):
    dmrd = comoving_distance(Model_func, redshift, param_dict, Type)/rd
    return dmrd

def dhrd(redshift, Model_func, param_dict, Type, rd):
    dhrd = Hubble(param_dict)*MODEL_func(redshift, param_dict, Type="SNe")/rd
    return dhrd
    
def dvrd(redshift, Model_func, param_dict, Type, rd):
    dvrd = (Hubble(param_dict)*((redshift*(comoving_distance(Model_func, redshift, param_dict, Type)/Hubble(param_dict))**2)/(1/MODEL_func(redshift, param_dict, Type="SNe")))**(1/3))/rd
    return dvrd
    
def dArd(redshift, Model_func, param_dict, Type, rd):
    dArd = (comoving_distance(Model_func, redshift, param_dict, Type)/(1+redshift))/rd
    return dArd