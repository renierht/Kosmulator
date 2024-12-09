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


