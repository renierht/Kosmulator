import numpy as np
import Plots as MP
import User_defined_modules as UDM
import scipy.linalg as la

def Calc_chi(Type, type_data, type_data_error,model):
    if Type=="CC":
        chi = Covariance_matrix(model, type_data, type_data_error)
    else:
        chi = np.sum(((type_data - model)**2) / (type_data_error ** 2))     
    return chi
    
def Calc_PantP_chi(mb, trig, cepheid, cov, model):
    M = -19.20
    meub = mb - M
    #Incorporate Cepheid calibrations or use model predictions
    moduli = np.where(trig == 1, cepheid, model)  # `trig` and `cepheid` must be defined
    delta = meub - moduli # Calculate residuals
    # Solve using covariance matrix (cov should be the actual covariance matrix of mb data)
    residuals = la.solve_triangular(cov, delta, lower=True, check_finite=False)
    chi= (residuals ** 2).sum()  # Sum of squared residuals for chi-squared value
    return chi
    
def Calc_BAO_chi(covd1, Model_func, param_dict, Type, rd):
    zz12 = [UDM.dvrd(0.295, Model_func, param_dict, Type, rd) - 7.925129270, UDM.dmrd(0.510, Model_func, param_dict, Type, rd) - 13.62003080, 
            UDM.dhrd(0.510, Model_func, param_dict, Type, rd) - 20.98334647, UDM.dmrd(0.706, Model_func, param_dict, Type, rd) - 16.84645313, 
            UDM.dhrd(0.706, Model_func, param_dict, Type, rd) - 20.07872919, UDM.dmrd(0.930, Model_func, param_dict, Type, rd) - 21.70841761,
            UDM.dhrd(0.930, Model_func, param_dict, Type, rd) - 17.87612922, UDM.dmrd(1.317, Model_func, param_dict, Type, rd) - 27.78720817, 
            UDM.dhrd(1.317, Model_func, param_dict, Type, rd) - 13.82372285, UDM.dvrd(1.491, Model_func, param_dict, Type, rd) - 26.07217182, 
            UDM.dmrd(2.330, Model_func, param_dict, Type, rd) - 39.70838281, UDM.dhrd(2.330, Model_func, param_dict, Type, rd) - 8.522565830]
    covinvd1 = np.linalg.inv(covd1)
    desi_chi = np.dot(zz12, np.dot(covinvd1, zz12))
    return desi_chi
    
    
    
def Covariance_matrix(model, type_data, type_data_error):
    Cov_matrix = np.diag(type_data_error**2)
    C_inv = np.linalg.inv(Cov_matrix)
    delta_H = type_data - model
    chi = np.dot(delta_H, np.dot(C_inv, delta_H))
    return chi
    
def AutoCorr(pos, iterations, sampler):
    index = 0
    autocorr = np.empty(iterations)
    old_tau = np.inf
    for sample in sampler.sample(pos, iterations=iterations, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        converged = np.all(tau * 100 < sampler.iteration)# Check convergence
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged: # Break if converged
            break
        old_tau = tau
        MP.autocorrPlot(autocorr, index)   