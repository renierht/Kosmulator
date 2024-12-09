import numpy as np
import Plots as MP

def Calc_chi(Type, type_data, type_data_error,model):
    if Type=="SNe" or Type=="OHD":
        chi = np.sum(((type_data - model)**2) / (type_data_error ** 2))
    elif Type=="CC":
        chi = Covariance_matrix(model, type_data, type_data_error)
    return chi
        
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