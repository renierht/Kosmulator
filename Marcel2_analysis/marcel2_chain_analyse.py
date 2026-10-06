import numpy as np

#Needed functions for analysis:
def DIC(log_prob, samples):
    dic_log = -2*log_prob
    d_bar = float(np.nanmean(dic_log))

    #p_D calc D_bar - D(theta_bar):
    theta_bar = float(np.nanmean(samples))
    pD = d_bar - theta_bar

    dic = d_bar + pD
    return dic

def AICc(log_prob, samples, ndim):
    N = len(samples)
    max_ll = np.max(log_prob)
    aicc = 2*N - 2*max_ll + (2*N * (N + 1))/(num_data_points_total - num_params - 1)




#loading the data
data_doom = np.load('flat_samples/IDE_CLASS_Included__Doom-Factor Stable (SiwCDM)__DESI_DR2+PantheonP_SH0ES.npz')
samples, log_probs, names = data_doom["samples"], data_doom['log_prob'], list(data_doom['names'])
ndim = data_doom['ndim']
print(f'ndim: {ndim}')
#Doom factor analysis:
dic_doom = DIC(log_probs, samples)
print(f'Doom-factor stable DIC: {dic_doom}')

#LCDM (one of them at least) analysis
data_lcdm_free = np.load('flat_samples/IDE_free_rd__Free__LCDM_v__BBN_PryMordial_CC_DESI_DR2_Pantheon__BBN_PryMordial+CC+DESI_DR2+Pantheon.npz')
samples_lcdm_free, log_probs_lcdm_free, names_lcdm_free = data_lcdm_free["samples"], data_lcdm_free['log_prob'], list(data_lcdm_free['names'])

dic_lcdm_free = DIC(log_probs_lcdm_free, samples_lcdm_free)
print(f"LCDM_free: {data_lcdm_free}")
doomvLCDMfree = dic_lcdm_free - dic_doom
print(f'dDIC (DOOM vs LCDM_free): {doomvLCDMfree}')