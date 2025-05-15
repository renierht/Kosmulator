#!/usr/bin/env python
# coding: utf-8

# In[140]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import camb
import healpy as hp
from classy import Class
from multiprocess import Pool,get_context
import corner


# In[141]:


l = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt', usecols = 0)
Dl = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt', usecols = 1)
Dl_minus = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt', usecols = 2)
Dl_plus = np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt', usecols = 3)

l = l[:2499]
Dl = Dl[:2499]
Dl_minus = Dl_minus[:2499]
Dl_plus = Dl_plus[:2499]

print(l)
l_theory = np.arange(2, 2501)

print(l_theory)
np.size(l)


# In[142]:


sigma = (Dl_minus+Dl_plus/2)
np.size(sigma)


# In[179]:


# Define your cosmology (what is not specified will be set to CLASS default parameters)
params = {
    'output': 'tCl lCl',
    'l_max_scalars': 2500,
    'lensing': 'yes',
    'A_s': 2.5e-9,
    'n_s': 0.9624, 
    'h': 0.6711,
    'omega_b': 0.022068,
    'omega_cdm': 0.12029}

# Create an instance of the CLASS wrapper
cosmo = Class()

# Set the parameters to the cosmological code
cosmo.set(params)

# Run the whole code. Depending on your output, it will call the
# CLASS modules more or less fast. For instance, without any
# output asked, CLASS will only compute background quantities,
# thus running almost instantaneously.
# This is equivalent to the beginning of the `main` routine of CLASS,
# with all the struct_init() methods called.
cosmo.compute()

# Access the lensed cl until l=2000
cls = cosmo.lensed_cl(2500)
l_theory = np.arange(2, 2501)

Dl_theory = cls['tt'][2:] * l_theory * (l_theory + 1) * 1e12


# In[180]:


plt.plot(l_theory,Dl_theory)
plt.errorbar(l, Dl, yerr=[Dl_plus,Dl_minus], xerr=None, fmt='.')
plt.ylabel('D_l [μK^2]')
plt.xlabel('l')
plt.title('Power spectrum (TT)')
plt.savefig('Power_spectrum.jpg')


# In[181]:


def likelihood(theta, l, Dl, Dl_minus, Dl_plus):
    # Correct the order of parameters
    ln_A_s, n_s, H0, Omega_b, Omega_cdm = theta

    A_s = np.exp(ln_A_s)
    params = {
        'output': 'tCl lCl',
        'l_max_scalars': 2500,
        'lensing': 'yes',
        'A_s': A_s,
        'n_s': n_s, 
        'h': H0/100,  # H0 in terms of H0/100
        'omega_b': Omega_b,
        'omega_cdm': Omega_cdm
    }

    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()

    l_theory = np.arange(2, 2501)
    
    cls = cosmo.lensed_cl()
    Dl_theory = cls['tt'][2:] * l_theory * (l_theory + 1) * 1e12

    chi2 = np.sum(((Dl - Dl_theory)**2) / (Dl_minus**2 + Dl_plus**2))
    return -0.5 * chi2


# In[182]:


def lnprior(theta):
    ln_A_s, n_s, H0, Omega_b, Omega_cdm = theta
    if (-30<ln_A_s<0 and 0.8<n_s< 1.2 and 50 < H0 < 70 and
        0 < Omega_b < 0.03 and 0 < Omega_cdm < 0.5):
        return 0.0
    return -np.inf


# In[183]:


def lnprob(theta, l, Dl, Dl_minus, Dl_plus):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + likelihood(theta, l, Dl, Dl_minus, Dl_plus)


# In[184]:


A_s_0 = np.log(2.3e-9)
n_s_0 = 0.9624
h_0 = 0.6711
omega_b_0 = 0.022068
omega_cdm_0 = 0.12029


# In[185]:


theta = [A_s_0, n_s_0, h_0, omega_b_0, omega_cdm_0]
nll = lambda *args: -likelihood(*args)
Minimize = op.minimize(nll, theta,args=(l, Dl, Dl_minus, Dl_plus),
                       bounds=((-30, 0.1**2), (0.8,1.2), (60, 80), (0.01, 0.03), (0.01, 0.5)))
print("Minimized parameters:", Minimize['x'])


# In[ ]:


ndim = 5
nwalker = 100
nsteps = 5000
burn = 100

pos = [Minimize["x"] + 1e-4 * np.random.randn(ndim) for i in range(nwalker)]

ncores = 32

with Pool(ncores) as pool:
    sampler = emcee.EnsembleSampler(nwalker,ndim,lnprob,args = (l,Dl,Dl_minus, Dl_plus), pool=pool)
    sampler.run_mcmc(pos,nsteps, progress=True)
    samples1 = sampler.chain[:,burn:,:].reshape((-1,ndim))


# In[ ]:


fig = plt.figure()
corner.corner(samples1,labels=["exp(A_s)","n_s","H","omega_b","omega_cdm"],color = "blue",plot_density=True, smooth=True,storechain=True, plot_datapoints=False, 
                        fill_contours=True,show_titles=True,title_fmt = '.3f',bins=100)
plt.savefig("Plank_2018.png",dpi=400)
plt.show()


# In[95]:


test_theta = [2.29703807e-09, 0.962941293, 60, 0.022040507, 0.121155044]
print("Likelihood for minimized parameters:", likelihood(test_theta, l, Dl, sigma))


# In[78]:


test_theta = [2.1e-9, 0.96, 67.4, 0.022, 0.12]  # Example input
print(likelihood(test_theta, l, Dl, Dl_minus, Dl_plus))


# In[94]:


print("lnprior for minimized parameters:", lnprior([2.29703807e-09, 0.962941293, 60, 0.022040507, 0.121155044]))


# In[ ]:


clik_file = '/Users/robertrugg/Downloads/baseline/plc_3.0/hi_l/plik/plik_rd12_HM_v22_TT.clik'
likelihood = clik.clik(clik_file)

def likelihood(theta):
    A_s, n_s, H0, Omega_b, Omega_cdm = theta
    params = {
        'A_s': A_s,
        'n_s': n_s,
        'h': H0 / 100,
        'omega_b': Omega_b,
        'omega_cdm': Omega_cdm
    }

    param_vector = [
        params['omega_b'], 
        params['omega_cdm'], 
        params['h'], 
        params['n_s'], 
        params['A_s']
    ]

    try:
        loglike = likelihood()(l, Dl, Dl_model, Dl_minus, Dl_plus)
    except Exception as e:
        loglike = -np.inf
        print(f"Error in likelihood calculation: {e}")
    
    return loglike


# In[96]:


test_theta = [2.29703807e-09, 0.962941293, 60, 0.022040507, 0.121155044]
print("lnprob for minimized parameters:", lnprob(test_theta, l, Dl, sigma))


# In[2]:


import numpy as np
print(np.__version__)


# In[39]:


print(type(clik))


# In[40]:


print(dir(likelihood))  # This will show all the available methods of the 'likelihood' object


# In[97]:


test_theta = [2.29703807e-09, 0.962941293, 60, 0.022040507, 0.121155044]
print("lnprob for minimized parameters:", lnprob(test_theta, l, Dl, sigma))


# In[98]:


for idx, p in enumerate(pos):
    print(f"Walker {idx} initial position: {p}, lnprior: {lnprior(p)}")


# In[100]:


# Ensure walkers start in a reasonable range, adding larger perturbations to test
pos = [
    np.array(test_theta) + np.random.randn(ndim) * 1e-2  # Increase step size
    for i in range(nwalker)
]


# In[ ]:




