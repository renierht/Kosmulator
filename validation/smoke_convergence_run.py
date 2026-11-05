"""Real Model 9 CC likelihood through the emcee convergence and resume path."""
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import User_defined_modules as U
from Kosmulator_main import Config, constants as K, Kosmulator_MCMC as M

with tempfile.TemporaryDirectory(prefix='ide_convergence_') as tmp, patch.object(K,'force_emcee',True,create=True):
    name='SignSwitch_IDE_1'
    configs,data=Config.create_config(U.Get_model_names([name]),model_name=[name],observation=[['CC']],
        prior_limits=dict(Omega_m=(.25,.35),H_0=(65,75),w=(-1.2,-.85),delta=(-.06,.06)),
        true_values=dict(Omega_m=.307,H_0=70,w=-1.05,delta=.02),
        free_w=True,free_w_dm=False,nwalkers=16,nsteps=120,burn=20)
    cfg=configs[name]; path=str(Path(tmp)/'model9.h5')
    kw=dict(data=data,saveChains=True,chain_path=path,overwrite=False,MODEL_func=U.Get_model_function(name),
        CONFIG=cfg,autoCorr=True,parallel=False,model_name=name,obs=['CC'],Type=['CC'],colors=[],
        convergence=.05,last_obs=True,PLOT_SETTINGS=dict(seed=413,init_jitter_frac=.01,
        autocorr_save_path=tmp,autocorr_check_every=50,dpi=60),obs_index=0,
        use_mpi=False,num_cores=1,pool=None,vectorised=True)
    np.random.seed(413)
    samples=M.run_mcmc(**kw)
    assert samples.shape==(1600,4) and np.isfinite(samples).all()
    cfg['nsteps']=160
    resumed=M.run_mcmc(**dict(kw,resumeChains=True))
    assert resumed.shape==(2240,4) and np.isfinite(resumed).all()
    report=json.loads(Path(path+'.convergence.json').read_text())
    assert report['status']['n_post']==140
    assert report['history']['it']==[150,160]
    assert report['rules']==dict(tau_factor=100.,ess_min=2000.,rhat_max=1.01,rtol=.05)
    with h5py.File(path) as f: assert not f.attrs['converged']
    assert list(Path(tmp).rglob('*.png'))
print('PASS: Model 9 emcee production path, fixed cold DM, convergence report, ceiling and resume')
