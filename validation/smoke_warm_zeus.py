"""Check the default vectorized sampler with fixed/free EOS and metadata."""
import sys
import tempfile
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
import numpy as np
import h5py
import User_defined_modules as U
from Kosmulator_main import Config, constants as K, Kosmulator_MCMC as M, chain_policy

report=[]
with tempfile.TemporaryDirectory(prefix='kosmulator_zeus_smoke_') as tmp, \
     patch.object(K,'force_emcee',False,create=True),patch.object(K,'force_zeus',True,create=True), \
     patch.object(K,'print_loglike',False,create=True):
    for name,fw,fd in [('SignSwitch_IDE_1',False,False),('SignSwitch_IDE_2',True,True)]:
        priors=dict(Omega_m=(.25,.35),H_0=(65,75),w=(-1.2,-.85),w_dm=(-.02,.02),delta=(-.06,.06))
        tv=dict(Omega_m=.307,H_0=70,w=-1.05,w_dm=.005,delta=-.02 if name.endswith('2') else .02)
        cfg,data=Config.create_config(U.Get_model_names([name]),true_values=tv,prior_limits=priors,
            observation=[['CC']],model_name=[name],free_w=fw,free_w_dm=fd,nwalkers=16,nsteps=24,burn=4)
        c=cfg[name];path=str(Path(tmp)/(name+'.h5'));saved=path.replace('.h5','_zeus.h5')
        kw=dict(data=data,saveChains=True,chain_path=path,overwrite=False,MODEL_func=U.Get_model_function(name),
            CONFIG=c,autoCorr=False,parallel=False,model_name=name,obs=['CC'],Type=['CC'],colors={},
            convergence=.01,last_obs=True,PLOT_SETTINGS={'seed':31013,'init_jitter_frac':.01,
                'callback_ncheck':1000,'autocorr_buffer_after_burn':0,
                'autocorr_save_path':str(Path(tmp)/'autocorr')},obs_index=0,
            use_mpi=False,num_cores=1,pool=None,vectorised=True)
        np.random.seed(31013)
        samples=M.run_mcmc(**kw)
        chain_policy.check(saved,c,0)
        assert samples.shape==(320,c['ndim'][0]) and np.isfinite(samples).all()
        np.testing.assert_array_equal(M.run_mcmc(**kw),samples)
        c['nsteps']=28
        resumed=M.run_mcmc(**dict(kw,resumeChains=True))
        assert resumed.shape==(384,c['ndim'][0]) and np.isfinite(resumed).all()
        with h5py.File(saved,'r') as f: assert f['samples'].shape[0]==28
        report.append(dict(model=name,free_w=fw,free_w_dm=fd,iterations=28,walkers=16,
                           retained_samples=len(resumed),metadata_roundtrip=True))
Path('validation/warm_zeus_results.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS: two Zeus sampling/load/resume EOS configurations')
