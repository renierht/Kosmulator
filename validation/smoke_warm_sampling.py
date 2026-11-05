"""Short real-CC sampling/save/load/resume checks, not convergence claims."""
import sys
import json
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from unittest.mock import patch
import emcee
import User_defined_modules as U
from Kosmulator_main import Config, constants as K, Kosmulator_MCMC as M, utils, chain_policy
from Plots import Plots

priors=dict(Omega_m=(.25,.35),H_0=(65,75),w=(-1.2,-.85),w_dm=(-.02,.02),delta=(-.06,.06))
report=[]
with tempfile.TemporaryDirectory(prefix='kosmulator_warm_smoke_') as tmp, \
     patch.object(K,'force_emcee',True,create=True), \
     patch.object(U,'ALLOW_NEGATIVE_ENERGIES',True), \
     patch.object(U,'ALLOW_BIG_RIP',True), \
     patch.object(U,'ALLOW_DOOM_FACTOR_INSTABILITIES',True):
    for name in ('NonLinear_IDE_2','SignSwitch_IDE_1','SignSwitch_IDE_2'):
        for fw in (False,True):
            for fd in (False,True):
                tv=dict(Omega_m=.307,H_0=70,w=-1.05,w_dm=.005,delta=-.02 if name=='SignSwitch_IDE_2' else .02)
                configs,data=Config.create_config(U.Get_model_names([name]),true_values=tv,
                    prior_limits=priors,observation=[['CC']],model_name=[name],
                    free_w=fw,free_w_dm=fd,nwalkers=16,nsteps=24,burn=4)
                c=configs[name]
                path=str(Path(tmp)/f'{name}_{fw}_{fd}.h5')
                kw=dict(data=data,saveChains=True,chain_path=path,overwrite=False,
                        MODEL_func=U.Get_model_function(name),CONFIG=c,autoCorr=False,
                        parallel=False,model_name=name,obs=['CC'],Type=['CC'],colors={},
                        convergence=.01,last_obs=True,PLOT_SETTINGS={'seed':31013,'init_jitter_frac':.01},
                        obs_index=0,use_mpi=False,num_cores=1,pool=None,vectorised=True)
                np.random.seed(31013)
                samples=M.run_mcmc(**kw)
                assert samples.shape==(20*16,c['ndim'][0]) and np.isfinite(samples).all()
                chain_policy.check(path,c,0)
                loaded=utils.load_or_run_chain(output_dir=tmp,chain_file=Path(path).name,
                    overwrite=False,CONFIG_model=c,data=data,MODEL_func=kw['MODEL_func'],
                    convergence=.01,parallel=False,pool=None,vectorised=True,load_only=True,obs_index=0)
                np.testing.assert_array_equal(samples,loaded)
                c['nsteps']=28
                resumed=M.run_mcmc(**dict(kw,resumeChains=True))
                assert resumed.shape==(24*16,c['ndim'][0]) and np.isfinite(resumed).all()
                backend=emcee.backends.HDFBackend(path,read_only=True)
                assert backend.iteration==28
                assert np.isfinite(backend.get_log_prob()).all()
                percentiles=np.percentile(resumed,[50,84,16],axis=0)
                med={par:tuple(percentiles[:,j]) for j,par in enumerate(c['parameters'][0])}
                plot_settings=dict(save_root='Plots/Saved_plots',
                    output_suffix=f'warm_validation/{name}_{fw}_{fd}',dpi=100,latex_enabled=False)
                Plots.best_fit_plots({name:{'CC':med}},{name:c},data,plot_settings)
                plot_path=Path('Plots/Saved_plots')/plot_settings['output_suffix']/name/'best_fits/CC/bestfit.png'
                assert plot_path.is_file() and plot_path.stat().st_size>1000
                report.append(dict(model=name,free_w=fw,free_w_dm=fd,ndim=c['ndim'][0],
                                   fixed_parameters=c['fixed_parameters'],iterations=int(backend.iteration),
                                   walkers=16,retained_samples=len(resumed),read_only_roundtrip=True,
                                   best_fit_plot=str(plot_path)))
Path('validation/warm_sampling_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'PASS: {len(report)} real-CC sampling/save/load/resume combinations')
