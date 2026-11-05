"""Real CLASS and paper-likelihood checks; no production chain access.
Run from repository root in Kosmulator_test. Outputs validation/cpl_results.json.
"""
import os,sys,json,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import emcee
from Kosmulator_main import Config,constants as K,Class_run as CR,rd_helpers as RD,Kosmulator_MCMC as M,utils
import User_defined_modules as U
paper=json.loads(Path('reproducibility/SAIP2026_IDE/configs/paper.json').read_text())
for k,v in paper['class_settings'].items():setattr(K,k,v)
assert CR.ensure_class_ready('w0waCDM',no_rebuild=True,announce=False),'Need existing LCDM CLASS binary'
Class=CR.get_Class()
results={'class_background':[]}
# Test CPL against the actual CLASS fluid evolution (radiation and neutrinos
# must be included separately; the paper's late-time E(z) omits them).
for w0,wa in [(-.8,-.7),(-1.2,.4),(-2.,1.9),(.9,-1.),(-2.9,-2.9)]:
 c=Class()
 try:
  c.set(dict(h=.7,omega_b=.048*.7**2,omega_cdm=(.3-.048)*.7**2,
    N_ncdm=3,m_ncdm='0.02,0.02,0.02',Neff=3.044,Omega_Lambda=0.,
    w0_fld=w0,wa_fld=wa,use_ppf='yes',output=''))
  c.compute();b=c.get_background();a=1/(1+b['z'])
  # Scale by total density to avoid meaningless relative errors for tiny DE.
  de=b['(.)rho_fld'][-1]*np.exp(-3*(1+w0+wa)*np.log(a)+3*wa*(a-1))
  err=float(np.max(abs(de-b['(.)rho_fld'])/b['(.)rho_tot']))
  assert err<2e-6,(w0,wa,err)
  results['class_background'].append(dict(w0=w0,wa=wa,scaled_density_error=err,rd=float(c.rs_drag())))
 finally:c.struct_cleanup();c.empty()
p=dict(Omega_m=.3111522458018671,H_0=71.86752563631747,M_abs=-19.299221738653277)
rd0=RD.compute_rd_model_class(dict(p,__model_name__='LCDM_v'))
rd1=RD.compute_rd_model_class(dict(p,__model_name__='w0waCDM',w0=-1.,wa=0.))
assert rd0==rd1
results['lcdm_limit_rd_Mpc']=rd0
priors={k:tuple(v) for k,v in paper['prior_limits'].items()};priors.update(w0=(-3,1),wa=(-3,2),r_d=(100,200))
configs,data=Config.create_config(U.Get_model_names(['LCDM_v','w0waCDM']),model_name=['LCDM_v','w0waCDM'],observation=paper['observations'],prior_limits=priors,free_w=True,free_w_dm=False)
cfg=configs['w0waCDM']
data['PantheonPS']['cov']=utils.compute_pantheon_cov(data,cfg,None,0,os.path.join(K.OBSERVATIONS_BASE,'PantheonP.cov'),obs_tag='PantheonPS')
def probability(theta,name='w0waCDM'):
 c=configs[name]
 return M.emcee_prob(theta,data,c['observation_types'][0],c,U.Get_model_function(name),name,c['observations'][0],0)
pp=dict(p,w0=-1.,wa=0.)
values={n:probability([pp[k] for k in configs[n]['parameters'][0]],n) for n in configs}
assert np.max(abs(np.array(values['LCDM_v'])-values['w0waCDM']))<1e-10,values
results['lcdm_limit_likelihood']=values
center=np.array([pp[k] for k in cfg['parameters'][0]])
# Small independent sampler smoke test, deliberately not a convergence run.
rng=np.random.default_rng(42);np.random.seed(42)
initial=center+rng.normal(size=(12,5))*[.001,.02,.03,.05,.005]
with tempfile.TemporaryDirectory(prefix='kosmulator-cpl-') as tmp:
 backend=emcee.backends.HDFBackend(str(Path(tmp)/'smoke.h5'));backend.reset(12,5)
 sampler=emcee.EnsembleSampler(12,5,probability,backend=backend)
 sampler.run_mcmc(initial,3,progress=False)
 resumed=emcee.EnsembleSampler(12,5,probability,backend=backend)
 resumed.run_mcmc(None,2,progress=False)
 assert backend.iteration==5
 assert np.isfinite(backend.get_log_prob()).all()
 assert (backend.get_chain()[:,:,1]+backend.get_chain()[:,:,2]<0).all()
 results['smoke']={'walkers':12,'steps':5,'resumed':True,'finite':True,'mean_acceptance':float(np.mean(resumed.acceptance_fraction))}
Path('validation/cpl_results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
