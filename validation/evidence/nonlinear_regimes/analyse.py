"""Explicit legacy-chain audit and two prior conventions; no chain modifications."""
from pathlib import Path
import sys,json,math,numpy as np,h5py
from scipy.integrate import quad
from scipy.stats import qmc
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root))
from postprocessing.evidence import estimate,temporal_subset,file_hash
from Kosmulator_main.convergence import convergence_status,convergence_rules
out=Path(__file__).resolve().parent
archive=Path('/mnt/c/Users/Marcel/PhD/Kosmulator_test_Class/Kosmulator/MCMC_Chains')
original=Path('/mnt/c/Users/Marcel/PhD/Kosmulator_test_Class/Kosmulator_referee_clean')
config_path=original/'reproducibility/SAIP2026_IDE/configs/paper.json'
cfg=json.loads(config_path.read_text());ref=json.loads((root/'reproducibility/SAIP2026_IDE/reference_results/model_comparison.json').read_text())
labels={'lcdm':'LCDM','iw':'iwCDM','plus_iw':'+iwCDM','siw':'SiwCDM'}


def support(om,w,delta,regime):
    r=(om-.048)/(1-om)
    ok=(delta>w)&(delta<=-w/r)
    if regime=='plus_iw':ok &=delta>=0
    if regime=='siw':ok &=w < -1
    return ok


def fraction(regime):
    if regime=='lcdm':return 1.,0.
    # Piecewise exact inner integrals: delta width is affine in w per segment.
    def area(om):
        r=(om-.048)/(1-om);lo,hi=-2.,(-1. if regime=='siw' else -.33)
        breaks=sorted(set([lo,hi]+[p for p in [-1.,-r] if lo<p<hi]))
        def width(w):return max(0.,min(1.,-w/r)-max(-1.,w,0. if regime=='plus_iw' else -1.))
        return sum((b-a)*(width(a)+width(b))/2 for a,b in zip(breaks[:-1],breaks[1:]))
    val,err=quad(area,.1,.5,epsabs=1e-12,epsrel=1e-12,points=[(.33+.048)/1.33])
    return val/(.4*1.67*2),err/(.4*1.67*2)


volumes={}
points=qmc.Sobol(3,scramble=True,seed=817).random_base2(20)
om=.1+.4*points[:,0];w=-2+1.67*points[:,1];delta=-1+2*points[:,2]
for regime in cfg['cases']:
 f,err=fraction(regime);sobol=1. if regime=='lcdm' else float(support(om,w,delta,regime).mean())
 assert abs(f-sobol)<1e-4,(regime,f,sobol)
 volumes[regime]={'allowed_fraction':f,'quadrature_error':err,'sobol_fraction':sobol,'lnZ_normalization_shift':-math.log(f)}
(out/'prior_volumes.json').write_text(json.dumps(volumes,indent=2)+'\n')
results=[]
for regime,case in cfg['cases'].items():
 p=archive/case['chain'];print('Reading',regime,flush=True)
 before=file_hash(p)
 with h5py.File(p,'r') as f:
  g=f['mcmc'];n=int(g.attrs['iteration']);x=g['chain'][cfg['burn']:n];ll=g['blobs'][cfg['burn']:n];lp=g['log_prob'][cfg['burn']:n];old_converged=bool(f.attrs.get('converged',False))
 assert n==case['completed_steps'] and ll.size==case['retained_samples']
 assert np.isfinite(x).all() and np.isfinite(ll).all() and np.all(np.abs(ll)<1e90)
 np.testing.assert_allclose(lp,ll,rtol=0,atol=1e-8)
 row=next(iter(ref[labels[regime]].values()));assert abs(float((-2*ll).mean())-row['D_bar'])<1e-7
 bounds=np.array([cfg['prior_limits'][k] for k in case['parameters']])
 assert np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1])
 if regime!='lcdm':assert support(x[:,:,0],x[:,:,1],x[:,:,2],regime).all()
 prev=convergence_status(x[:-100],convergence_rules(),param_names=case['parameters'])
 st=convergence_status(x,convergence_rules(),prev['tau_max'],case['parameters'])
 tau=st['tau_max'];trials=[]
 for factor,frac,block in [(f,s,'all') for f in [1,2,5] for s in [.5,1.]]+[(2,1.,'first'),(2,1.,'second')]:
  for seed in [101,202,303]:
   trial=dict(tau_factor=factor,fraction=frac,block=block,seed=seed,stride=int(np.ceil(factor*tau)))
   try:
    a,b=temporal_subset(x,ll,trial['stride'],seed,frac,block)
    trial.update(estimate(a,b,bounds));trial['valid']=True
   except ValueError as exc:trial.update(valid=False,error=str(exc))
   trials.append(trial)
 primary=[t['logZ']['1'] for t in trials if t['valid'] and t['tau_factor']==2 and t['fraction']==1 and t['block']=='all']
 if len(primary)!=3:raise ValueError('Primary evidence trials failed; inspect thinning')
 values=[z for t in trials if t['valid'] for z in t['logZ'].values()]
 f=volumes[regime]['allowed_fraction'];logz=float(np.median(primary));env=[min(values),max(values)]
 assert file_hash(p)==before,'Source changed during analysis'
 r=dict(regime=regime,label=labels[regime],chain=str(p),chain_sha256=before,steps=n,burn=cfg['burn'],parameters=case['parameters'],priors={k:cfg['prior_limits'][k] for k in case['parameters']},
  original_converged_attribute=old_converged,retrospective_convergence=st,
  logZ_box=logz,logZ_joint_uniform_allowed=logz-math.log(f),box_envelope=env,
  allowed_prior_fraction=f,normalization_shift=-math.log(f),trials=trials,
  status='provisional' if st['ok'] else 'exploratory: fails current convergence policy',
  warnings=['Original runs lack modern metadata/convergence sidecars; config and saved D_bar were checked against frozen paper reproduction artifacts.',
  'Joint-uniform mathematical-region prior is a different model definition from a box prior with zero likelihood outside the region.',
  'CLASS numerical-failure exclusions are retained as zero likelihood, not included in the prior-volume normalization.',
  'kNN balls can cross internal mathematical/positive/stability boundaries; box-crossing diagnostics do not capture these boundaries. No boundary correction applied.',
  'Sensitivity envelopes are not uncertainty intervals; independent ensembles unavailable.'])
 results.append(r)
 print(regime,'Rhat',st['rhat_max'],'criteria',st['criteria'],'box lnZ',logz,'allowed lnZ',r['logZ_joint_uniform_allowed'],flush=True)

def serial(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 raise TypeError(type(v).__name__)
base=results[0]
for r in results:
 r['delta_logZ_box']=r['logZ_box']-base['logZ_box'];r['delta_logZ_joint_uniform_allowed']=r['logZ_joint_uniform_allowed']-base['logZ_box']
 r['delta_box_sensitivity_envelope']=[r['box_envelope'][0]-base['box_envelope'][1],r['box_envelope'][1]-base['box_envelope'][0]]
 if r is base:r['delta_box_sensitivity_envelope']=[0.,0.]
(out/'results.json').write_text(json.dumps({'config':str(config_path),'config_sha256':file_hash(config_path),'prior_volumes':volumes,'runs':results},default=serial,indent=2)+'\n')
print('Completed',out,flush=True)
