"""Bounded one-thread likelihood profiling, no sampling or production edits."""
import os,sys,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root));os.chdir(root)
from Kosmulator_main import Config,constants as K,Class_run as C,Kosmulator_MCMC as M
import User_defined_modules as U
out=Path(__file__).resolve().parent
paper=json.loads((root/'reproducibility/SAIP2026_IDE/configs/paper.json').read_text())
ref=json.loads((root/'reproducibility/SAIP2026_IDE/reference_results/model_comparison.json').read_text())
for k,v in paper['class_settings'].items():setattr(K,k,v)
for k in ['ALLOW_NEGATIVE_ENERGIES','ALLOW_BIG_RIP','ALLOW_DOOM_FACTOR_INSTABILITIES']:setattr(U,k,True)
results=[];start=time.monotonic()
for model,key in [('LCDM_v','LCDM'),('NonLinear_IDE_2','iwCDM')]:
 assert C.ensure_class_ready(model,no_rebuild=True,announce=False)
 cfgs,data=Config.create_config(U.Get_model_names([model]),model_name=[model],observation=[['PantheonPS','DESI_DR2','CC']],prior_limits=dict({k:tuple(v) for k,v in paper['prior_limits'].items()},r_d=(100,200)),free_w=True,free_w_dm=False)
 cfg=cfgs[model];names=cfg['parameters'][0];fn=U.Get_model_function(model)
 center=next(iter(ref[key].values()))['MLE_parameters'];x0=np.array([center[k] for k in names])
 scales=np.array([dict(Omega_m=.01,H_0=1.,M_abs=.03,w=.05,delta=.003)[k] for k in names]);bounds=np.array([paper['prior_limits'][k] for k in names])
 obs=cfg['observations'][0];types=cfg['observation_types'][0]
 def components(x):return {tag:-2*float(M.model_likelihood(x,data[tag],typ,cfg,fn,model,tag,0)) for tag,typ in zip(obs,types)}
 for use_cc in [False,True]:
  calls=[0];best=[float('inf'),None]
  def objective(y):
   x=x0+scales*y;calls[0]+=1
   if np.any(x<bounds[:,0]) or np.any(x>bounds[:,1]):return 1e90
   if not np.isfinite(M.log_prior_all(x[None,:],cfg,0)).all():return 1e90
   terms={tag:-2*float(M.model_likelihood(x,data[tag],typ,cfg,fn,model,tag,0)) for tag,typ in zip(obs,types) if use_cc or tag!='CC'}
   chi=sum(terms.values())
   if not np.isfinite(chi) or chi>1e80:return 1e90
   if chi<best[0]:best[:]=[chi,x.copy()]
   return chi
  initial=objective(np.zeros(len(names)))
  simplex=np.vstack([np.zeros(len(names)),np.eye(len(names))*.2])
  fit=minimize(objective,np.zeros(len(names)),method='Nelder-Mead',options=dict(maxfev=350,xatol=2e-4,fatol=1e-5,initial_simplex=simplex))
  x=best[1];terms=components(x)
  r=dict(model=model,include_cc=use_cc,parameters=dict(zip(names,x.tolist())),chi2=best[0],components=terms,initial_chi2=initial,success=bool(fit.success),message=str(fit.message),nfev=int(fit.nfev),simplex_objective_spread=float(np.ptp(fit.final_simplex[1])),simplex_scaled_parameter_spread=float(np.max(np.ptp(fit.final_simplex[0],axis=0))),k=len(names),N=1706 if use_cc else 1670)
  r['AIC']=r['chi2']+2*r['k'];r['AICc']=r['AIC']+2*r['k']*(r['k']+1)/(r['N']-r['k']-1);r['BIC']=r['chi2']+r['k']*np.log(r['N'])
  results.append(r);(out/'optimization_current.json').write_text(json.dumps(dict(results=results,elapsed_seconds=time.monotonic()-start,note='One-start bounded Nelder-Mead; diagnostic minima, not certified global optima. No posterior/DIC estimated.'),indent=2));print(json.dumps(r),flush=True)
