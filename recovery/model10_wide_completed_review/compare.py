import sys,json
from pathlib import Path
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from Kosmulator_main.convergence import convergence_status
out=Path(__file__).parent;data={};report={};pars=['Omega_m','w','delta','H_0','M_abs']
for key,run in [('original','Model10_unrestricted_v1_iw'),('wide','Model10_wide_priors_v2_iw')]:
 p=next((root/'MCMC_Chains'/run).rglob('*.h5'))
 with h5py.File(p,'r') as f:
  n=int(f['mcmc'].attrs['iteration']);x=f['mcmc/chain'][:n];lp=f['mcmc/log_prob'][1000:n];acc=f['mcmc/accepted'][:]/n
 flat=x[1000:].reshape(-1,5);data[key]=flat
 moves=np.any(np.diff(x[-2000:],axis=0)!=0,axis=2).mean(axis=0)
 report[key]={'steps':n,'finite':bool(np.isfinite(flat).all() and np.isfinite(lp).all()),'acceptance':[float(acc.min()),float(acc.max())],'recent_movement':[float(moves.min()),float(moves.max())],'quantiles':np.quantile(flat,[.16,.5,.84],axis=0).tolist(),'mean':np.mean(flat,axis=0).tolist(),'Dbar_from_logprob':float(-2*np.mean(lp)),'corr_Om_delta':float(np.corrcoef(flat[:,0],flat[:,2])[0,1])}
 if key=='wide':
  report[key]['boundaries']={label:float(np.mean(mask)) for label,mask in [('delta_lt_minus1',flat[:,2]<-1),('delta_lt_minus1_9',flat[:,2]<-1.9),('delta_lt_minus1_95',flat[:,2]<-1.95),('Om_gt_0_5',flat[:,0]>.5),('Om_gt_0_65',flat[:,0]>.65)]}
  restricted=flat[(flat[:,2]>=-1)&(flat[:,2]<=1)&(flat[:,0]<=.5)]
  report[key]['restricted_to_old_bounds']=np.quantile(restricted,[.16,.5,.84],axis=0).tolist()
  report[key]['burn_sensitivity']={}
  for burn in [1000,3000,5000]:
   post=x[burn:];fpost=post.reshape(-1,5);st=convergence_status(post,{'tau_factor':100,'ess_min':2000,'rhat_max':1.01,'rtol':.05})
   report[key]['burn_sensitivity'][burn]={'median':np.median(fpost,axis=0).tolist(),'tau':st['tau_max'],'n_over_tau':st['n_over_tau'],'rhat':st['rhat_max'],'ESS':st['ess']}
  a,b=np.array_split(flat,2);report[key]['half_shift_SD']=((np.median(b,axis=0)-np.median(a,axis=0))/np.std(flat,axis=0)).tolist()
 del x,lp
fig,axs=plt.subplots(2,3,figsize=(12,7))
for i,ax in enumerate(axs.flat):
 if i<5:
  for key,col in [('original','#2878b5'),('wide','#d45c29')]:ax.hist(data[key][:,i],bins=90,density=True,histtype='step',lw=1.6,color=col,label=key)
  ax.set_xlabel(pars[i]);ax.set_ylabel('Posterior density')
 else:
  for key,col in [('original','#2878b5'),('wide','#d45c29')]:
   f=data[key];idx=np.random.default_rng(23).choice(len(f),3000,replace=False);ax.scatter(f[idx,2],f[idx,0],s=2,alpha=.15,color=col,rasterized=True)
  ax.axvline(-1,color='.4',ls='--');ax.axhline(.5,color='.4',ls='--');ax.set_xlabel('delta');ax.set_ylabel('Omega_m')
 ax.grid(alpha=.2)
axs[0,0].legend();fig.suptitle('Model 10: original versus wider priors (burn = 1000)');fig.tight_layout()
fig.savefig(out/'posterior_comparison.png',dpi=170);fig.savefig(out/'posterior_comparison.pdf')
(out/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
