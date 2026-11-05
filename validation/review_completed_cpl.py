"""Reproduce completed-chain reports and exercise actual null likelihood checks."""
import os,sys,json,copy
from pathlib import Path
import numpy as np,h5py
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));os.chdir(root)
from Kosmulator_main import Config,constants as K,Class_run as CR,utils,Model_comparison as MC
from Kosmulator_main.significance import posterior_for_group,validate_nesting
import User_defined_modules as U
warm=root.parent/'Kosmulator_warm_dm';paper=json.loads((root/'reproducibility/SAIP2026_IDE/configs/paper.json').read_text())
for k,v in paper['class_settings'].items():setattr(K,k,v)
flags={k:True for k in ('ALLOW_NEGATIVE_ENERGIES','ALLOW_BIG_RIP','ALLOW_DOOM_FACTOR_INSTABILITIES')}
for k,v in flags.items():setattr(U,k,v)
models=['LCDM_v','SignSwitch_IDE_1','SignSwitch_IDE_2','w0waCDM']
priors={k:tuple(v) for k,v in paper['prior_limits'].items()};priors.update(w0=(-3,1),wa=(-3,2),r_d=(100,200))
configs,data=Config.create_config(U.Get_model_names(models),model_name=models,observation=paper['observations'],prior_limits=priors,burn=1000,free_w=True,free_w_dm=False)
data['PantheonPS']['cov']=utils.compute_pantheon_cov(data,configs['LCDM_v'],None,0,os.path.join(K.OBSERVATIONS_BASE,'PantheonP.cov'),obs_tag='PantheonPS')
from Kosmulator_main import convergence as CV
from Kosmulator_main.utils import save_stats_to_file
from Kosmulator_main.Plot_metadata import observation_label
p=next((root/'MCMC_Chains/CPL_comparison_v1_cpl').rglob('*.h5'))
report=json.loads(Path(str(p)+'.convergence.json').read_text())
with h5py.File(p,'r') as f:
 n=int(f['mcmc'].attrs['iteration']);chain=f['mcmc/chain'][:n];lp=f['mcmc/log_prob'][:n];accept=f['mcmc/accepted'][:]/n;flag=bool(f.attrs['converged'])
x=chain[1000:];names=configs['w0waCDM']['parameters'][0]
status=CV.convergence_status(x,report['rules'],report['history']['tau_max'][-2],names)
move=np.any(np.diff(x,axis=0)!=0,axis=2).mean(axis=0)
a,b=np.array_split(x,2);shift=(np.median(b.reshape(-1,5),axis=0)-np.median(a.reshape(-1,5),axis=0))/x.reshape(-1,5).std(axis=0)
audit=dict(steps=n,retained_samples=len(x)*120,converged_flag=flag,finite_chain=bool(np.isfinite(chain).all()),finite_logprob=bool(np.isfinite(lp).all()),acceptance_range=[float(accept.min()),float(accept.max())],postburn_move_range=[float(move.min()),float(move.max())],half_median_shift_sd=dict(zip(names,shift.tolist())),status=status)
results=json.loads((warm/'Statistical_analysis_tables/Model9_fresh_vs_LCDM_significance/model_comparison.json').read_text())
cpl=json.loads((root/'Statistical_analysis_tables/CPL_comparison_v1_cpl/model_comparison.json').read_text());results.update(cpl)
key=next(iter(cpl['w0waCDM']));row=results['w0waCDM'][key];ref=results['LCDM_v'][key]
assert all(r['Data_counts']==ref['Data_counts'] and r['Burn']==1000 and r['Thin']==1 for g in results.values() for r in g.values())
row['Nesting_check']=validate_nesting('w0waCDM','LCDM_v',row,ref,configs['w0waCDM'],configs['LCDM_v'],data,0,0)
MC.compare_models(results,'LCDM_v')
for base in [root,warm]:
 out=base/'Statistical_analysis_tables/LCDM_CPL_Model9_comparison';MC.export_comparison(results,out)
 for model,groups in results.items():
  dest=out/model;dest.mkdir(exist_ok=True)
  save_stats_to_file(model,str(dest),[dict(r,Observation=observation_label(k)) for k,r in groups.items()])
 (out/'cpl_chain_audit.json').write_text(json.dumps(audit,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item(),indent=2)+'\n')
print(json.dumps(audit,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item(),indent=2))
print((out/'model_comparison.csv').read_text())
print('CPL optimizer:',[(v['success'],v.get('chi2')) for v in row['Optimizer_runs']]);print('excluded',row['Excluded_samples'])
