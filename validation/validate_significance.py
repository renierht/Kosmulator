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
for name in models:
 configs[name]['_postprocessing_switches']=flags.copy()
 assert CR.ensure_class_ready(name,no_rebuild=False,announce=True), name
results=json.loads((warm/'Statistical_analysis_tables/Model9_fresh_vs_LCDM/model_comparison.json').read_text());key=next(iter(results['LCDM_v']));ref=results['LCDM_v'][key]
checks={}
for name in models[1:]:
 cfg=configs[name]
 if name=='SignSwitch_IDE_1':row=results[name][key]
 else:
  # Only exercising null evaluations here, NOT a production fit or significance.
  row=copy.deepcopy(ref);row['MLE_parameters']={**ref['MLE_parameters'],**({'w0':-1.,'wa':0.} if name=='w0waCDM' else {'w':-1.,'delta':0.})};row['Physicality_switches']=flags.copy()
 checks[name]=validate_nesting(name,'LCDM_v',row,ref,cfg,configs['LCDM_v'],data,0,0)
 print(name,checks[name],flush=True)
 if name=='SignSwitch_IDE_1':row['Nesting_check']=checks[name]
for name,suffix in [('LCDM_v','Model9_unrestricted_20261014_iw'),('SignSwitch_IDE_1','Model9_fresh_20261016_iw')]:
 path=next((warm/'MCMC_Chains'/suffix/name).rglob('*.h5'))
 with h5py.File(path,'r') as f:
  n=int(f['mcmc'].attrs['iteration']);chain=f['mcmc/chain'][1000:n];samples=chain.reshape(-1,chain.shape[-1])
 results[name][key]['Posterior_preferences']=posterior_for_group(samples,configs[name],0,str(path))
MC.compare_models(results,'LCDM_v')
out=warm/'Statistical_analysis_tables/Model9_fresh_vs_LCDM_significance';MC.export_comparison(results,out)
(root/'validation/significance_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print('OUTPUT',out,flush=True)
