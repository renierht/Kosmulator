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

import traceback
from Kosmulator_main import Post_processing as PP
from Plots.Plots import best_fit_plots
from Plots.Plot_functions import normalize_save_roots
out=root/'recovery/model10_postprocessing'
model='SignSwitch_IDE_2';cfg=configs[model];cfg['_postprocessing_switches']=flags.copy();cfg['postprocessing']=paper['postprocessing']
path=next((root/'MCMC_Chains/Model10_unrestricted_v1_iw').rglob('*.h5'))
with h5py.File(path,'r') as f:
 n=int(f['mcmc'].attrs['iteration']);chain=f['mcmc/chain'][1000:n];samples=chain.reshape(-1,chain.shape[-1])
key='DESI_DR2+PantheonP_SH0ES'
print('Read-only chain',n,'steps;',len(samples),'retained samples',flush=True)
print('Native parameter summaries and best-fit plotting',flush=True)
res,table,structured=PP.calculate_asymmetric_from_samples({key:samples},cfg['parameters'],cfg['observations'])
settings={'output_suffix':'Model10_unrestricted_v1_iw','save_root':'Plots/Saved_plots','latex_enabled':False}
normalize_save_roots(settings)
try:
 best_fit_plots({model:structured},configs,data,settings)
 print('Best-fit plots completed',flush=True)
except Exception:
 traceback.print_exc()
 (out/'bestfit_error.txt').write_text(traceback.format_exc())
print('Native statistical analysis with paper polishing settings',flush=True)
row=MC.analyse_group(samples,data,cfg,model,0,source=str(path),options=paper['postprocessing'])
(out/'model10_native_result.json').write_text(json.dumps(MC._json_safe({model:{key:row}}),indent=2)+'\n')
print('Native statistics complete',flush=True)
results=json.loads((root/'Statistical_analysis_tables/LCDM_CPL_Model9_comparison/model_comparison.json').read_text())
ref=results['LCDM_v'][key]
row['Nesting_check']=validate_nesting(model,'LCDM_v',row,ref,cfg,configs['LCDM_v'],data,0,0)
results[model]={key:row};MC.compare_models(results,'LCDM_v')
from Kosmulator_main.utils import save_stats_to_file
from Kosmulator_main.Plot_metadata import observation_label
for folder in [root/'Statistical_analysis_tables/LCDM_CPL_Model9_Model10_comparison',warm/'Statistical_analysis_tables/LCDM_CPL_Model9_Model10_comparison',root/'Statistical_analysis_tables/Model10_unrestricted_v1_iw']:
 MC.export_comparison(results,folder)
 for name,groups in results.items():
  dest=folder/name;dest.mkdir(exist_ok=True)
  save_stats_to_file(name,str(dest),[dict(r,Observation=observation_label(k)) for k,r in groups.items()])
 (folder/'PROVENANCE.md').write_text('Four completed chains; DESI DR2 + Pantheon+ SH0ES, burn=1000, native polished likelihood and posterior-mean DIC. All deltas reference LCDM. Model 10 reprocessed read-only; original completed corner plot preserved. Other three models use their completed native statistics. See recovery/model10_postprocessing for recovery log.\n')
print('Four-model tables exported',flush=True)
print((root/'Statistical_analysis_tables/LCDM_CPL_Model9_Model10_comparison/model_comparison.csv').read_text(),flush=True)
