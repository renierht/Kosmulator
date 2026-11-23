"""Read-only comparison of native IC summaries and saved likelihood blobs."""
import json,csv
from pathlib import Path
import h5py,numpy as np
root=Path(__file__).resolve().parents[2]
folders={
'LCDM':'Model5_EOS_CC_v1_lcdm_iw_cc',
'Uncoupled, free w_dm':'Model5_EOS_CC_v1_warm_lambda_iw_cc',
'Model 4, fixed EOS':'Model4_EOS_CC_v2_model4_lambda_iw_cc',
'Model 4, free w_de':'Model4_EOS_CC_v2_model4_free_de_burn3000_iw_cc',
'Model 4, free w_dm':'Model4_EOS_CC_v2_model4_warm_lambda_postprocess_iw_cc'}
rows=[];manifests=[];details={}
for label,folder in folders.items():
 source=root/'Statistical_analysis_tables'/folder/'model_comparison.json'
 d=json.loads(source.read_text());v=next(iter(next(iter(d.values())).values()))
 chain=Path(v['Likelihood_source'].removeprefix('saved likelihood blobs: '))
 manifest=next(p/'run_settings.json' for p in chain.parents if (p/'run_settings.json').exists())
 m=json.loads(manifest.read_text());manifests.append(m)
 assert v['N']==1706 and v['Finite_samples']==v['Retained_samples'] and v['Excluded_samples']==0
 with h5py.File(chain,'r') as f:
  n=int(f['mcmc'].attrs['iteration']);ll=f['mcmc/blobs'][v['Burn']:n:v['Thin']].reshape(-1)
  assert len(ll)==v['Retained_samples'] and np.isfinite(ll).all()
  dev=-2*ll;assert abs(dev.mean()-v['D_bar'])<1e-7
  if label=='Model 4, free w_dm':
   x=f['mcmc/chain'][v['Burn']:n:v['Thin']].reshape(-1,5)
 r={ 'Model':label,**{k:v[k] for k in ['k','p_D','Chi_squared','AIC','AICc','DIC','BIC','Burn']},'pV':float(dev.var(ddof=0)/2),'DIC_variance':float(dev.mean()+dev.var(ddof=0)/2),'source':str(source)}
 rows.append(r)
for m in manifests[1:]:
 for k in ['dataset_hashes','observations','class_settings','distance_policy','likelihood_policy']:
  assert m[k]==manifests[0][k],k
for r in rows:
 for k in ['Chi_squared','AIC','AICc','DIC','BIC','DIC_variance']:r['Delta_'+k]=r[k]-rows[0][k]
s=x[:,1]-x[:,2];A=(x[:,0]-.048)*(1+x[:,1])/(1+s)
for name,arr in [('w_dm_minus_delta',s),('dark_scaling_amplitude',A)]:
 q=np.percentile(arr,[16,50,84]);details[name]={'median':float(q[1]),'lower_error':float(q[1]-q[0]),'upper_error':float(q[2]-q[1])}
details['corr_w_dm_delta']=float(np.corrcoef(x[:,1],x[:,2])[0,1])
details['w_dm_outer_10_percent_fraction']=float(np.mean(abs(x[:,1])>.135))
out=root/'Statistical_analysis_tables/Model4_EOS_CC_comparison';out.mkdir(exist_ok=True)
(out/'comparison.json').write_text(json.dumps({'rows':rows,'degeneracy':details,'notes':['Native posterior-mean DIC retained; variance DIC uses Dbar+Var(D)/2 and population variance.','Original per-run burns retained: 1000 for older runs, 3000 for free-DE and free-DM Model 4.','AIC/BIC use nominal sampled parameter counts; near-nonidentifiability limits their regular asymptotic interpretation.','Dataset hashes, observations, CLASS settings, distance and likelihood policies match.']},indent=2))
with (out/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
lines=['# Model 4 comparison with matched data\n','Pantheon+SH0ES + DESI DR2 + CC; N=1706. Differences are model minus LCDM.\n','| Model | k | chi2 min | Delta AIC | Delta AICc | Delta DIC mean | Delta BIC | Delta DIC variance |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:lines.append('| '+r['Model']+' | '+str(r['k'])+' | '+' | '.join(f'{r[k]:.3f}' for k in ['Chi_squared','Delta_AIC','Delta_AICc','Delta_DIC','Delta_BIC','Delta_DIC_variance'])+' |')
lines+=['\nThe warm Model 4 fit has correlation(w_dm,delta)='+str(details['corr_w_dm_delta'])+'.', '\nFor w_de=-1, define s=w_dm-delta. The conservation equations imply rho_dm=rho_dm0 a^[-3(1+s)] and rho_de=rho_de0+delta rho_dm0/(1+s) (a^[-3(1+s)]-1). Thus the total dark density depends on s and amplitude A=rho_dm0(1+w_dm)/(1+s), rather than independently on all three original dark-sector parameters. This explains the background degeneracy when Omega_m can compensate. CLASS thermodynamics need not preserve an exact degeneracy.', '\nOriginal burns and native IC parameter counts are preserved. The nearly redundant parameter means standard AIC/BIC regularity assumptions are questionable; report nominal-count values with this caveat. Mean-based DIC is parameterization-sensitive. Variance-based DIC is provided separately, not substituted silently. Grid warnings and rough KDE rendering do not change sample-based intervals or saved likelihood statistics.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:10]));print(json.dumps(details,indent=2))
