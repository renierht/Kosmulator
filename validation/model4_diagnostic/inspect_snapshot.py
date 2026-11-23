from pathlib import Path
import h5py,shutil,json,numpy as np,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from Kosmulator_main.convergence import split_rhat
out=Path(__file__).resolve().parent
source=next((root/'MCMC_Chains/Model4_EOS_CC_v1_model4_lambda_iw_cc').rglob('*.h5'))
snapshot=out/'snapshot.h5';shutil.copyfile(source,snapshot)
with h5py.File(snapshot,'r') as f:
 g=f['mcmc'];end=int(g.attrs['iteration'])-1;x=g['chain'][:end];lp=g['log_prob'][:end]
names=json.loads(Path(str(source)+'.metadata.json').read_text())['parameters']
last=lp[-1];best=float(last.max());bad=np.flatnonzero(last<best-50)
report=dict(step=end,names=names,rhat_all=split_rhat(x[1000:]).tolist(),best_last_logp=best,bad_count=len(bad),walkers=[])
for i in bad:
 report['walkers'].append(dict(id=int(i),initial=x[0,i].tolist(),last=x[-1,i].tolist(),initial_logp=float(lp[0,i]),last_logp=float(lp[-1,i]),best_logp=float(lp[:,i].max()),moves_last1000=int(np.any(np.diff(x[-1001:,i],axis=0)!=0,axis=1).sum()),block_median_logp=[float(np.median(a)) for a in np.array_split(lp[1000:,i],4)]))
good=last>=best-50
report['rhat_diagnostic_excluding_outliers_NOT_results']=split_rhat(x[1000:,good]).tolist()
report['main_median']=np.median(x[-1000:,good].reshape(-1,len(names)),axis=0).tolist()
report['main_last']=x[-1,int(np.argmax(last))].tolist()
report['source']=str(source);report['context']=str(next(source.parent.glob('.kosm_pool_args_*.pkl')))
(out/'snapshot_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
