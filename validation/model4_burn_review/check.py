"""Read-only snapshot burn sensitivity; never changes the live sampler."""
from pathlib import Path
import sys, shutil, json, numpy as np, h5py
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from Kosmulator_main.convergence import convergence_status,convergence_rules
out=Path(__file__).resolve().parent
source=next((root/'MCMC_Chains/Model4_EOS_CC_v2_model4_free_de_iw_cc').rglob('*.h5'))
snapshot=out/'snapshot.h5';shutil.copyfile(source,snapshot)
with h5py.File(snapshot,'r') as f:
 n=int(f['mcmc'].attrs['iteration']);x=f['mcmc/chain'][:n];lp=f['mcmc/log_prob'][:n]
names=json.loads(Path(str(source)+'.metadata.json').read_text())['parameters']
report={'source':str(source),'steps':n,'parameters':names,'finite':bool(np.isfinite(x).all() and np.isfinite(lp).all()),'burns':{}}
for burn in [1000,3000,4000]:
 prev=convergence_status(x[burn:n-100],convergence_rules(),param_names=names)
 status=convergence_status(x[burn:n],convergence_rules(),prev['tau_max'],names)
 flat=x[burn:].reshape(-1,len(names)); q=np.quantile(flat,[.16,.5,.84],axis=0)
 blocks=[np.median(b.reshape(-1,len(names)),axis=0) for b in np.array_split(x[burn:],4)]
 report['burns'][burn]={'status':status,'quantiles_16_50_84':q,'block_medians':blocks}
 print('burn',burn,{k:status[k] for k in ['tau_max','n_over_tau','ess','rhat_max','rel_change','criteria','ok']},flush=True)
 print('medians',q[1],flush=True)
def serial(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 raise TypeError(type(v).__name__)
(out/'report.json').write_text(json.dumps(report,default=serial,indent=2)+'\n')
