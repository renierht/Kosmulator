"""Sparse retrospective convergence plot, using the native plotter and policy."""
from pathlib import Path
import sys,json,h5py
from types import SimpleNamespace
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from Kosmulator_main.convergence import convergence_status,convergence_rules
from Plots.Plots import convergence_plot
r=json.loads((Path(__file__).parent/'report.json').read_text());n=r['steps']
with h5py.File(r['source'],'r') as f:x=f['mcmc/chain'][:n]
h={k:[] for k in ['it','tau_max','rel','rhat_max','ok']};rules=convergence_rules()
for end in [10000,15000,n-100,n]:
 prev=convergence_status(x[3000:end-100],rules,param_names=r['parameters'])
 s=convergence_status(x[3000:end],rules,prev['tau_max'],r['parameters'])
 for k,v in [('it',end),('tau_max',s['tau_max']),('rel',s['rel_change']),('rhat_max',s['rhat_max']),('ok',s['ok'])]:h[k].append(v)
m=SimpleNamespace(hist=h,pre=None,rules=rules,burn=3000,walkers=120,earliest_stop=3100,consecutive=1,stopped_at=None,ended_at=n,streak=int(h['ok'][-1]))
out=root/'Plots/Saved_plots/Model4_EOS_CC_v2_model4_free_de_burn3000_iw_cc/Linear_IDE_4/auto_corr/retrospective_burn3000.png'
convergence_plot(m,str(out),'Linear_IDE_4','retrospective burn-3000 review',{'dpi':200})
(Path(__file__).parent/'retrospective_plot_history.json').write_text(json.dumps(h,indent=2)+'\n')
print(out)
