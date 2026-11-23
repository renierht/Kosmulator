"""Read-only bounded burn sensitivity; fixed thinning for a comparable diagnostic."""
from pathlib import Path
import json,sys,math
import h5py
import numpy as np
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root))
from postprocessing.evidence import estimate,temporal_subset,file_hash
p=Path(__file__).with_name('results.json'); saved=json.loads(p.read_text()); rows=[]
for run in saved['runs']:
    path=Path(run['chain']); assert file_hash(path)==run['chain_sha256']
    bounds=np.array([run['priors'][k] for k in run['parameters']])
    stride=math.ceil(2*run['retrospective_convergence']['tau_max'])
    for burn in (1000,2000,3000):
        with h5py.File(path,'r') as f:
            n=int(f['mcmc'].attrs['iteration']);x=f['mcmc/chain'][burn:n];ll=f['mcmc/blobs'][burn:n]
        zs=[]
        for seed in (101,202,303):
            a,b=temporal_subset(x,ll,stride,seed,1.,'all')
            zs.append(estimate(a,b,bounds)['logZ']['1'])
        rows.append(dict(regime=run['regime'],burn=burn,stride=stride,logZ_box=float(np.median(zs)),seed_values=zs))
    assert file_hash(path)==run['chain_sha256']
for r in rows:
    r['delta_logZ_box']=r['logZ_box']-next(b['logZ_box'] for b in rows if b['regime']=='lcdm' and b['burn']==r['burn'])
out=root/'Statistical_analysis_tables/Evidence_nonlinear_regimes_exploratory/burn_sensitivity.json'
out.write_text(json.dumps({'note':'Diagnostic only. Same stride as original burn=1000 estimate; each burn uses matching LCDM. Does not repair curved-boundary bias or validate convergence. Original files unchanged.','results':rows},indent=2)+'\n')
print(json.dumps(rows,indent=2))
