"""Pinned upstream parity; compatibility shims affect validation process only."""
from pathlib import Path
import sys,json,importlib.util,numpy as np
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
import sklearn.neighbors,sklearn.metrics
# DistanceMetric moved in current sklearn; upstream's import is otherwise unused.
if not hasattr(sklearn.neighbors,'DistanceMetric'):
    sklearn.neighbors.DistanceMetric=sklearn.metrics.DistanceMetric
from postprocessing.evidence import estimate
spec=importlib.util.spec_from_file_location('upstream_mce',Path(__file__).parent/'upstream/MCEvidence.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=np.random.default_rng(72);x=rng.normal(size=(8000,3));x[:,1]+=.7*x[:,0]
u=x.copy();u[:,1]-=.7*u[:,0];ll=-.5*(u*u).sum(axis=1)
bounds=np.tile([-10.,10.],(3,1))
# Upstream chain arrays: weight, negative log-likelihood, sampled parameters.
raw=np.column_stack([np.ones(len(x)),-ll,x])
mc=m.MCEvidence([raw],kmax=4,priorvolume=8000.,verbose=0)
a=np.asarray(mc.evidence(verbose=0,nproc=1));b=np.array(list(estimate(x,ll,bounds)['logZ'].values()))
np.testing.assert_allclose(a,b,atol=1e-10,rtol=0)
report={'upstream':a.tolist(),'kosmulator':b.tolist(),'max_absolute_difference':float(abs(a-b).max()),'shims':['sklearn.neighbors.DistanceMetric -> sklearn.metrics.DistanceMetric'],'upstream_provenance':json.loads((Path(__file__).parent/'upstream/PROVENANCE.json').read_text())}
(Path(__file__).parent/'upstream_parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
