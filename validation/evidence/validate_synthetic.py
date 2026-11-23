"""Analytic, non-Gaussian, missed-mode, and correlation failure-mode checks."""
from pathlib import Path
import sys,json,numpy as np
from scipy.special import logsumexp
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from postprocessing.evidence import estimate
rng=np.random.default_rng(20261117);n=30000;results={}
for d in [3,5]:
 errors=[]
 for seed in [10,20,30]:
  x=np.random.default_rng(seed).normal(size=(n,d));ll=-.5*(x*x).sum(axis=1)
  truth=d*(.5*np.log(2*np.pi)-np.log(20))
  e=estimate(x,ll,np.tile([-10.,10.],(d,1)))
  errors.append(e['logZ']['1']-truth)
 assert max(abs(np.array(errors)))<.08
 results[f'gaussian_{d}d']={'errors':errors,'tolerance':.08}
# Banana transformation has unit Jacobian; integral unchanged on a very wide box.
x=rng.normal(size=(n,3));ll=-.5*(x*x).sum(axis=1);x[:,1]+=.4*(x[:,0]**2-1)
e=estimate(x,ll,np.tile([-20.,20.],(3,1)));truth=1.5*np.log(2*np.pi)-3*np.log(40)
results['curved_3d']={'error':e['logZ']['1']-truth};assert abs(e['logZ']['1']-truth)<.12
# Two separated equally weighted normalized Gaussian modes. Both are required.
u=rng.normal(size=(n,2));x=u.copy();x[:,0]+=rng.choice([-5.,5.],n)
logpdf=logsumexp(np.stack([-.5*((x-[a,0])**2).sum(axis=1)-np.log(2*np.pi)-np.log(2) for a in [-5,5]]),axis=0)
bounds=np.tile([-15.,15.],(2,1));truth=-2*np.log(30)
e=estimate(x,logpdf,bounds);assert abs(e['logZ']['1']-truth)<.1
one=x[:,0]>0;miss=estimate(x[one],logpdf[one],bounds)
results['two_modes']={'both_modes_error':e['logZ']['1']-truth,'missing_mode_error':miss['logZ']['1']-truth,'expected_missing_mode_error':-float(np.log(2))}
assert abs(miss['logZ']['1']-truth+np.log(2))<.1
# Repeated rejection states cannot be converted into unique iid draws by jitter.
try:estimate(np.repeat(u[:1000],3,axis=0),np.repeat(-.5*(u[:1000]**2).sum(axis=1),3),bounds)
except ValueError as exc:results['repeated_states']={'rejected':True,'reason':str(exc)}
else:raise AssertionError('Repeated states accepted')
# Hard support boundary is a known density-volume approximation limitation.
x=rng.uniform(size=(n,3));e=estimate(x,np.zeros(n),np.tile([0.,1.],(3,1)))
results['uniform_boundary']={'known_logZ':0.,'estimated_logZ':e['logZ'],'boundary_ball_fraction':e['neighbour_balls_crossing_box_fraction']}
(Path(__file__).parent/'synthetic_validation.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
