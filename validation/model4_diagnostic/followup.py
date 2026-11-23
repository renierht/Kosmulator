from pathlib import Path
exec(Path(__file__).with_name('replay.py').read_text().split('rows=[]')[0])
from scipy.optimize import minimize
outcomes={}
# Direct CLASS replay makes the exact error visible, independently of likelihood sentinels.
records=[]
for d in (-.15,-.05,-.002,.2):
 for epoch in (None,1e-18,1e-24):
  R._rd_model_class_core.cache_clear()
  p=dict(Omega_m=.301,H_0=72.,w=-1.,w_dm=0.,delta=d,__model_name__='Linear_IDE_4')
  row=dict(delta=d,a_ini=epoch)
  try:row['rd']=R.compute_rd_model_class(p)
  except Exception as e:row['error']=str(e)
  records.append(row)
outcomes['direct_class']=records
epoch=None;R._rd_model_class_core.cache_clear()
def prob(t):
 result=M.emcee_prob(t,*args);return float(result[0] if isinstance(result,tuple) else result)
# Test only the existing startup controls; priors stay unchanged.
config=args[2];names=r['names'];prior=config['prior_limits'][0]
ic=np.array([.307,0.,70.,-19.35])
cloud,*_=M.make_initial_positions(ic,prior,names,120,np.random.default_rng(20261115),base_frac=.002)
values=np.array([prob(t) for t in cloud])
outcomes['narrow_initialization']=dict(jitter_fraction=.002,center=ic.tolist(),tested=120,invalid=int(np.sum(values < -1e100)),logp_quantiles=np.quantile(values,[0,.16,.5,.84,1]).tolist())
# Bounded local searches, in scaled coordinates, including the observed poor region.
bounds=[prior[n] for n in names];scale=np.array([.03,.02,2.,.1]);fits=[]
for label,start in [('main',r['main_median']),('poor',r['walkers'][0]['last']),('dispersed',[.4,.03,74.,-19.25])]:
 start=np.array(start)
 def fn(y):
  t=start+scale*y
  if any(t[i]<lo or t[i]>hi for i,(lo,hi) in enumerate(bounds)):return 1e8
  v=prob(t);return -v if np.isfinite(v) and v>-1e100 else 1e8
 sol=minimize(fn,np.zeros(4),method='Nelder-Mead',options=dict(maxfev=100,xatol=1e-4,initial_simplex=np.vstack([np.zeros(4),np.eye(4)*.3])))
 fits.append(dict(start=label,theta=(start+scale*sol.x).tolist(),logp=-float(sol.fun),evaluations=sol.nfev,optimizer_converged=bool(sol.success)))
 print('fit',fits[-1],flush=True)
 outcomes['bounded_fits']=fits
 (out/'followup_report.json').write_text(json.dumps(outcomes,indent=2)+'\n')
print(json.dumps(outcomes,indent=2))
