from pathlib import Path
import sys,json,pickle,numpy as np,logging
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from Kosmulator_main import Class_run as C,rd_helpers as R,Kosmulator_MCMC as M,constants as K
import User_defined_modules as U
out=Path(__file__).resolve().parent;r=json.loads((out/'snapshot_report.json').read_text())
settings=json.loads((root/'MCMC_Chains/Model4_EOS_CC_v1_model4_lambda_iw_cc/run_settings.json').read_text())
for k,v in settings['class_settings'].items():setattr(K,k,v)
U.ALLOW_NEGATIVE_ENERGIES=U.ALLOW_BIG_RIP=U.ALLOW_DOOM_FACTOR_INSTABILITIES=True
U.FREE_W=U.FREE_W_DM=False
assert C.ensure_class_ready('Linear_IDE_4',no_rebuild=True,announce=False)
with open(r['context'],'rb') as f:args=pickle.load(f)
logging.getLogger('Kosmulator.rd_helpers').setLevel(logging.CRITICAL)
Ctor=C.get_Class();epoch=None;errors=[]
class Proxy:
 def __init__(self):self.c=Ctor()
 def set(self,pars):
  pars=dict(pars)
  if epoch is not None:pars['a_ini_over_a_today_default']=epoch
  return self.c.set(pars)
 def compute(self):
  try:return self.c.compute()
  except Exception as e:errors.append(str(e));raise
 def __getattr__(self,k):return getattr(self.c,k)
C.get_Class=lambda:Proxy
rows=[]
main=np.array(r['main_median']);bad=np.array(r['walkers'][0]['last'])
points=[('main',main),('main_saved',np.array(r['main_last'])),('bad_saved',bad)]
points += [('bridge_'+str(i),main+(bad-main)*i/10) for i in range(1,10)]
points += [('delta_'+str(d),np.array([main[0],d,main[2],main[3]])) for d in (-.15,-.05,-.01,0.,.05,.1,.2,.3)]
for label,theta in points:
 for epoch in (None,1e-18,1e-24):
  R._rd_model_class_core.cache_clear();errors.clear()
  value=M.emcee_prob(theta,*args); val=float(value[0] if isinstance(value,tuple) else value)
  row=dict(label=label,theta=theta.tolist(),a_ini=epoch,logp=val if np.isfinite(val) else None,errors=list(errors))
  rows.append(row)
  (out/'likelihood_replay.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(label,epoch,val,flush=True)
