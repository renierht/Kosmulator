"""Bounded Model 1 preflight; no chains or production physics modified."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
import numpy as np
from scipy.integrate import solve_ivp
import User_defined_modules as U
from Kosmulator_main import Config, constants as K, ide_background as I, rd_helpers as R, Class_run as C
raw=Path('validation/model1_noCC_plans.jsonlog').read_text(); plans=[]
decoder=json.JSONDecoder()
while raw.strip():
 obj,end=decoder.raw_decode(raw.lstrip());plans.append(obj);raw=raw.lstrip()[end:]
assert len(plans)==3
checks=[]
for plan in plans:
 assert plan['observations']==[['PantheonPS','DESI_DR2']]
 assert plan['priors']['delta_dm']==[-1.,1.] and plan['priors']['delta_de']==[-1.,1.]
 assert 'delta' not in plan['priors']
 with patch.object(Config,'load_all_data',return_value={}),patch.object(K,'DERIVE_RD_WITH_MODEL_CLASS',True):
  cfg,_=Config.create_config(U.Get_model_names(['Linear_IDE_1']),prior_limits=dict(plan['priors'],r_d=(130,170)),observation=plan['observations'],model_name=['Linear_IDE_1'],free_w=plan['free_w'],free_w_dm=plan['free_w_dm'])
 c=cfg['Linear_IDE_1'];assert set(c['parameters'][0])==set(plan['priors'])
 assert c['ndim'][0]==5+int(plan['free_w'] or plan['free_w_dm'])
 checks.append(dict(case=plan['output_suffix'],parameters=c['parameters'][0],fixed=c['fixed_parameters']))
assert C.ensure_class_ready('Linear_IDE_1',no_rebuild=True,announce=False)
a=np.geomspace(1.,1/1101.,151);results=[]
for d,e in [(0.,0.),(.001,0.),(0.,.001),(.001,-.001)]:
 for w,v in [(-1.,0.),(-.95,0.),(-1.,.005)]:
  p=dict(Omega_m=.3,H_0=70.,w=w,w_dm=v,delta_dm=d,delta_de=e,__model_name__='Linear_IDE_1')
  dm,de=I.densities(1,a,p)
  def rhs(t,y):
   q=3*(d*y[0]+e*y[1]);return [-3*(1+v)*y[0]+q,-3*(1+w)*y[1]-q]
  sol=solve_ivp(rhs,(0.,np.log(a[-1])),[.252,.7],t_eval=np.log(a),rtol=2e-11,atol=1e-13)
  assert sol.success
  err=float(np.max(np.abs(np.array([dm,de])-sol.y)/(abs(dm)+abs(de))))
  assert err<2e-8
  rd=R.compute_rd_model_class(p);assert np.isfinite(rd) and rd>0
  results.append(dict(delta_dm=d,delta_de=e,w=w,w_dm=v,ODE_scaled_error=err,rd=rd))
base=R.compute_rd_model_class(dict(Omega_m=.3,H_0=70.,__model_name__='LCDM_v'))
assert abs(results[0]['rd']/base-1)<1e-5
assert abs(results[3]['rd']-results[0]['rd'])>1e-3
assert abs(results[6]['rd']-results[0]['rd'])>1e-3
out=dict(config_checks=checks,numerical_checks=results,lcdm_rd=base)
Path('validation/model1_eos_preflight.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS: three parameter configurations; 12 independent ODE/CLASS checks; max scaled ODE error',max(r['ODE_scaled_error'] for r in results))
