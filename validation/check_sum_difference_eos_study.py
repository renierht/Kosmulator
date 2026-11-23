"""Bounded numerical preflight, not an MCMC run or a chosen prior."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.integrate import solve_ivp
from Kosmulator_main import ide_background as I, rd_helpers as R, Class_run as C

a=np.geomspace(1.,1/1101.,151)
results=[]
for model, sign in ((2,1),(3,-1)):
    assert C.ensure_class_ready(f'Linear_IDE_{model}',no_rebuild=True,announce=False)
    for d in (0.,.001):
        for w,v in ((-1.,0.),(-.95,0.),(-1.,.005)):
            p=dict(Omega_m=.3,H_0=70.,w=w,w_dm=v,delta=d,__model_name__=f'Linear_IDE_{model}')
            dm,de=I.densities(model,a,p)
            def rhs(x,y): return [-3*(1+v)*y[0]+3*d*(y[0]+sign*y[1]),-3*(1+w)*y[1]-3*d*(y[0]+sign*y[1])]
            sol=solve_ivp(rhs,(0.,np.log(a[-1])),[.252,.7],t_eval=np.log(a),rtol=2e-11,atol=1e-13)
            assert sol.success
            err=float(np.max(np.abs(np.array([dm,de])-sol.y)/(np.abs(dm)+np.abs(de))))
            assert err<2e-8
            rd=R.compute_rd_model_class(p)
            assert np.isfinite(rd) and rd>0
            results.append(dict(model=model,delta=d,w=w,w_dm=v,ODE_scaled_error=err,rd=rd))
base=R.compute_rd_model_class(dict(Omega_m=.3,H_0=70.,__model_name__='LCDM_v'))
relative=max(abs(r['rd']/base-1) for r in results if r['delta']==0 and r['w']==-1 and r['w_dm']==0)
assert relative<1e-5,relative
assert abs(results[2]['rd']-results[0]['rd'])>.01
report=dict(cases=results,lcdm_rd=base,cold_zero_coupling_rd_relative=relative,
            note='Validation points only; no production prior or chains created.')
Path('validation/sum_difference_eos_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
