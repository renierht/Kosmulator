"""Independent ratio/total conservation ODE and proper-time tail checks.

These finite numerical checks complement the analytic asymptotic derivation;
they are not an exhaustive proof over all real parameters.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.integrate import solve_ivp
from Kosmulator_main import ide_background as I

def evolution(m,p):
    w,v,d=p['w'],p['w_dm'],p['delta']
    def rhs(x,y):
        f,L=y  # DM fraction of dark total, and log dark total
        if m==1: q=3*(p['delta_dm']*f+p['delta_de']*(1-f))
        elif m==2: q=3*d
        elif m==3: q=3*d*(2*f-1)
        elif m==4: q=3*d*f
        elif m==5: q=3*d*(1-f)
        elif m==6: q=3*d*f*(1-f)
        elif m==7: q=3*d*f*f
        elif m==8: q=3*d*(1-f)**2
        else:
            q=1.5*d*((1+3*v)*f+(1+3*w)*(1-f))*(f if m==9 else 1-f)
        return [q+3*(w-v)*f*(1-f),-3*(1+w+(v-w)*f)]
    return rhs

p0=dict(Omega_m=.307,Omega_b=.048,Omega_r=8e-5,w=-1.1,w_dm=.025,
        delta=.04,delta_dm=.03,delta_de=.02)
cases=[]
for m in range(1,11):
    cases.append((m,dict(p0)))
    cases.append((m,dict(p0,w=-.85)))
cases += [(1,dict(p0,delta_dm=.05,delta_de=.2)),
          (2,dict(p0,delta=.2)),(5,dict(p0,delta=.2)),
          (6,dict(p0,delta=2.)),(8,dict(p0,delta=.2)),
          (9,dict(p0,delta=-1.3)),(10,dict(p0,delta=-.2))]
results=[]
for m,p in cases:
    total=1-p['Omega_b']-p['Omega_r']
    y0=[(p['Omega_m']-p['Omega_b'])/total,np.log(total)]
    xs=np.linspace(0,60,3001)
    f=evolution(m,p)
    sol=solve_ivp(f,(0,60),y0,t_eval=xs,method='DOP853',rtol=1e-11,atol=1e-12,max_step=.1)
    assert sol.success,(m,p,sol.message)
    slope=f(60,sol.y[:,-1])[1]
    expected=I.future_behavior(m,p)
    assert expected in ('big_rip','no_big_rip'),(m,p,expected)
    assert (slope>1e-6)==(expected=='big_rip'),(m,p,slope,expected)
    # H^2/H0^2 includes the independent baryon and radiation terms.
    logH2=np.logaddexp(sol.y[1],np.logaddexp(np.log(p['Omega_b'])-3*xs,
                                          np.log(p['Omega_r'])-4*xs))
    dt=np.exp(-logH2/2)
    early=np.trapezoid(dt[(xs>=30)&(xs<=40)],xs[(xs>=30)&(xs<=40)])
    late=np.trapezoid(dt[(xs>=50)&(xs<=60)],xs[(xs>=50)&(xs<=60)])
    assert (late<early)==(expected=='big_rip'),(m,expected,early,late)
    results.append(dict(model=m,w=p['w'],w_dm=p['w_dm'],delta=p['delta'],
                        delta_dm=p['delta_dm'],delta_de=p['delta_de'],
                        classification=expected,numerical_log_density_slope=slope,
                        late_to_early_proper_time_tail=late/early))

# Model 10 delta>2/3: locate the finite-a endpoint with an independent ODE.
p=dict(p0,delta=.8)
def endpoint(x,y): return y[0]+1e6
endpoint.terminal=True; endpoint.direction=-1
total=1-p['Omega_b']-p['Omega_r']
sol=solve_ivp(evolution(10,p),(0,10),[(p['Omega_m']-p['Omega_b'])/total,np.log(total)],
              events=endpoint,method='DOP853',rtol=1e-10,atol=1e-11,max_step=.01)
assert len(sol.t_events[0])==1
u=p['w']-p['w_dm'];r=(p['Omega_m']-p['Omega_b'])/(1-p['Omega_m']-p['Omega_r'])
k=1.5*(2*u+p['delta']*(1+3*p['w_dm']));b=1.5*p['delta']*(1+3*p['w'])
D=1-b/k;coefficient=r+b/k
endpoint_exact=np.log(-D/coefficient)/k
assert abs(sol.t_events[0][0]-endpoint_exact)<1e-5
assert I.future_behavior(10,p)=='finite_scale_singularity'
results.append(dict(model=10,delta=.8,classification='finite_scale_singularity',
                    analytic_ln_a=endpoint_exact,numerical_ln_a=float(sol.t_events[0][0])))
Path('validation/warm_future_results.json').write_text(json.dumps(results,indent=2)+'\n')
print(f'PASS: {len(cases)} asymptotic/proper-time-tail cases and finite-scale endpoint')
