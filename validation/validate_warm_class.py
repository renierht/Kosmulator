"""Full compiled CLASS validation; run from repository root in Kosmulator_test."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from Kosmulator_main import Class_run as C, ide_background as I, rd_helpers as RD

assert C.ensure_class_ready('SignSwitch_IDE_1',announce=False)
Class=C.get_Class()
results=[]
for v in (0.,.025):
    for m in range(1,11):
        c=Class()
        pars=dict(h=.7,Omega_b=.048,Omega_cdm=.259,N_ur=3.044,N_ncdm=0,
                  Omega_Lambda=0.,w0_fld=-1.08,wa_fld=0.,ide_model=m,w_dm_ide=v,
                  delta_ide=-.04 if m==10 else .04,delta_dm_ide=.03,delta_de_ide=.02,output='')
        try:
            c.set(pars);c.compute(); b=c.get_background()
            h02=c.Hubble(0)**2
            radiation=(b['(.)rho_g'][-1]+b['(.)rho_ur'][-1])/h02
            p=dict(Omega_m=.307,Omega_b=.048,Omega_r=radiation,w=-1.08,w_dm=v,
                   delta=pars['delta_ide'],delta_dm=.03,delta_de=.02)
            a=1/(1+b['z']); dm,de=I.densities(m,a,p)
            total=dm+de+.048/a**3+radiation/a**4
            eh=np.sqrt(total)*c.Hubble(0)
            h_error=float(np.max(abs(b['H [1/Mpc]']/eh-1)))
            scale=np.maximum(abs(dm)+abs(de),1e-30)
            density_error=float(np.max(np.maximum(abs(b['(.)rho_cdm']/h02-dm),abs(b['(.)rho_fld']/h02-de))/scale))
            expected_pressure=(radiation/a**4/3+v*dm+p['w']*de)*h02
            pressure_error=float(np.max(abs(b['(.)p_tot']-expected_pressure)/(b['(.)rho_tot'])))
            # Independent finite derivative of total pressure verifies both
            # conservation terms in p'_tot, including the warm-DM contribution.
            numerical_dp=np.gradient(expected_pressure,np.log(a),edge_order=2)
            code_dp=b['(.)p_tot_prime']/(a*b['H [1/Mpc]'])
            pressure_derivative_error=float(np.max(abs(code_dp-numerical_dp)/b['(.)rho_tot']))
            assert h_error<2e-9,(m,v,h_error)
            assert density_error<2e-9,(m,v,density_error)
            assert pressure_error<2e-9,(m,v,pressure_error)
            assert pressure_derivative_error<1e-5,(m,v,pressure_derivative_error)
            rd=float(c.rs_drag());assert np.isfinite(rd) and rd>0
            results.append(dict(model=m,w_dm=v,H_relative=h_error,density_scaled=density_error,
                                pressure_scaled=pressure_error,pressure_derivative_scaled=pressure_derivative_error,
                                rs_drag=rd,points=len(a)))
        finally:
            c.struct_cleanup();c.empty()
    # All ten zero-coupling selectors must give the same warm/cold rs_drag.
    reference=None
    for m in range(1,11):
        c=Class()
        try:
            c.set(dict(h=.7,Omega_b=.048,Omega_cdm=.259,N_ur=3.044,N_ncdm=0,
                       Omega_Lambda=0.,w0_fld=-1.,ide_model=m,w_dm_ide=v,
                       delta_ide=0.,delta_dm_ide=0.,delta_de_ide=0.,output=''))
            c.compute();rd=float(c.rs_drag())
            if reference is None:reference=rd
            assert abs(rd/reference-1)<1e-7,(m,v,rd,reference)
        finally: c.struct_cleanup();c.empty()
# Exercise production routing with its massive-neutrino calibration settings.
for name in I.MODEL_NAMES:
    m=I.SELECTORS[name]
    p=dict(__model_name__=name,Omega_m=.307,H_0=70,w=-1.,w_dm=.01,
           delta=-.02 if m==10 else .02,delta_dm=.01,delta_de=.01)
    rd=RD.compute_rd_model_class(p)
    assert np.isfinite(rd) and rd>0
    results.append(dict(route=name,rs_drag=rd))
Path('validation/warm_class_results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
