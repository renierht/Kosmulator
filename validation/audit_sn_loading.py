"""Real-file geometry and covariance audit; no sampling or CLASS calls."""
import sys,json,logging
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from Kosmulator_main import Config,utils
import User_defined_modules as U

loaded=Config.load_all_data({'observations':[['DESY5'],['Union3']]},logger=logging.getLogger(__name__))
frame=pd.read_csv('Observations/DESY5.dat')
des=loaded['DESY5']
np.testing.assert_array_equal(des['redshift'],frame.zHD.to_numpy())
np.testing.assert_array_equal(des['z_hel'],frame.zHEL.to_numpy())
report={}
for name in ('DESY5','Union3'):
    obs=loaded[name]; n=len(obs['redshift'])
    file='DESY5_covsys_000.txt' if name=='DESY5' else 'Union3_mag_covmat.txt'
    raw=np.loadtxt(Path('Observations')/file)[1:].reshape(n,n)
    expected=(raw+raw.T)/2
    if name=='DESY5': expected+=np.diag(obs['type_data_error']**2)
    np.testing.assert_array_equal(obs['cov'],expected)
    residual=obs['type_data']*.001
    q1=float(residual@obs['inv_cov']@residual)
    q2=float(residual@np.linalg.solve(expected,residual))
    np.testing.assert_allclose(q1,q2,rtol=1e-10)
    report[name]=dict(rows=n,covariance_matches_explicit_recipe=True,
        quadratic_relative_difference=abs(q1/q2-1))
model=U.Get_model_function('LCDM_v'); p=dict(Omega_m=.3,H_0=70)
old_dl=utils.Comoving_distance_vectorized(model,frame.zCMB.to_numpy(),p)*(1+frame.zCMB.to_numpy())
new_dl=utils.Comoving_distance_vectorized(model,des['redshift'],p)*(1+des['z_hel'])
report['DESY5'].update(changed_integration_redshifts=int(np.sum(frame.zHD!=frame.zCMB)),
    max_abs_delta_distance_modulus=float(np.max(np.abs(5*np.log10(new_dl/old_dl)))),
    geometry_comparison='LCDM Om=.3 H0=70; both use corrected dense quadrature')
Path('validation/sn_loading_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
