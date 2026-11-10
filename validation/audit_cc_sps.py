"""Upstream equivalence and fixed-point impact of optional SPS covariance."""
import ast,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from Kosmulator_main import Config,cc_covariance as C,utils
import User_defined_modules as U
source=subprocess.check_output(['git','show','34091d1:Kosmulator_main/Config.py'],text=True)
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='cc_sps_covariance')
namespace=dict(np=np,K=SimpleNamespace(CC_MORESCO2020_TABLE3=C.CC_MORESCO2020_TABLE3,CC_D4000_Z=C.CC_D4000_Z))
exec(compile(ast.Module(body=[node],type_ignores=[]),'<upstream>','exec'),namespace)
data=Config.load_all_data({'observations':[['CC']]})['CC']
z,h=data['redshift'],data['type_data']
r=h-70*utils._Ez(U.Get_model_function('LCDM_v'),z,dict(Omega_m=.3,H_0=70))
report={'baseline_chi2':float(r@data['inv_cov']@r),'cases':[]}
for mode in ('full','ooo'):
    original=namespace['cc_sps_covariance'](z,h,mode)
    for lib in (True,False):
        extra=C.cc_sps_covariance(z,h,mode,include_library=lib,extrapolation='hold')
        if lib: np.testing.assert_allclose(extra,original,rtol=1e-14,atol=1e-12)
        cov=data['cov']+extra
        report['cases'].append(dict(mode=mode,include_library=lib,extrapolation='hold',
            upstream_max_abs_difference=float(np.max(np.abs(extra-original))) if lib else None,
            chi2=float(r@np.linalg.solve(cov,r)),minimum_eigenvalue=float(np.linalg.eigvalsh(cov).min())))
Path('validation/cc_sps_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
