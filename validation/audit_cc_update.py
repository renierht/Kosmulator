"""Fixed-point CC audit, not a posterior comparison or production run."""
import sys,json,subprocess
from pathlib import Path
from io import StringIO
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from Kosmulator_main import Config,utils
import User_defined_modules as U
old=np.loadtxt(StringIO(subprocess.check_output(['git','show','5e28be3:Observations/CC.dat'],text=True)))
data=Config.load_all_data({'observations':[['CC']]})['CC']
report={'rows':len(old),'minimum_covariance_eigenvalue':float(np.linalg.eigvalsh(data['cov']).min()),'models':[]}
p=dict(Omega_m=.3,H_0=70,w=-.95,w0=-.9,wa=-.5,w_dm=0,delta=-.01)
for name in ('LCDM_v','w0waCDM','SignSwitch_IDE_1','SignSwitch_IDE_2'):
    model=U.Get_model_function(name)
    previous=float(np.sum(((old[:,1]-70*utils._Ez(model,old[:,0],p))/old[:,2])**2))
    r=data['type_data']-70*utils._Ez(model,data['redshift'],p)
    diag=float(np.sum((r/data['type_data_error'])**2))
    correlated=float(r@data['inv_cov']@r)
    independent=float(r@np.linalg.solve(data['cov'],r))
    np.testing.assert_allclose(correlated,independent,rtol=1e-12)
    report['models'].append(dict(model=name,old_chi2=previous,updated_diagonal_chi2=diag,updated_correlated_chi2=correlated))
Path('validation/cc_update_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
