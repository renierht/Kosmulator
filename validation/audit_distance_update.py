"""Compare old and upstream distances with independent adaptive quadrature."""
import ast,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.integrate import quad
from Kosmulator_main import utils
import User_defined_modules as U

source=subprocess.check_output(["git","show","5e28be3:Kosmulator_main/utils.py"],text=True)
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="Comoving_distance_vectorized")
namespace=dict(vars(utils))
exec(compile(ast.Module(body=[node],type_ignores=[]),"<upstream>","exec"),namespace)
legacy=namespace["Comoving_distance_vectorized"]
new=utils.Comoving_distance_vectorized
report=[]
for name in ("LCDM_v","w0waCDM","SignSwitch_IDE_1","SignSwitch_IDE_2"):
    model=U.Get_model_function(name)
    p=dict(Omega_m=.3,H_0=70,w=-.95,w0=-.9,wa=-.5,w_dm=0,delta=-.01)
    z=np.array([.295,.51,.706,.93,1.317,1.491,2.33])
    reference=np.array([utils.C_KM_S/70*quad(lambda t:1/float(np.asarray(utils._Ez(model,np.array([t]),p)).reshape(-1)[0]),0,t,epsabs=1e-11,epsrel=1e-11)[0] for t in z])
    old=legacy(model,z,p)
    proposed=new(model,z,p)
    row=dict(model=name,old_max_relative_error=float(np.max(np.abs(old/reference-1))),
             proposed_max_relative_error=float(np.max(np.abs(proposed/reference-1))))
    assert row["proposed_max_relative_error"]<1e-5
    report.append(row)
Path("validation/distance_update_audit.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
