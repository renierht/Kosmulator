"""Read-only regression against baseline Python and its archived CLASS binary.

Run from this worktree; never builds or edits the original checkout.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import types
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from Kosmulator_main import ide_background as I, Class_run as C

parser=argparse.ArgumentParser()
parser.add_argument('--baseline',type=Path,required=True)
args=parser.parse_args()
source=subprocess.check_output(['git','-C',str(args.baseline),'show',
    'c2bc139951067d78f5884742efda458b468dc933:User_defined_modules.py'],text=True)
old=types.ModuleType('baseline_udm')
old.__file__=str(args.baseline/'User_defined_modules.py')
exec(compile(source,old.__file__,'exec'),old.__dict__)
old.ALLOW_NEGATIVE_ENERGIES=old.ALLOW_BIG_RIP=old.ALLOW_DOOM_FACTOR_INSTABILITIES=True
rng=np.random.default_rng(20261013)
z=np.expm1(np.linspace(-2,12,501))
worst=0.
for m,name in enumerate(I.MODEL_NAMES[:8],1):
    for _ in range(16):
        p=dict(Omega_m=rng.uniform(.25,.35),Omega_b=.048,Omega_r=8e-5,
               w=rng.uniform(-1.3,-.75),w_dm=0.,delta=rng.uniform(.001,.05),
               delta_dm=rng.uniform(.001,.025),delta_de=rng.uniform(.001,.025))
        ref=old.Get_model_function(name)(z,p)
        got=I.expansion(m,z,p)
        assert np.isfinite(ref).all() and np.isfinite(got).all(),(m,p)
        error=float(np.max(abs(got/ref-1)))
        assert error<2e-11,(m,error)
        worst=max(worst,error)

binary=args.baseline/'Class/.cache/IDE_background/8b81a76fa7fc20140418b9643351f29e66a51f8a54120df2de30cfa719305fac/classy.cpython-311-x86_64-linux-gnu.so'
assert binary.exists(),binary
zs=np.expm1(np.linspace(0,16,161))
def runs(Class):
    out=[]
    for m in range(9):
        c=Class()
        pars=dict(h=.7,Omega_b=.048,Omega_cdm=.259,N_ur=3.044,N_ncdm=0,
                  Omega_Lambda=0.,w0_fld=-1.08,wa_fld=0.,ide_model=m,
                  delta_ide=.04,delta_dm_ide=.03,delta_de_ide=.02,output='')
        try:
            c.set(pars);c.compute()
            out.append((c.rs_drag(),np.array([c.Hubble(float(z)) for z in zs]),
                        c.get_background() if m==0 else None))
        finally: c.struct_cleanup();c.empty()
    return out
baseline=runs(C.load_classy_so(str(binary)).Class)
assert C.ensure_class_ready('SignSwitch_IDE_1',announce=False)
current=runs(C.get_Class())
report=dict(baseline_commit='c2bc139951067d78f5884742efda458b468dc933',
            python_cases=128,python_H_max_relative=worst,class_models=[])
for m,(ref,got) in enumerate(zip(baseline,current)):
    rd_error=abs(got[0]/ref[0]-1)
    h_error=float(np.max(abs(got[1]/ref[1]-1)))
    assert rd_error<1e-7,(m,'rd',rd_error)
    assert h_error<1e-9,(m,'H',h_error)
    report['class_models'].append(dict(selector=m,rd_relative=rd_error,H_relative=h_error))
    if m==0:
        for key in ref[2]: np.testing.assert_array_equal(ref[2][key],got[2][key])
        report['selector_zero_all_columns_identical']=True
Path('validation/warm_legacy_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
