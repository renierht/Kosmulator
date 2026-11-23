"""Small CLASS domain screen, not a likelihood/prior boundary determination."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Kosmulator_main import rd_helpers as R,Class_run as C
assert C.ensure_class_ready('Linear_IDE_5',no_rebuild=True,announce=False)
rows=[]
for d in (0.,.02):
 for v in (-.5,-.1,0.,.1,.2,.3,.34,.5):
    start=time.monotonic()
    row=dict(delta=d,w_dm=v,Omega_m=.3,H_0=70.,w=-1.)
    try:
        row['rd']=R.compute_rd_model_class(dict(row,__model_name__='Linear_IDE_5'))
        row['ok']=True
    except Exception as exc:
        row.update(ok=False,error=str(exc))
    row['seconds']=time.monotonic()-start
    rows.append(row)
    Path('validation/model5_wdm_domain.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(row),flush=True)
