"""Bounded endpoint screen for the requested +/-0.15 EOS prior; no MCMC."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Kosmulator_main import rd_helpers as R,Class_run as C
assert C.ensure_class_ready('Linear_IDE_5',no_rebuild=True,announce=False)
rows=[]
for om in (.1,.3,.5):
 for h in (60.,75.,90.):
  for d in (0.,-.1,.1):
   for v in (-.15,0.,.15):
    row=dict(delta=d,w_dm=v,Omega_m=om,H_0=h,w=-1.)
    start=time.monotonic()
    try:
     row['rd']=R.compute_rd_model_class(dict(row,__model_name__='Linear_IDE_5'))
     row['ok']=True
    except Exception as exc:
     row.update(ok=False,error=str(exc))
    row['seconds']=time.monotonic()-start
    rows.append(row)
    Path('validation/model5_wdm015_domain.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Completed',len(rows),'evaluations;',sum(x['ok'] for x in rows),'finite')
