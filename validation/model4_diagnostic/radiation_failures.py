from pathlib import Path
exec(Path(__file__).with_name('replay.py').read_text().split('rows=[]')[0])
rows=[]
for d in (-.2,-.25,-.3,-.35,-.5):
 for epoch in (None,1e-18):
  R._rd_model_class_core.cache_clear();row=dict(delta=d,a_ini=epoch)
  try:row['rd']=R.compute_rd_model_class(dict(Omega_m=.301,H_0=72.,w=-1.,w_dm=0.,delta=d,__model_name__='Linear_IDE_4'))
  except Exception as e:row['error']=str(e)
  rows.append(row)
  print(d,epoch,row.get('rd'),row.get('error','')[-180:],flush=True)
(out/'radiation_failures.json').write_text(json.dumps(rows,indent=2)+'\n')
