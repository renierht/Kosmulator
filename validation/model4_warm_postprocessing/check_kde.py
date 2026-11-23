import json,itertools
from pathlib import Path
import h5py,numpy as np
from getdist import MCSamples
root=Path.cwd();chain=next((root/'MCMC_Chains/Model4_EOS_CC_v2_model4_warm_lambda_iw_cc').rglob('*.h5'))
with h5py.File(chain,'r') as f:
 x=f['mcmc/chain'][3000:int(f['mcmc'].attrs['iteration'])].reshape(-1,5)
meta=json.loads(Path(str(chain)+'.metadata.json').read_text());print(str(meta)[:3500],flush=True)
np.save('validation/model4_warm_postprocessing/samples.npy',x)
names=meta['parameters'];bounds=dict(Omega_m=[.1,.5],w_dm=[-.15,.15],delta=[-1,1],H_0=[60,90],M_abs=[-20.5,-18])
print('correlation',np.corrcoef(x.T),flush=True)
results=[]
for smooth in (2.,-1.):
 m=MCSamples(samples=x,names=names,ranges=bounds,settings={'fine_bins_2D':1024,'smooth_scale_2D':smooth,'smooth_scale_1D':2.,'boundary_correction_order':1})
 for a,b in itertools.combinations(names,2):
  try:
   d=m.get2DDensityGridData(a,b,num_plot_contours=2);r=dict(smooth=smooth,pair=[a,b],ok=True,contours=d.contours.tolist())
  except Exception as e:r=dict(smooth=smooth,pair=[a,b],ok=False,error=str(e))
  print(r,flush=True);results.append(r)
Path('validation/model4_warm_postprocessing/kde_checks.json').write_text(json.dumps(results,indent=2))
