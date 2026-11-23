"""Derived-parameter diagnostic: all retained draws; no resampling or refitting."""
from pathlib import Path
import json, hashlib
import numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter, gaussian_filter1d
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def one(a,bounds):
 sd=np.std(a);bw=max(sd*10000**(-1/5),1e-10)
 lo=max(bounds[0],a.min()-4*bw);hi=min(bounds[1],a.max()+4*bw)
 hist,edges=np.histogram(a,bins=700,range=(lo,hi));dx=edges[1]-edges[0]
 # Reflection is applied only at a prior boundary; empty outer tails otherwise.
 den=gaussian_filter1d(hist.astype(float),bw/dx,mode='reflect')
 return (edges[:-1]+edges[1:])/2,den/den.max()

def two(x,y,bx,by,bins=360):
 a=np.column_stack((x,y));mu=a.mean(axis=0);L=np.linalg.cholesky(np.cov(a,rowvar=False));z=np.linalg.solve(L,(a-mu).T).T
 h=10000**(-1/6)
 lim=[(z[:,i].min()-4*h,z[:,i].max()+4*h) for i in range(2)]
 hist,xe,ye=np.histogram2d(z[:,0],z[:,1],bins=bins,range=lim)
 dx,dy=xe[1]-xe[0],ye[1]-ye[0]
 den=gaussian_filter(hist.astype(float),[h/dx,h/dy],mode='constant')
 u,v=np.meshgrid((xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2,indexing='ij')
 X=mu[0]+L[0,0]*u;Y=mu[1]+L[1,0]*u+L[1,1]*v
 valid=(X>=bx[0])&(X<=bx[1])&(Y>=by[0])&(Y<=by[1]);den[~valid]=0
 q=np.sort(den.ravel())[::-1];mass=np.cumsum(q)/q.sum()
 levels=[q[min(np.searchsorted(mass,p),len(q)-1)] for p in (.95,.68)]
 assert 0<levels[0]<levels[1]<den.max()
 enclosed=[float(den[den>=lev].sum()/den.sum()) for lev in levels]
 return X,Y,den,levels,enclosed

suffix='Model4_EOS_noCC_v1_model4_warm_lambda_postprocess_iw'
d=json.loads((ROOT/'Statistical_analysis_tables'/suffix/'model_comparison.json').read_text())
stats=next(iter(next(iter(d.values())).values()))
chain=Path(stats['Likelihood_source'].split('saved likelihood blobs: ')[1])
before=chain.stat()
with h5py.File(chain,'r') as f:
 n=int(f['mcmc'].attrs['iteration'])
 a=f['mcmc/chain'][stats['Burn']:n:stats['Thin']].reshape(-1,len(stats['Parameter_names']))
assert len(a)==stats['Retained_samples'] and np.isfinite(a).all()
data={k:a[:,i] for i,k in enumerate(stats['Parameter_names'])}
data['effective']=data['w_dm']-data['delta']
settings=json.loads(next(p/'run_settings.json' for p in chain.parents if (p/'run_settings.json').exists()).read_text())
bounds=settings['priors'];bounds['effective']=[bounds['w_dm'][0]-bounds['delta'][1],bounds['w_dm'][1]-bounds['delta'][0]]
names=['H_0','Omega_m','w_dm','delta','M_abs','effective']
labels=[r'$H_0$',r'$\Omega_{m,0}$',r'$w_{\rm dm}$',r'$\delta$',r'$M_{\rm abs}$',r'$w_{\rm dm}^{\rm eff}=w_{\rm dm}-\delta$']
plt.rcParams.update({'font.size':10,'axes.labelsize':12,'pdf.fonttype':42})
fig,axes=plt.subplots(6,6,figsize=(15,14))
fig.subplots_adjust(left=.08,bottom=.07,right=.99,top=.97,wspace=.08,hspace=.08)
limits={}
for key in names:
 lo,hi=np.quantile(data[key],[.0005,.9995]);pad=(hi-lo)*.06
 limits[key]=(max(bounds[key][0],lo-pad),min(bounds[key][1],hi+pad))
checks=[]
for i,y in enumerate(names):
 for j,x in enumerate(names):
  ax=axes[i,j]
  if j>i:ax.set_visible(False);continue
  ax.set_xlim(limits[x]);ax.xaxis.set_major_locator(MaxNLocator(3));ax.tick_params(direction='in',top=True,right=True,labelsize=8)
  if i==j:
   xx,dd=one(data[x],bounds[x]);ax.plot(xx,dd,color='#367bb5');ax.set_ylim(0,1.08);ax.set_yticks([])
  else:
   X,Y,D,lev,mass=two(data[x],data[y],bounds[x],bounds[y])
   ax.contourf(X,Y,D,levels=[lev[0],lev[1],D.max()*1.001],colors=['#c3d9ed','#578db9'])
   ax.contour(X,Y,D,levels=lev,colors=['#367bb5'],linewidths=.7)
   ax.set_ylim(limits[y]);ax.yaxis.set_major_locator(MaxNLocator(3));checks.append({'pair':[x,y],'enclosed_mass_95_68':mass})
  if i<5:ax.tick_params(labelbottom=False)
  else:ax.set_xlabel(labels[j])
  if j>0 or i==j:ax.tick_params(labelleft=False)
  else:ax.set_ylabel(labels[i])
for ext in ['png','pdf']:fig.savefig(OUT/f'corner_Model4_noCC_with_effective_wdm.{ext}',dpi=180)
plt.close(fig)
summary={k:{'median':float(np.median(data[k])),'interval68':np.quantile(data[k],[.16,.84]).tolist(),'std':float(np.std(data[k]))} for k in ['w_dm','delta','effective']}
summary.update(samples=len(a),burn=stats['Burn'],steps=n,chain=str(chain),correlation_wdm_delta=float(np.corrcoef(data['w_dm'],data['delta'])[0,1]),contour_checks=checks,method='Whitened histogram KDE; all retained draws; 68/95 percent smoothed contours; display axes use 0.05–99.95 percent quantiles; derived prior projection only. Diagnostic smoothing matches previous EOS overlays.')
assert chain.stat().st_mtime_ns==before.st_mtime_ns and chain.stat().st_size==before.st_size
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='contour_checks'},indent=2))
