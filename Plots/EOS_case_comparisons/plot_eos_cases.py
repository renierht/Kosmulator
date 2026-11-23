"""Read-only EOS overlays. Whitened histogram KDE resolves thin correlated ridges.
All retained draws enter histograms; no posterior samples are dropped or modified.
68/95 percent contours integrate the smoothed density within rectangular priors.
"""
from pathlib import Path
import json,hashlib
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator,AutoMinorLocator
from scipy.ndimage import gaussian_filter,gaussian_filter1d
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
SETS={
 'Uncoupled':('Q=0', ['Model5_EOS_CC_v1_lcdm_iw_cc','Model5_EOS_CC_v1_wcdm_iw_cc','Model5_EOS_CC_v1_warm_lambda_iw_cc']),
 'Model4':(r'Q=3H\delta\rho_{\rm dm}', ['Model4_EOS_CC_v2_model4_lambda_iw_cc','Model4_EOS_CC_v2_model4_free_de_burn3000_iw_cc','Model4_EOS_CC_v2_model4_warm_lambda_postprocess_iw_cc']),
 'Model5':(r'Q=3H\delta\rho_{\rm de}', ['Model5_EOS_CC_v1_model5_lambda_iw_cc','Model5_EOS_CC_v1_model5_free_de_iw_cc','Model5_EOS_CC_v1_model5_warm_lambda_iw_cc'])}
COLORS=['#484848','#e65b60','#367bb5']
LABELS=[r'$w_{\rm de}=-1,\ w_{\rm dm}=0$',r'Free $w_{\rm de}$; $w_{\rm dm}=0$',r'$w_{\rm de}=-1$; free $w_{\rm dm}$']
AXLABEL={'H_0':r'$H_0\ [{\rm km\,s^{-1}\,Mpc^{-1}}]$','Omega_m':r'$\Omega_{m,0}$','w':r'$w_{\rm de}$','w_dm':r'$w_{\rm dm}$','delta':r'$\delta$'}
REF={'w':-1.,'w_dm':0.,'delta':0.}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':14,'xtick.labelsize':10,'ytick.labelsize':10,'pdf.fonttype':42,'axes.linewidth':.85})

def load(folder):
 p=ROOT/'Statistical_analysis_tables'/folder/'model_comparison.json'
 d=json.loads(p.read_text());v=next(iter(next(iter(d.values())).values()))
 chain=Path(v['Likelihood_source'].split('saved likelihood blobs: ')[1])
 with h5py.File(chain,'r') as f:
  n=int(f['mcmc'].attrs['iteration']);x=f['mcmc/chain'][v['Burn']:n:v['Thin']].reshape(-1,len(v['Parameter_names']))
 assert np.isfinite(x).all() and len(x)==v['Retained_samples']
 m=json.loads(next(q/'run_settings.json' for q in chain.parents if (q/'run_settings.json').exists()).read_text())
 with chain.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(x=x,names=v['Parameter_names'],fixed=v['Fixed_parameters'],bounds=m['priors'],chain=str(chain),sha256=sha,burn=v['Burn'],steps=n,n=len(x),dataset_signature=v['Dataset_signature'])

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

provenance={}
for tag,(kernel,folders) in SETS.items():
 runs=[load(f) for f in folders];assert len({r['dataset_signature'] for r in runs})==1
 names=['H_0','Omega_m','w','w_dm']+([] if tag=='Uncoupled' else ['delta']);N=len(names)
 limits={}
 for name in names:
  vals=[]
  for r in runs:
   if name in r['names']:
    a=r['x'][:,r['names'].index(name)];vals.extend(np.quantile(a,[.0005,.9995]))
   elif name in r['fixed']:vals.append(r['fixed'][name])
  if name in REF:vals.append(REF[name])
  low,high=min(vals),max(vals);pad=.06*(high-low);limits[name]=(low-pad,high+pad)
 fig,axes=plt.subplots(N,N,figsize=(2.55*N,2.45*N),squeeze=False)
 fig.subplots_adjust(left=.085,bottom=.08,right=.99,top=.99,wspace=.075,hspace=.075)
 checks=[]
 for i,y in enumerate(names):
  for j,x in enumerate(names):
   ax=axes[i,j]
   if j>i:ax.set_visible(False);continue
   ax.set_xlim(limits[x]);ax.xaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_minor_locator(AutoMinorLocator(2))
   ax.tick_params(which='both',direction='in',top=True,right=True)
   if i==j:
    ax.set_ylim(0,1.08);ax.set_yticks([])
   else:
    ax.set_ylim(limits[y]);ax.yaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
   for r,color in zip(runs,COLORS):
    hasx=x in r['names'];hasy=y in r['names']
    if i==j:
     if hasx:
      xx,dd=one(r['x'][:,r['names'].index(x)],r['bounds'][x]);ax.plot(xx,dd,c=color,lw=1.65)
     elif x in r['fixed']:ax.axvline(r['fixed'][x],c=color,lw=.85,ls='--',alpha=.8)
    elif hasx and hasy:
     X,Y,den,levels,mass=two(r['x'][:,r['names'].index(x)],r['x'][:,r['names'].index(y)],r['bounds'][x],r['bounds'][y])
     ax.contourf(X,Y,den,levels=[levels[0],levels[1],den.max()*1.00001],colors=[color,color],alpha=.16)
     ax.contour(X,Y,den,levels=levels,colors=[color],linewidths=[.75,1.05],alpha=.9)
     checks.append(dict(case=r['chain'],pair=[x,y],enclosed_mass_95_68=mass))
    else:
     if not hasx and x in r['fixed']:ax.axvline(r['fixed'][x],color=color,ls='--',lw=.7,alpha=.65)
     if not hasy and y in r['fixed']:ax.axhline(r['fixed'][y],color=color,ls='--',lw=.7,alpha=.65)
   # A zero-coupling reference is not a fixed case for the interacting models.
   if x=='delta':ax.axvline(0,color='#888888',ls=':',lw=.65)
   if y=='delta' and i!=j:ax.axhline(0,color='#888888',ls=':',lw=.65)
   if i==N-1:ax.set_xlabel(AXLABEL[x])
   else:ax.tick_params(labelbottom=False)
   if j==0 and i>0:ax.set_ylabel(AXLABEL[y])
   elif i!=j:ax.tick_params(labelleft=False)
 fig.legend([Line2D([0],[0],color=c,lw=2) for c in COLORS],LABELS,loc='upper right',bbox_to_anchor=(.98,.96),frameon=False,title='$'+kernel+'$',fontsize=11,title_fontsize=15)
 fig.text(.57,.72,'DESI DR2 BAO + Pantheon+SH0ES + CC\n68% and 95% credible contours\nDashed lines: fixed values; dotted: $\\delta=0$',fontsize=11,linespacing=1.6)
 fig.savefig(OUT/f'{tag}_three_EOS_cases.pdf',bbox_inches='tight');fig.savefig(OUT/f'{tag}_three_EOS_cases.png',dpi=180,bbox_inches='tight');plt.close(fig)
 provenance[tag]={'kernel':kernel,'runs':[{k:v for k,v in r.items() if k!='x'} for r in runs],'contour_checks':checks}
 print('Saved',tag,flush=True)
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
(OUT/'README.md').write_text('''# Three-case EOS corner figures

Each PDF/PNG overlays the three cases within one kernel. Uncoupled, Model 4 and Model 5 are complete. Model 2/3 await complete three-case chains. No LCDM fourth curve is added to the interacting figures.

All retained samples enter the density histograms, with original postprocessing burns (see provenance.json). Fixed parameters are reference lines, never artificial narrow probability distributions. The w_de versus w_dm panel has no joint contour because neither is free simultaneously. M_abs is marginalized, matching the reference figure. One-dimensional curves are peak-normalized for display.

To resolve the nearly singular Model 4 w_dm/delta ridge without a noisy Cartesian KDE, two-dimensional densities are smoothed on a whitened 360x360 histogram grid then transformed back. Gaussian bandwidth uses a fixed effective size of 10000 for consistent presentation, not a new ESS estimate. Density outside rectangular priors is masked, and contour masses are normalized on remaining support. This smoothing is a visualization choice, not a change to samples, priors, likelihood or tabulated credible intervals. It is not an exact correction for coupled physical boundaries. Display axes cover the union of 0.05–99.95 percentile ranges with padding and fixed/reference values; contour levels are computed on the full histogram domain. All draws, including tails beyond the displayed axes, contribute.

Reproduce using the Kosmulator_test Python environment: python Plots/EOS_case_comparisons/plot_eos_cases.py. No CLASS evaluations or sampling are performed. Source paths, hashes, burns, retained counts and numerical enclosed contour masses are recorded in provenance.json.
''')
