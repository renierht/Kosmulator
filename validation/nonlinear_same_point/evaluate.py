"""Isolated same-point evaluation; no sampling, fitting, or chain writes."""
import os,sys,json,inspect,hashlib,time
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();sys.path.insert(0,str(root));os.chdir(root)
from Kosmulator_main import Config,constants as K,Class_run as C,rd_helpers as R,utils,Kosmulator_MCMC as M
import User_defined_modules as U
paper=json.loads((root/'reproducibility/SAIP2026_IDE/configs/paper.json').read_text())
for k,v in paper['class_settings'].items():setattr(K,k,v)
for k in ['ALLOW_NEGATIVE_ENERGIES','ALLOW_BIG_RIP','ALLOW_DOOM_FACTOR_INSTABILITIES']:setattr(U,k,True)
name='NonLinear_IDE_2';assert C.ensure_class_ready(name,no_rebuild=True,announce=False)
p=dict(Omega_m=.296,H_0=73.44,w=-.913,w_dm=0.,delta=-.006,M_abs=-19.243,Omega_b=.048,__model_name__=name)
obs=['PantheonPS','DESI_DR2','CC'];kw={}
if 'free_w' in inspect.signature(Config.create_config).parameters:kw.update(free_w=True,free_w_dm=False)
cfgs,data=Config.create_config(U.Get_model_names([name]),model_name=[name],observation=[obs],prior_limits=dict({k:tuple(v) for k,v in paper['prior_limits'].items()},r_d=(100.,200.)),**kw)
cfg=cfgs[name];assert 'r_d' not in cfg['parameters'][0], 'r_d must be derived, not sampled';fn=U.Get_model_function(name);theta=np.array([p[k] for k in cfg['parameters'][0]])
rd=R.compute_rd_model_class(p);assert np.isfinite(rd)
ll={}
for tag,typ in zip(cfg['observations'][0],cfg['observation_types'][0]):
 ll[tag]=float(M.model_likelihood(theta,data[tag],typ,cfg,fn,name,tag,0))
z=np.array([.295,.51,.706,.93,1.317,1.491,2.33]);E=np.asarray(fn(z,p));dc=np.asarray(utils.Comoving_distance_vectorized(fn,z,p))
from scipy.integrate import quad
ref=np.array([utils.C_KM_S/p['H_0']*quad(lambda zz:1/float(np.asarray(fn(np.array([zz]),p)).ravel()[0]),0,zz,epsabs=1e-8,epsrel=1e-10)[0] for zz in z])
hashes={}
for pattern in ['PantheonP.*','DESI_DR2*','CC*']:
 for f in (root/'Observations').glob(pattern):
  if f.is_file():hashes[str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
res=dict(root=str(root),parameters=p,class_settings=paper['class_settings'],sampled_parameters=cfg['parameters'][0],rd=rd,z=z.tolist(),E=E.tolist(),Dc=dc.tolist(),Dc_quad=ref.tolist(),distance_relative_error=(dc/ref-1).tolist(),chi2={k:-2*v for k,v in ll.items()},data_hashes=hashes)
out.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
