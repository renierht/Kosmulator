"""Read-only exact-point likelihood decomposition, isolated by checkout process."""
import os,sys,json,inspect,ast
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();reference=Path(sys.argv[3]).resolve()
sys.path.insert(0,str(root));os.chdir(root)
from Kosmulator_main import Config,constants as K,Class_run as C,rd_helpers as R,utils,Kosmulator_MCMC as M
import User_defined_modules as U
paper=json.loads((root/'reproducibility/SAIP2026_IDE/configs/paper.json').read_text());ref=json.loads(reference.read_text())
for k,v in paper['class_settings'].items():setattr(K,k,v)
for k in ['ALLOW_NEGATIVE_ENERGIES','ALLOW_BIG_RIP','ALLOW_DOOM_FACTOR_INSTABILITIES']:setattr(U,k,True)
current='free_w' in inspect.signature(Config.create_config).parameters
orig_distance=utils.Comoving_distance_vectorized
oldroot=Path('/mnt/c/Users/Marcel/PhD/Kosmulator_test_Class/Kosmulator_referee_clean')
node=next(n for n in ast.parse((oldroot/'Kosmulator_main/utils.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='Comoving_distance_vectorized')
ns=dict(vars(utils));exec(compile(ast.Module(body=[node],type_ignores=[]),'<preserved-distance>', 'exec'),ns);old_distance=ns['Comoving_distance_vectorized']
rows=[]
for model,key in [('LCDM_v','LCDM'),('NonLinear_IDE_2','iwCDM')]:
 assert C.ensure_class_ready(model,no_rebuild=True,announce=False)
 oldrow=next(iter(ref[key].values()));center=dict(oldrow['MLE_parameters'])
 points=[('archived_MLE',center)]
 variations={'H_0':.8,'Omega_m':.008}
 if model=='NonLinear_IDE_2':variations.update(w=.04,delta=.002)
 for par,step in variations.items():
  for sign in [-1,1]:points.append((f'{par}_{sign:+d}',dict(center,**{par:center[par]+sign*step})))
 kw={'free_w':True,'free_w_dm':False} if current else {}
 cfgs,data=Config.create_config(U.Get_model_names([model]),model_name=[model],observation=[['PantheonPS','DESI_DR2','CC']],prior_limits=dict({k:tuple(v) for k,v in paper['prior_limits'].items()},r_d=(100,200)),**kw)
 cfg=cfgs[model];fn=U.Get_model_function(model);assert 'r_d' not in cfg['parameters'][0]
 for label,p0 in points:
  p=dict(w=-1.,w_dm=0.,delta=0.,Omega_b=.048,**{k:v for k,v in p0.items() if k not in ['w','delta']});p.update(p0);p['__model_name__']=model
  theta=np.array([p[k] for k in cfg['parameters'][0]])
  record=dict(model=model,point=label,parameters=p,rd=R.compute_rd_model_class(p),E=np.asarray(fn(np.array([0,.3,.7,1.,2.33,10.,1100.]),p)).tolist(),variants={})
  variants=[('native',orig_distance,False)] if not current else [('native',orig_distance,False),('old_distance_only',old_distance,False),('old_SN_prefactor_only',orig_distance,True),('both_old',old_distance,True)]
  for variant,distance,oldsn in variants:
   utils.Comoving_distance_vectorized=distance
   U.Comoving_distance_vectorized=distance
   chi={}
   for tag,typ in zip(cfg['observations'][0],cfg['observation_types'][0]):
    datum=data[tag]
    if oldsn and tag=='PantheonPS':datum=dict(datum);datum.pop('z_hel',None)
    chi[tag]=-2*float(M.model_likelihood(theta,datum,typ,cfg,fn,model,tag,0))
   record['variants'][variant]=chi
  utils.Comoving_distance_vectorized=orig_distance
  U.Comoving_distance_vectorized=orig_distance
  if label=='archived_MLE':record['archived_chi2']=oldrow['Chi_squared']
  rows.append(record);print(model,label,record['variants'],flush=True)
 out.write_text(json.dumps(dict(root=str(root),rows=rows),indent=2))
