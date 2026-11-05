"""Bounded fixed-work likelihood benchmark; never opens production chains."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='1'
import sys,time,json,pickle,resource,multiprocessing as mp
from functools import partial
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from Kosmulator_main import Config, constants as K, Class_run as CR, rd_helpers as RD, Kosmulator_MCMC as M, utils
import User_defined_modules as U

def evaluate(theta,context):
    data,cfg,name=context
    RD._rd_model_class_core.cache_clear()
    ans=M.emcee_prob(theta,data,cfg['observation_types'][0],cfg,U.Get_model_function(name),name,cfg['observations'][0],0)
    return ans,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def main():
    paper=json.loads(Path('reproducibility/SAIP2026_IDE/configs/paper.json').read_text())
    for k,v in paper['class_settings'].items(): setattr(K,k,v)
    runtime=dict(constants=paper['class_settings'],models={k:getattr(U,k) for k in ('FREE_W','FREE_W_DM','ALLOW_NEGATIVE_ENERGIES','ALLOW_BIG_RIP','ALLOW_DOOM_FACTOR_INSTABILITIES')})
    result=[]
    for name in ('LCDM_v','SignSwitch_IDE_1'):
        cfgs,data=Config.create_config(U.Get_model_names([name]),model_name=[name],observation=paper['observations'],
            prior_limits=dict({k:tuple(v) for k,v in paper['prior_limits'].items()},r_d=(100,200)),free_w=True,free_w_dm=False)
        cfg=cfgs[name]
        data['PantheonPS']['cov']=utils.compute_pantheon_cov(data,cfg,None,0,
            os.path.join(K.OBSERVATIONS_BASE,'PantheonP.cov'),obs_tag='PantheonPS')
        CR.ensure_class_ready(name,no_rebuild=True,announce=False)
        rng=np.random.default_rng(613)
        points=[]
        for i in range(32):
            p=dict(Omega_m=.307+rng.uniform(-.015,.015),H_0=70+rng.uniform(-2,2),M_abs=-19.25+rng.uniform(-.03,.03),w=-1.03+rng.uniform(-.02,.02),delta=.02+rng.uniform(-.01,.01))
            points.append(np.array([p[k] for k in cfg['parameters'][0]]))
        context=(data,cfg,name); fn=partial(evaluate,context=context)
        reference=None
        for workers in (1,8,16):
            RD._rd_model_class_core.cache_clear()
            start=time.perf_counter()
            if workers==1:
                values=list(map(fn,points)); startup=0.
            else:
                with mp.get_context('spawn').Pool(workers,initializer=CR.init_clik_worker,
                        initargs=(None,None,None,None,None,runtime)) as pool:
                    # warm each process through representative full likelihoods
                    pool.map(fn,points[:workers],chunksize=1)
                    startup=time.perf_counter()-start
                    start=time.perf_counter()
                    values=pool.map(fn,points,chunksize=1)
            elapsed=time.perf_counter()-start
            arr=np.array([v[0] for v in values]);assert np.isfinite(arr).all(),arr
            if reference is None: reference=arr
            diff=float(np.max(np.abs(arr-reference)))
            assert diff<1e-8,(name,workers,diff)
            row=dict(model=name,workers=workers,evaluations=len(points),seconds=elapsed,startup_warmup_seconds=startup,
                     max_loglike_diff=diff,max_worker_rss_MiB=max(v[1] for v in values)/1024,
                     context_pickle_MiB=len(pickle.dumps(context))/2**20)
            result.append(row);print(row,flush=True)
        for h in (60.,82.,83.,87.71355,90.):
            p=dict(__model_name__=name,H_0=h,Omega_m=.307,w=-1.03,delta=.02,w_dm=0.)
            try: outcome=dict(rd=RD.compute_rd_model_class(p))
            except Exception as exc: outcome=dict(error=str(exc)[-1100:])
            result.append(dict(model=name,H_0=h,omega_b=.048*(h/100)**2,**outcome))
    Path('validation/pool_benchmark_results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
