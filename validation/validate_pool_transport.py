"""Bounded real likelihood transport and sampler persistence checks."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import sys, json, time, pickle, tempfile, multiprocessing as mp
from pathlib import Path
from functools import partial
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import emcee
from Kosmulator_main import Config, constants as K, Class_run as CR, Kosmulator_MCMC as M
from Kosmulator_main import pool_context
import User_defined_modules as U

@pool_context.cleanup_after_run
def main():
    name = "SignSwitch_IDE_1"
    results = {}
    with tempfile.TemporaryDirectory() as folder:
        configs, data = Config.create_config(U.Get_model_names([name]), model_name=[name],
            observation=[["CC"]], prior_limits=dict(Omega_m=(.25,.35), H_0=(65,75),
            w=(-1.2,-.85),delta=(-.06,.06)),free_w=True,free_w_dm=False,
            nwalkers=16,nsteps=24,burn=4)
        cfg = configs[name]
        with patch.object(K,"force_emcee",True), patch.object(K,"EMCEE_MOVES","de_snooker"):
            chains=[]
            for optimized in (False,True):
                path=str(Path(folder)/f"cc_{optimized}.h5")
                with mp.get_context("spawn").Pool(2) as pool:
                    kw=dict(data=data, saveChains=True,chain_path=path,overwrite=False,
                        MODEL_func=U.Get_model_function(name),CONFIG=cfg,autoCorr=False,
                        parallel=True,model_name=name,obs=["CC"],Type=["CC"],colors={},
                        convergence=.05,last_obs=True,PLOT_SETTINGS={"seed":173,"init_jitter_frac":.01},
                        obs_index=0,use_mpi=False,num_cores=2,pool=pool,vectorised=True)
                    original=pool_context.prepare
                    implementation=original if optimized else lambda fn,args,pool,path:(fn,args)
                    with patch.object(pool_context,"prepare",implementation):
                        np.random.seed(173)
                        cfg["nsteps"]=24
                        M.run_mcmc(**kw)
                        cfg["nsteps"]=28
                        M.run_mcmc(**dict(kw,resumeChains=True))
                backend=emcee.backends.HDFBackend(path,read_only=True)
                assert backend.iteration==28
                chains.append(backend.get_chain())
            np.testing.assert_array_equal(*chains)
            assert not list(Path(folder).glob(".kosm_pool_args_*"))
            results["CC_save_resume"] = dict(steps=28, walkers=16, exact_transport_agreement=True)

        paper=json.loads(Path("reproducibility/SAIP2026_IDE/configs/paper.json").read_text())
        for key,value in paper["class_settings"].items(): setattr(K,key,value)
        U.ALLOW_NEGATIVE_ENERGIES=U.ALLOW_BIG_RIP=U.ALLOW_DOOM_FACTOR_INSTABILITIES=True
        configs,data=Config.create_config(U.Get_model_names([name]),model_name=[name],
            observation=[["DES_Dovekie","DESI_DR2"]],prior_limits=dict(paper["prior_limits"],r_d=(100,200)),
            free_w=True,free_w_dm=False)
        cfg=configs[name]
        assert cfg["parameters"][0]==["Omega_m","w","delta","H_0"]
        runtime=dict(constants=paper["class_settings"],models={k:getattr(U,k) for k in
            ("FREE_W","FREE_W_DM","ALLOW_NEGATIVE_ENERGIES","ALLOW_BIG_RIP","ALLOW_DOOM_FACTOR_INSTABILITIES")})
        CR.ensure_class_ready(name,no_rebuild=True,announce=False)
        args=(data,cfg["observation_types"][0],cfg,U.Get_model_function(name),name,cfg["observations"][0],0)
        points=[np.array([.3,-.92,-.01,69+i*.1]) for i in range(16)]
        results["DES_transport"]=[]
        reference=None
        # Reversed order in the second round; fresh workers avoid cross-method caches.
        for mode in ("original","cached","cached","original"):
            with mp.get_context("spawn").Pool(4,initializer=CR.init_clik_worker,
                    initargs=(None,None,None,None,None,runtime)) as pool:
                if mode=="cached":
                    fn,_=pool_context.prepare(M.emcee_prob,args,pool,str(Path(folder)/"des.h5"))
                else:
                    fn=partial(M.emcee_prob,**dict(zip(
                        ("data","Type","CONFIG","MODEL_func","model_name","obs","obs_index"),args)))
                pool.map(fn,points[:4],chunksize=1)
                start=time.perf_counter()
                values=np.array(pool.map(fn,points[4:],chunksize=1))
                elapsed=time.perf_counter()-start
                assert np.isfinite(values).all()
                if reference is None: reference=values
                np.testing.assert_allclose(values,reference,rtol=0,atol=1e-10)
                results["DES_transport"].append(dict(mode=mode,seconds=elapsed,evaluations=12,
                    callable_bytes=len(pickle.dumps(fn)),max_difference=float(np.max(np.abs(values-reference)))))
        Path("validation/pool_transport_results.json").write_text(json.dumps(results,indent=2))
        print(json.dumps(results,indent=2),flush=True)

if __name__=="__main__":
    main()
