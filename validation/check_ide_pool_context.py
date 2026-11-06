"""Real Model 9 CC likelihood agreement under spawn transport."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json, pickle, tempfile, multiprocessing as mp
import numpy as np
from functools import partial
from Kosmulator_main import Config, Kosmulator_MCMC as M
from Kosmulator_main.pool_context import prepare, cleanup_after_run
import User_defined_modules as U

@cleanup_after_run
def main():
    name = "SignSwitch_IDE_1"
    configs, data = Config.create_config(U.Get_model_names([name]), model_name=[name],
        observation=[["CC"]], prior_limits=dict(Omega_m=(.1,.5),H_0=(60,90),w=(-2,-.33),delta=(-1,1)),
        free_w=True,free_w_dm=False)
    cfg=configs[name]
    args=(data,cfg["observation_types"][0],cfg,U.Get_model_function(name),name,["CC"],0)
    points=[np.array([dict(Omega_m=.3,H_0=70+i*.1,w=-.95,delta=0)[p]
                      for p in cfg["parameters"][0]]) for i in range(8)]
    expected=[M.emcee_prob(p,*args) for p in points]
    with tempfile.TemporaryDirectory() as folder:
        with mp.get_context("spawn").Pool(2) as pool:
            fn,_=prepare(M.emcee_prob,args,pool,str(Path(folder)/"test.h5"))
            actual=pool.map(fn,points)
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
            result=dict(max_difference=float(np.max(np.abs(np.array(actual)-expected))),
                        old_callable_bytes=len(pickle.dumps(partial(M.emcee_prob, *(), **dict(zip(
                            ["data","Type","CONFIG","MODEL_func","model_name","obs","obs_index"],args))))),
                        new_callable_bytes=len(pickle.dumps(fn)))
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
