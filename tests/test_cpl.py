"""CPL comparison: independent background, joint prior, CLASS routing and resume."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.integrate import solve_ivp
import User_defined_modules as U
from Kosmulator_main import Config, constants as K, Kosmulator_MCMC as M, chain_policy, rd_helpers as RD, Class_run as CR

class CPLTests(unittest.TestCase):
    def config(self):
        with patch.object(Config, 'load_all_data', return_value={}), patch.object(K, 'DERIVE_RD_WITH_MODEL_CLASS', True):
            c, _ = Config.create_config(U.Get_model_names(['w0waCDM']), model_name=['w0waCDM'],
                observation=[['PantheonPS','DESI_DR2']],
                prior_limits=dict(Omega_m=(.1,.5), w0=(-3,1), wa=(-3,2), H_0=(60,90), M_abs=(-20.5,-18), r_d=(100,200)))
        return c['w0waCDM']

    def test_lcdm_limit_and_normalization(self):
        z = np.linspace(0,4,101)
        for om in (.1,.3,.5):
            p=dict(Omega_m=om,w0=-1.,wa=0.)
            np.testing.assert_array_equal(U.w0waCDM_MODEL_vectorised(z,p),U.LCDM_MODEL_vectorised(z,p))
            self.assertEqual(U.w0waCDM_MODEL_vectorised(0.,p),1.)

    def test_independent_conservation_equation(self):
        x=np.linspace(0,-np.log(5),101)
        for w0,wa in [(-.8,-.7),(-1.2,.4),(-2.,1.9),(.9,-1.),(-2.9,-2.9)]:
            sol=solve_ivp(lambda t,y:-3*(1+w0+wa*(1-np.exp(t)))*y,(x[0],x[-1]),[1.],t_eval=x,rtol=2e-11,atol=1e-13)
            expected=np.sqrt(.3*np.exp(-3*x)+.7*sol.y[0])
            np.testing.assert_allclose(U.w0waCDM_MODEL_vectorised(np.expm1(-x),dict(Omega_m=.3,w0=w0,wa=wa)),expected,rtol=2e-10)

    def test_configuration_and_prior(self):
        c=self.config(); names=c['parameters'][0]
        self.assertEqual(names,['Omega_m','w0','wa','H_0','M_abs'])
        self.assertNotIn('r_d',names)
        theta=np.array([[.3,-.8,-.7,70,-19.3],[.3,-.8,.8,70,-19.3],[.3,-.8,1.,70,-19.3]])
        np.testing.assert_array_equal(np.isfinite(M.log_prior_all(theta,c,0)),[True,False,False])

    def test_resume_policy(self):
        c=self.config()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'test.h5';p.touch();chain_policy.write(p,c,0);chain_policy.check(p,c,0)
            c['prior_limits'][0]['wa']=(-2.,2.)
            with self.assertRaises(ValueError):chain_policy.check(p,c,0)

    def test_launcher_plan(self):
        import subprocess, sys, json
        root=Path(__file__).resolve().parents[1]
        cmd=[sys.executable,str(root/'reproducibility/SAIP2026_IDE/scripts/run_analysis.py'),
             '--only-model','cpl','--run-name','test_cpl','--num-cores','8','--plan']
        result=subprocess.run(cmd,cwd=root,text=True,capture_output=True,check=True)
        plan=json.loads(result.stdout)
        self.assertEqual(plan['models'],['w0waCDM'])
        self.assertEqual(plan['output_suffix'],'test_cpl_cpl')
        self.assertEqual((plan['walkers'],plan['burn'],plan['max_steps']),(120,1000,100000))
        self.assertEqual(plan['coupled_prior'],'w0+wa<0')
        self.assertEqual(plan['priors']['wa'],[-3.,2.])

    def test_failed_class_has_no_approximate_fallback(self):
        with patch.object(K,'DERIVE_RD_WITH_MODEL_CLASS',True), patch.object(RD,'compute_rd_class',side_effect=ValueError('test failure')), patch.object(RD,'compute_rd') as fallback:
            self.assertIsNone(RD._try_compute_rd({'__model_name__':'w0waCDM'}))
            fallback.assert_not_called()

    def test_class_parameters_cache_and_limit(self):
        seen=[]
        class Fake:
            def set(self,p): self.p=p;seen.append(p)
            def compute(self): pass
            def rs_drag(self): return 140.+self.p.get('wa_fld',0.)
            def struct_cleanup(self): pass
            def empty(self): pass
        RD._rd_model_class_core.cache_clear()
        with patch.object(CR,'_current_class_model','LCDM_v'),patch.object(CR,'get_Class',return_value=Fake):
            p=dict(__model_name__='w0waCDM',Omega_m=.3,H_0=70,w0=-.8,wa=-.5)
            a=RD.compute_rd_model_class(p); b=RD.compute_rd_model_class(dict(p,wa=-.6))
            self.assertNotEqual(a,b)
            self.assertEqual(RD.compute_rd_model_class(p),a);self.assertEqual(len(seen),2)
            self.assertEqual(seen[0]['w0_fld'],-.8);self.assertEqual(seen[0]['use_ppf'],'yes')
            RD.compute_rd_model_class(dict(p,w0=-1,wa=0))
            self.assertNotIn('w0_fld',seen[-1])
        RD._rd_model_class_core.cache_clear()

if __name__=='__main__':unittest.main()
