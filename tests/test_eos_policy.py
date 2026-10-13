import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import User_defined_modules as U
from Kosmulator_main import Config, Kosmulator_MCMC as M, chain_policy as CP, constants as K, utils
from Kosmulator_main.ide_background import MODEL_NAMES


class EOSPolicyTests(unittest.TestCase):
    def config(self, name, free_w, free_dm):
        priors=dict(Omega_m=(.2,.4),w=(-1.3,-.8),w_dm=(-.05,.05),
                    delta=(-.1,.1),delta_dm=(-.1,.1),delta_de=(-.1,.1),H_0=(65,75))
        # Fixed parameters must not need a prior at all.
        if not free_w: priors.pop('w')
        if not free_dm: priors.pop('w_dm')
        with patch.object(Config,'load_all_data',return_value={}):
            configs,_=Config.create_config(U.Get_model_names([name]),prior_limits=priors,
                observation=[['CC']],model_name=[name],free_w=free_w,free_w_dm=free_dm)
        return configs[name]

    def test_all_models_and_four_switch_combinations(self):
        for name in MODEL_NAMES:
            for fw in (False,True):
                for fd in (False,True):
                    with self.subTest(name=name,w=fw,dm=fd):
                        c=self.config(name,fw,fd)
                        names=c['parameters'][0]
                        self.assertEqual('w' in names,fw)
                        self.assertEqual('w_dm' in names,fd)
                        self.assertEqual(c['ndim'][0],len(names))
                        self.assertEqual(set(c['prior_limits'][0]),set(names))
                        expected={}
                        if not fw: expected['w']=-1.
                        if not fd: expected['w_dm']=0.
                        self.assertEqual(c['fixed_parameters'],expected)
                        theta=c['true_values'][0]
                        seen=[]
                        def background(z,p):
                            seen.append(dict(p)); return np.ones_like(z)
                        obs=dict(redshift=np.array([.1,.5]),type_data=np.array([70.,70.]),
                                 type_data_error=np.array([5.,5.]))
                        self.assertTrue(np.isfinite(M.model_likelihood(theta,obs,'CC',c,background,name,'CC',0)))
                        for k,v in expected.items(): self.assertEqual(seen[0][k],v)

    def test_same_dimension_different_eos_cannot_reuse_chain(self):
        a=self.config('NonLinear_IDE_2',True,False)
        b=self.config('NonLinear_IDE_2',False,True)
        self.assertEqual(a['ndim'],b['ndim'])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'chain.h5';p.touch()
            CP.write(p,a,0); CP.check(p,a,0)
            with self.assertRaisesRegex(ValueError,'mismatch'): CP.check(p,b,0)
            Path(str(p)+'.metadata.json').unlink()
            CP.check(p,a,0)
            with self.assertRaisesRegex(ValueError,'Legacy'): CP.check(p,b,0)

    def test_configuration_does_not_mutate_registry(self):
        before=U.Get_model_names(['NonLinear_IDE_2'])['NonLinear_IDE_2']['parameters'][:]
        self.config('NonLinear_IDE_2',False,False)
        self.assertEqual(U.Get_model_names(['NonLinear_IDE_2'])['NonLinear_IDE_2']['parameters'],before)

    def test_new_models_derive_rd_in_bao_combinations(self):
        with patch.object(Config,'load_all_data',return_value={}), \
             patch.object(K,'DERIVE_RD_WITH_MODEL_CLASS',True):
            for name in ('SignSwitch_IDE_1','SignSwitch_IDE_2'):
                configs,_=Config.create_config(U.Get_model_names([name]),
                    prior_limits=dict(Omega_m=(.2,.4),delta=(-.1,.1),H_0=(65,75),r_d=(130,170)),
                    observation=[['CC','DESI_DR2']],model_name=[name],free_w=False,free_w_dm=False)
                self.assertNotIn('r_d',configs[name]['parameters'][0])
                self.assertNotIn('r_d',configs[name]['prior_limits'][0])

    def test_zeus_flush_between_callback_intervals_and_resume(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'chain.h5')
            first=np.arange(24*4*2).reshape(24,4,2)
            writer=utils.AppendProgressCallback(path,100)
            writer(24,first,np.zeros((24,4)))
            self.assertFalse(Path(path).exists())
            writer(24,first,np.zeros((24,4)),force=True)
            writer(24,first,np.zeros((24,4)),force=True)
            next_block=np.ones((4,4,2))
            utils.AppendProgressCallback(path,100)(4,next_block,np.zeros((4,4)),force=True)
            with h5py.File(path,'r') as f:
                np.testing.assert_array_equal(f['samples'][:],np.concatenate((first,next_block)))

    def test_unclassified_stability_prior_fails_before_sampling(self):
        with patch.object(U,'ALLOW_DOOM_FACTOR_INSTABILITIES',False):
            for name,fw,fd in [('NonLinear_IDE_2',True,True),
                               ('SignSwitch_IDE_1',True,False),
                               ('SignSwitch_IDE_2',False,False),
                               ('Linear_IDE_1',False,False)]:
                with self.subTest(model=name),self.assertRaisesRegex(ValueError,'doom-factor'):
                    self.config(name,fw,fd)


if __name__=='__main__': unittest.main()
