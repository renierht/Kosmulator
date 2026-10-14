"""Production stopping policy: finite diagnostics, burn and resumed totals."""
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import emcee
import h5py
from Kosmulator_main import convergence as C, utils


class ConvergenceTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(413).normal(size=(1000, 24, 2))
        self.rules = C.convergence_rules()

    def test_defaults(self):
        self.assertEqual(self.rules, dict(tau_factor=100.,ess_min=2000.,rhat_max=1.01,rtol=.05))

    def test_mixed_chain_passes_and_stalled_parameter_fails(self):
        prev = C.convergence_status(self.x[:900],self.rules)['tau_max']
        self.assertTrue(C.convergence_status(self.x,self.rules,prev)['ok'])
        self.x[:,:,1] = 4.
        self.assertFalse(C.convergence_status(self.x,self.rules,prev)['ok'])

    def test_separated_walkers_fail(self):
        self.x[:,:12] += 5
        self.assertFalse(C.convergence_status(self.x,self.rules,1.)['criteria']['rhat'])

    def test_burn_and_consecutive_checks(self):
        m=C.ConvergenceMonitor(100,self.rules,consecutive=2)
        with patch.object(C,'convergence_status',return_value=dict(tau_max=1.,tau_param='x',n_post=900,
                ess=24000,rel_change=0.,rhat_max=1.,ok=True,walkers=24)) as status:
            self.assertFalse(m.update(1000,self.x))
            np.testing.assert_array_equal(status.call_args.args[0],self.x[100:])
            self.assertTrue(m.update(1100,np.concatenate([self.x,self.x[:100]])))
            self.assertTrue(m.final(1100,self.x)['converged'])

    def test_real_emcee_ceiling_resume_and_report(self):
        with tempfile.TemporaryDirectory() as tmp, patch('Plots.Plots.convergence_plot'):
            path=str(Path(tmp)/'chain.h5')
            backend=emcee.backends.HDFBackend(path)
            sampler=emcee.EnsembleSampler(8,2,lambda p:-.5*np.sum(p*p),backend=backend)
            ps=dict(autocorr_save_path=tmp,autocorr_check_every=10,conv_tau_factor=1e9)
            initial=np.random.default_rng(7).normal(size=(8,2))
            utils.emcee_autocorr_stopping(initial,sampler,30,'toy',[],['CC'],ps,global_burn=5)
            with h5py.File(path) as f: self.assertFalse(f.attrs['converged'])
            sampler2=emcee.EnsembleSampler(8,2,lambda p:-.5*np.sum(p*p),backend=backend)
            utils.emcee_autocorr_stopping(backend.get_last_sample(),sampler2,20,'toy',[],['CC'],ps,
                                           global_burn=5,resume_offset=30)
            self.assertEqual(sampler2.iteration,50)
            self.assertEqual(sampler2.kosm_monitor.hist['it'],[40,50])
            report=json.loads(Path(path+'.convergence.json').read_text())
            self.assertEqual(report['status']['n_post'],45)
            self.assertFalse(report['status']['converged'])

    def test_zeus_callback_uses_saved_prefix(self):
        m=C.ConvergenceMonitor(100,self.rules,check_every=100)
        callback=C.ZeusConvergenceCallback(m,100,done=500,prefix_chain=self.x[:500])
        callback(100,self.x[500:600],None)
        self.assertEqual(m.hist['it'],[600])
        self.assertEqual(m.hist['n_post'],[500])


if __name__ == '__main__': unittest.main()
