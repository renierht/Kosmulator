import unittest
import numpy as np
from Kosmulator_main.significance import posterior_checks, likelihood_ratio, attach_lrt

class SignificanceTests(unittest.TestCase):
    def test_desi_conversion(self):
        p,z=likelihood_ratio(11.2808092514,2)
        self.assertAlmostEqual(p,.003551431115,10);self.assertAlmostEqual(z,2.9154789,6)
        self.assertEqual(likelihood_ratio(-4,2),(None,None))
        self.assertEqual(likelihood_ratio(4,0),(None,None))
        self.assertEqual(likelihood_ratio(0,2),(1.,0.))
    def test_prior_and_fixed(self):
        a=np.linspace(-1,1,1000)[:,None]
        x=posterior_checks(a,['delta'],{'delta':(-1,1)},{'w_dm':0})
        self.assertEqual(x['parameters']['delta']['sigma'],0)
        self.assertEqual(x['parameters']['w_dm']['status'],'fixed')
        x=posterior_checks(-abs(a),['delta'],{'delta':(-1,0)})
        self.assertIsNone(x['parameters']['delta']['sigma'])
        self.assertIn('restricted',x['parameters']['delta']['status'])
    def test_zero_tail_and_coupled_prior(self):
        a=np.linspace(-1,-.1,1000)[:,None]
        self.assertIsNone(posterior_checks(a,['delta'],{'delta':(-1,1)})['parameters']['delta']['sigma'])
        x=posterior_checks(a,['delta'],{'delta':(-1,1)},prior_probe=lambda t:t[0]<0)
        self.assertIn('restricted',x['parameters']['delta']['status'])
    def test_batch_dependence(self):
        # Perfectly dependent walkers must not generate an iid standard error.
        chain=np.repeat(np.linspace(-1,1,1000)[:,None,None],120,axis=1)
        x=posterior_checks(chain.reshape(-1,1),['delta'],{'delta':(-2,2)},chain=chain)
        self.assertGreater(x['parameters']['delta']['batch_mcse']['100']['se'],.1)
    def test_missing_nesting_suppressed(self):
        r={'model':{'obs':{'dChi_squared':-11.28}}};attach_lrt(r,'LCDM_v')
        self.assertIsNone(r['model']['obs']['LRT_sigma'])
    def test_bad_samples(self):
        self.assertIn('unavailable',posterior_checks([[np.nan]],['w'],{'w':(-2,0)})['status'])
    def test_failed_nesting_gates(self):
        from Kosmulator_main.significance import validate_nesting
        import copy
        cfg={'parameters':[['Omega_m','w','delta']], 'prior_limits':[{'Omega_m':(.1,.5),'w':(-2,-.33),'delta':(-1,1)}], 'observations':[['CC']]}
        rc={'parameters':[['Omega_m']], 'prior_limits':[{'Omega_m':(.1,.5)}], 'observations':[['CC']]}
        row={'Data_counts':{'CC':20},'Optimizer_runs':[{'success':True}], 'MLE_parameters':{'Omega_m':.3,'w':-.9,'delta':-.01}}
        ref={'Data_counts':{'CC':20},'Optimizer_runs':[{'success':True}], 'MLE_parameters':{'Omega_m':.3}}
        bad=copy.deepcopy(row);bad['Optimizer_runs'][0]['success']=False
        self.assertIn('optimization',validate_nesting('SignSwitch_IDE_1','LCDM_v',bad,ref,cfg,rc,{},0,0)['status'])
        restricted=copy.deepcopy(cfg);restricted['prior_limits'][0]['delta']=(-1,0)
        self.assertIn('null at/outside',validate_nesting('SignSwitch_IDE_1','LCDM_v',row,ref,restricted,rc,{},0,0)['status'])
        bad=copy.deepcopy(row);bad['Data_counts']={'CC':21}
        self.assertIn('different datasets',validate_nesting('SignSwitch_IDE_1','LCDM_v',bad,ref,cfg,rc,{},0,0)['status'])

if __name__=='__main__':unittest.main()
