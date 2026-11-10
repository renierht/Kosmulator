"""Offset projection checked against independent scalar minimization."""
import unittest
import numpy as np
from scipy.optimize import minimize_scalar
from Kosmulator_main.Statistical_packages import Calc_Generic_SNe_chi

class SNOffsetTests(unittest.TestCase):
    def test_projection_matches_fitted_offset_and_shift_invariance(self):
        model = np.array([38., 39., 40.])
        data = model + np.array([.1, -.2, .3])
        C = np.array([[.04,.01,0.],[.01,.09,.01],[0.,.01,.16]])
        for full in (False, True):
            W = np.linalg.inv(C) if full else np.diag(1 / np.diag(C))
            obs = dict(type_data=data, type_data_error=np.sqrt(np.diag(C)), marginalise_offset=True)
            if full: obs['inv_cov'] = W
            expected = minimize_scalar(lambda offset: (data-model-offset) @ W @ (data-model-offset)).fun
            value = Calc_Generic_SNe_chi(obs, model, {})
            self.assertAlmostEqual(value, expected, places=12)
            self.assertAlmostEqual(Calc_Generic_SNe_chi(obs, model+7., {'M_abs':-19.3}), value, places=12)

    def test_legacy_and_distance_modulus_modes(self):
        model = np.array([1.,2.])
        obs = dict(type_data=np.array([2.,4.]),type_data_error=np.ones(2))
        self.assertEqual(Calc_Generic_SNe_chi(obs,model,{'M_abs':1}),1.)
        obs['data_is_distance_modulus'] = True
        self.assertEqual(Calc_Generic_SNe_chi(obs,model,{'M_abs':1}),5.)

    def test_real_loaders_and_offset_invariance(self):
        import logging
        from Kosmulator_main import Config
        loaded = Config.load_all_data({'observations':[['DESY5'],['Union3']]}, logger=logging.getLogger(__name__))
        for name in ('DESY5', 'Union3'):
            obs = loaded[name]
            self.assertTrue(obs['marginalise_offset'])
            self.assertTrue(obs['data_is_distance_modulus'])
            self.assertEqual(obs['inv_cov'].shape, (len(obs['redshift']),)*2)
            model = obs['type_data'] + .01 * np.sin(obs['redshift'])
            self.assertAlmostEqual(Calc_Generic_SNe_chi(obs, model, {}),
                                   Calc_Generic_SNe_chi(obs, model+3, {'M_abs':-19}), places=8)
