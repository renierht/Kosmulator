"""Independent analytic distances, ordering and input validity."""
import unittest
import numpy as np
from Kosmulator_main import utils

class DistanceTests(unittest.TestCase):
    def test_analytic_linear_expansion(self):
        z = np.array([2.33, 0., .51, .51, -.1, .01])
        actual = utils.Comoving_distance_vectorized(lambda z,p: 1 + z, z, {'H_0':70})
        expected = utils.C_KM_S / 70 * np.log1p(z)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-10)
        self.assertEqual(actual[2], actual[3])

    def test_zero_empty_and_invalid_inputs(self):
        model = lambda z,p: np.ones_like(z)
        self.assertEqual(utils.Comoving_distance_vectorized(model, [], {'H_0':70}).size, 0)
        np.testing.assert_array_equal(utils.Comoving_distance_vectorized(model, [0,0], {'H_0':70}), [0,0])
        self.assertTrue(np.isnan(utils.Comoving_distance_vectorized(model, [np.nan], {'H_0':70})).all())
        self.assertTrue(np.isnan(utils.Comoving_distance_vectorized(lambda z,p:-np.ones_like(z), [.5], {'H_0':70})).all())
