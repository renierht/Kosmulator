import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from Kosmulator_main import Config, Kosmulator_MCMC as M
from Kosmulator_main.cc_covariance import attach_cc_correlations

class CCCovarianceTests(unittest.TestCase):
    def test_real_data_and_likelihood(self):
        data = Config.load_all_data({'observations':[['CC']]})['CC']
        self.assertEqual(len(data['redshift']),36)
        z = data['redshift']; i = np.flatnonzero(z==.46)[0]; j = np.flatnonzero(z==.67)[0]
        self.assertAlmostEqual(data['cov'][i,j], .932*12.33*17.82)
        self.assertIn(.4497,z)
        residual = data['type_data']-70*(1+z)
        expected = residual@np.linalg.solve(data['cov'],residual)
        with patch.object(M.RD,'_maybe_calibrate_rd'), patch.object(M.utils,'ensure_background_params',side_effect=lambda p:p), patch.object(M.utils,'E_of_z',return_value=1+z):
            actual = M.model_likelihood(np.array([70.]),data,'CC',{'parameters':[['H_0']]},None,'LCDM_v','CC',0)
        self.assertAlmostEqual(actual,-.5*expected,places=11)
        np.testing.assert_allclose(np.diag(data['cov']),data['type_data_error']**2)

    def test_invalid_pairs_fail(self):
        data = dict(redshift=np.array([.1,.2,.3]),type_data_error=np.ones(3))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pairs'
            for rows in ([ [.1,.4,.5] ],[[.1,.2,1.2]],[[.1,.2,.5],[.2,.1,.5]],[[.1,.2,.9],[.2,.3,.9],[.1,.3,-.9]]):
                np.savetxt(p,rows)
                with self.assertRaises((ValueError,np.linalg.LinAlgError)): attach_cc_correlations(dict(data),p)
            np.savetxt(p,[[.1,.2,.5]])
            with self.assertRaises(ValueError): attach_cc_correlations(dict(data,redshift=np.array([.1,.1,.2])),p)
            p.unlink()
            with self.assertRaises(FileNotFoundError): attach_cc_correlations(data,p)
