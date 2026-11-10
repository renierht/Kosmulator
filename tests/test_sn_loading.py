import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from Kosmulator_main import Config, Kosmulator_MCMC as M
from Kosmulator_main.sn_covariance import load_sn_covariance

class SNLoadingTests(unittest.TestCase):
    def test_redshift_columns_and_bad_rows(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'sn.csv'
            header = 'zCMB,zHD,zHEL,MU,MUERR_FINAL\n'
            path.write_text(header+'0.1,0.11,0.12,38,0.1\n')
            data = Config.load_named_sne_with_zcmb(path)
            np.testing.assert_array_equal(data['redshift'], [.11])
            np.testing.assert_array_equal(data['z_hel'], [.12])
            path.write_text(header+'0.1,0.11,nan,38,0.1\n')
            with self.assertRaises(ValueError): Config.load_named_sne_with_zcmb(path)
            path.write_text('#name zcmb mb dmb\none 0.1 38 0\ntwo 0.2 broken 0\n')
            with self.assertRaises(ValueError): Config.load_named_sne_with_zcmb(path)

    def test_explicit_covariance_semantics(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'cov.txt'
            covariance = np.array([[4.,.1],[.1,9.]])
            np.savetxt(path,np.r_[2,covariance.ravel()])
            actual, precision = load_sn_covariance(path,2)
            np.testing.assert_array_equal(actual,covariance)
            np.testing.assert_allclose(precision,np.linalg.inv(covariance))
            actual,_ = load_sn_covariance(path,2,np.array([1.,2.]))
            np.testing.assert_array_equal(actual,covariance+np.diag([1.,4.]))
            with self.assertRaises(ValueError): load_sn_covariance(path,3)
            np.savetxt(path,[2,1,2,2,1])
            with self.assertRaises(np.linalg.LinAlgError): load_sn_covariance(path,2)
            path.unlink()
            with self.assertRaises(FileNotFoundError): load_sn_covariance(path,2)

    def test_likelihood_uses_heliocentric_prefactor(self):
        data = dict(redshift=np.array([.1,.2]), z_hel=np.array([.11,.23]),
                    type_data=np.zeros(2), type_data_error=np.ones(2))
        config = {'parameters':[['H_0']]}
        with patch.object(M.RD,'_maybe_calibrate_rd'), patch.object(M.utils,'ensure_background_params',side_effect=lambda p:p), \
             patch.object(M.utils,'Comoving_distance_vectorized',return_value=np.array([100.,200.])), \
             patch.object(M.SP,'Calc_Generic_SNe_chi',return_value=8.) as chi:
            actual = M.model_likelihood(np.array([70.]),data,'SNe',config,lambda z,p:1.,'LCDM_v','DESY5',0)
            self.assertEqual(actual,-4.)
            np.testing.assert_allclose(chi.call_args.kwargs['model'],25+5*np.log10(np.array([100.,200.])*[1.11,1.23]))
