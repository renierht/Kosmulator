import unittest
from unittest.mock import patch
import numpy as np
from Kosmulator_main import Config, constants as K
from Kosmulator_main import cc_covariance as C

class CCSPS(unittest.TestCase):
    def test_independent_pairwise_equation_and_selection(self):
        data=Config.load_all_data({'observations':[['CC']]})['CC']
        z,h=data['redshift'],data['type_data']; t=np.asarray(C.CC_MORESCO2020_TABLE3)
        for mode,col in [('full',3),('ooo',4)]:
            for library in (True,False):
                actual=C.cc_sps_covariance(z,h,mode,include_library=library,extrapolation='hold')
                expected=np.zeros_like(actual)
                for i,zi in enumerate(z):
                    for j,zj in enumerate(z):
                        if zi in C.CC_D4000_Z and zj in C.CC_D4000_Z:
                            expected[i,j]=sum(np.interp(zi,t[:,0],t[:,k])*np.interp(zj,t[:,0],t[:,k])*h[i]*h[j]/10000 for k in ([1,2,col] if library else [1,col]))
                np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=1e-12)
                self.assertGreater(np.linalg.eigvalsh(actual).min(),-1e-10)
                self.assertEqual(np.count_nonzero(np.diag(actual)),15)

    def test_default_off_and_explicit_extrapolation(self):
        baseline=Config.load_all_data({'observations':[['CC']]})['CC']
        original=baseline['cov'].copy()
        self.assertIs(C.attach_cc_systematics(baseline,None),baseline)
        np.testing.assert_array_equal(original,baseline['cov'])
        with patch.object(K,'CC_SYS_COV','full'):
            with self.assertRaisesRegex(ValueError,'does not cover'):
                Config.load_all_data({'observations':[['CC']]})
            with patch.object(K,'CC_SPS_EXTRAPOLATION','hold'):
                data=Config.load_all_data({'observations':[['CC']]})['CC']
        np.linalg.cholesky(data['cov'])
        np.testing.assert_allclose(data['cov']@data['inv_cov'],np.eye(36),atol=1e-13)
        with self.assertRaises(ValueError): C.attach_cc_systematics(data,'full',extrapolation='hold')

    def test_invalid_policies_and_rows(self):
        z=np.array(C.CC_D4000_Z); h=np.full(len(z),100.)
        for kwargs in ({'mode':'typo'},{'mode':'full','extrapolation':'silent'},{'mode':'ooo','include_library':'false'}):
            with self.assertRaises(ValueError): C.cc_sps_covariance(z,h,**kwargs)
        with self.assertRaises(ValueError): C.cc_sps_covariance(z[:-1],h[:-1],'full')
