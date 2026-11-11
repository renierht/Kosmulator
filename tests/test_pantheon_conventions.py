import unittest
import numpy as np
from Kosmulator_main import Config
from Kosmulator_main.Statistical_packages import Calc_PantP_chi

class PantheonConventions(unittest.TestCase):
    def test_selection_and_cepheid_flags(self):
        d=dict(zHD=np.array([.005,.02,.03,.008]),zHEL=np.array([.006,.021,.031,.009]),
               IS_CALIBRATOR=np.array([1,1,0,0]),m_b_corr=np.arange(4.),CEPH_DIST=np.arange(4.)+30)
        plain=Config.prepare_pantheonP_data(d,mode='Pplus')
        anchored=Config.prepare_pantheonP_data(d,mode='PplusSH0ES')
        np.testing.assert_array_equal(plain['indices'],[1,2])
        np.testing.assert_array_equal(plain['IS_CALIBRATOR'],[0,0])
        np.testing.assert_array_equal(plain['z_hel'],[.021,.031])
        np.testing.assert_array_equal(anchored['indices'],[0,1,2])
        np.testing.assert_array_equal(anchored['IS_CALIBRATOR'],[1,1,0])

    def test_chi2_matches_independent_covariance_solve(self):
        cov=np.array([[.04,.01],[.01,.09]])
        mb=np.array([15.,20.]); cep=np.array([34.,-9.]); model=np.array([35.,39.])
        for trig in (np.array([1,0]),np.zeros(2)):
            r=mb+19-np.where(trig==1,cep,model)
            self.assertAlmostEqual(Calc_PantP_chi(mb,trig,cep,np.linalg.cholesky(cov),model,{'M_abs':-19.}),r@np.linalg.solve(cov,r),places=10)

    def test_real_masks_and_covariance_shapes(self):
        import pandas as pd
        df=pd.read_csv('Observations/PantheonP.dat',sep=r'\s+')
        data=Config.load_all_data({'observations':[['PantheonP'],['PantheonPS']]})
        for tag in ('PantheonP','PantheonPS'):
            mask=(df.zHD>.01).to_numpy()
            if tag=='PantheonPS': mask=mask | (df.IS_CALIBRATOR.to_numpy()>0)
            obs=data[tag]
            np.testing.assert_array_equal(obs['indices'],np.flatnonzero(mask))
            np.testing.assert_array_equal(obs['z_hel'],df.zHEL.to_numpy()[mask])
            self.assertEqual(obs['cov'].shape,(mask.sum(),mask.sum()))
