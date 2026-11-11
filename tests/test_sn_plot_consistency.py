import ast, unittest
from pathlib import Path
import numpy as np

class SNPlotConsistency(unittest.TestCase):
    def setUp(self):
        tree=ast.parse(Path('Plots/Plot_functions.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sn_plot_values')
        ns=dict(np=np,compute_Dc=lambda z,p,m:np.array([100.,200.]))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<plot>','exec'),ns)
        self.fn=ns['sn_plot_values']

    def test_marginalized_residuals_and_total_errors(self):
        cov=np.array([[.04,.01],[.01,.09]]); W=np.linalg.inv(cov)
        obs=dict(redshift=np.array([.1,.2]),z_hel=np.array([.11,.21]),type_data=np.array([38.,39.]),type_data_error=np.zeros(2),cov=cov,inv_cov=W,marginalise_offset=True)
        y,p,e=self.fn({'Union3':obs},'Union3',{},'LCDM_v')
        raw=obs['type_data']-p
        expected=raw@W@raw-(np.ones(2)@W@raw)**2/W.sum()
        self.assertAlmostEqual((y-p)@W@(y-p),expected,places=8)
        np.testing.assert_allclose(e,np.sqrt(np.diag(cov)))
        np.testing.assert_allclose(p,25+5*np.log10(np.array([111.,242.])))

    def test_calibrator_uses_cepheid_distance(self):
        obs=dict(zHD=np.array([.1,.2]),z_hel=np.array([.11,.21]),m_b_corr=np.array([15.,20.]),IS_CALIBRATOR=np.array([1,0]),CEPH_DIST=np.array([34.,-9]),type_data_error=np.ones(2))
        y,p,e=self.fn({'PantheonPS':obs},'PantheonP_SH0ES',{'M_abs':-19},'LCDM_v')
        self.assertEqual(p[0],34.)
        np.testing.assert_array_equal(y,[34.,39.])
