"""Regression checks for positive-delta limit selection."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
spec = importlib.util.spec_from_file_location('boundary_comparison',
    Path(__file__).resolve().parents[1]/'Kosmulator_main/Model_comparison.py')
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

class PositiveDeltaTests(unittest.TestCase):
    samples = np.array([[.8,.03],[1.2,.02],[1.,.01]])
    def run_case(self, fn, **options):
        return mc.polish_likelihood(self.samples, np.array([fn(x) for x in self.samples]),
            fn, [(0.,2.),(0.,1.)], ['x','delta'], dict(mc.DEFAULTS,maxfev=800,**options),
            positive_delta=True)
    def test_exact_zero_artificial_dip_is_never_evaluated(self):
        seen=[]
        def fn(x):
            seen.append(x[1])
            return 100. if x[1]==0 else -.5*((x[0]-1)**2+x[1])
        best,ll,runs,note=self.run_case(fn)
        self.assertTrue(all(v>0 for v in seen))
        self.assertEqual(best[1],1e-10)
        self.assertAlmostEqual(-2*ll,1e-10,places=7)
        self.assertTrue(runs[-1]['boundary_sequence_validated'])
        self.assertIn('from above',note)
    def test_true_interior_maximum_is_retained(self):
        fn=lambda x: -.5*((x[0]-1)**2+(x[1]-.2)**2)
        best,ll,runs,note=self.run_case(fn)
        self.assertAlmostEqual(best[1],.2,places=5)
        self.assertIn('interior MLE',note)
    def test_failed_competitive_sequence_is_not_silently_accepted(self):
        actual=mc.minimize
        def fail(*a,**kw):
            result=actual(*a,**kw); result.success=False;return result
        with patch.object(mc,'minimize',fail),self.assertRaisesRegex(ValueError,'requires review'):
            self.run_case(lambda x:-.5*((x[0]-1)**2+x[1]))
    def test_unpolished_mode_does_not_change_chain_minimum(self):
        fn=lambda x:-.5*((x[0]-1)**2+x[1])
        best,ll,runs,note=self.run_case(fn,polish=False)
        np.testing.assert_array_equal(best,self.samples[2])
        self.assertEqual(runs,[])

if __name__=='__main__':unittest.main()
