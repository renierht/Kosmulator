"""Plot-policy tests without requiring CLASS, chains, or plotting dependencies."""
import ast
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('plot_metadata', ROOT/'Kosmulator_main/Plot_metadata.py')
meta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meta)


class PlotMetadataTests(unittest.TestCase):
    def flags(self, negative=True, rip=True, doom=True):
        return dict(zip(meta.SWITCHES, (negative, rip, doom)))
    def cfg(self, **kw):
        return {'parameters': [['Omega_m', 'w', 'delta']],
                'prior_limits': [{'Omega_m': (.1, .5), 'w': (-2., -.33), 'delta': (-1., 1.)}],
                '_postprocessing_switches': self.flags(**kw)}
    def test_group_labels_preserve_dataset_names(self):
        for key in ('DESI_DR2+PantheonPS', 'DESI_DR2_PantheonP_SH0ES',
                    'DESI+DR2+PantheonPS', 'DESI_DR2+Pantheon+SH0ES'):
            self.assertEqual(meta.observation_label(key), 'DESI DR2 + Pantheon+ + SH0ES')
        self.assertEqual(meta.observation_label('CC+f_sigma_8'), 'CC + f_sigma_8')
    def test_regime_labels_follow_saved_flags(self):
        for flags, label in ((self.flags(), 'iwCDM'), (self.flags(negative=False), '+iwCDM'),
                             (self.flags(doom=False), 'SiwCDM')):
            self.assertEqual(meta.model_label('NonLinear_IDE_2', flags=flags), label)
        self.assertEqual(meta.model_label('LCDM_v'), 'LCDM')
        self.assertEqual(meta.model_label('Other_model', flags=self.flags()), 'Other_model')
    def test_positive_delta_support_does_not_impose_w_above_minus_one(self):
        x = np.array([[.3, -1.04, .001], [.31, -1.03, 0.]])
        old = x.copy()
        limits = meta.corner_ranges(self.cfg(negative=False), 0, 'NonLinear_IDE_2', x)
        self.assertEqual(limits['delta'], [0., 1.])
        self.assertEqual(limits['w'], [-2., -.33])
        np.testing.assert_array_equal(x, old)
    def test_stable_support_is_phantom_and_delta_can_be_negative(self):
        limits = meta.corner_ranges(self.cfg(doom=False), 0, 'NonLinear_IDE_2',
                                    np.array([[.3, -1.01, -.002]]))
        self.assertEqual(limits['w'], [-2., -1.])
        self.assertEqual(limits['delta'], [-1., 1.])
    def test_inconsistent_samples_raise_without_clipping(self):
        with self.assertRaises(ValueError):
            meta.corner_ranges(self.cfg(negative=False), 0, 'NonLinear_IDE_2',
                               np.array([[.3, -1.04, -.001]]))
        with self.assertRaises(ValueError):
            meta.corner_ranges(self.cfg(doom=False), 0, 'NonLinear_IDE_2',
                               np.array([[.3, -.99, -.001]]))
    def test_other_models_only_get_declared_prior_bounds(self):
        limits = meta.corner_ranges(self.cfg(negative=False, doom=False), 0, 'Other_model',
                                    np.array([[.3, -.9, -.001]]))
        self.assertEqual(limits['w'], [-2., -.33])
        self.assertEqual(limits['delta'], [-1., 1.])
    def test_scale_summary_uses_curve_resolver_and_propagates_failures(self):
        path = ROOT/'Plots/Plot_functions.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'bao_scale_summary')
        calls = []
        ns = {'np': np, 'resolve_plot_rd': lambda p,m,t: calls.append((p,m,t)) or 140.}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        pkg = types.ModuleType('Kosmulator_main')
        pkg.constants = types.SimpleNamespace(C_KM_S=299792.458, DERIVE_RD_WITH_MODEL_CLASS=True)
        with patch.dict('sys.modules', {'Kosmulator_main': pkg}):
            row = ns['bao_scale_summary']({'H_0': 72.}, 'LCDM_v', 'DESI_DR2')
            self.assertAlmostEqual(row['S'], 299792.458/(72.*140.))
            self.assertEqual(calls[-1][1:], ('LCDM_v', 'DESI_DR2'))
            self.assertEqual(row['policy'], 'model-derived CLASS')
            def fail(*a): raise RuntimeError('CLASS failure')
            ns['resolve_plot_rd'] = fail
            with self.assertRaises(RuntimeError):
                ns['bao_scale_summary']({'H_0':72.}, 'LCDM_v')
    def test_stdout_context_manager_remains_decorated(self):
        tree = ast.parse((ROOT/'Plots/Plots.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_filter_stdout')
        self.assertTrue(any(isinstance(d, ast.Name) and d.id == 'contextmanager' for d in node.decorator_list))


if __name__ == '__main__':
    unittest.main()
