"""Run with python -m unittest discover -s tests -p test_model_comparison.py -v."""
import importlib.util
import ast
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location(
    "comparison_under_test",
    Path(__file__).resolve().parents[1] / "Kosmulator_main/Model_comparison.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)


class ComparisonTests(unittest.TestCase):
    def test_saved_posterior_subtracts_prior_and_requires_exact_alignment(self):
        samples = np.array([[1.], [2.]])
        backend = types.SimpleNamespace(
            get_chain=lambda **kw: samples.copy(), has_blobs=lambda: False,
            get_log_prob=lambda **kw: np.array([-2., -5.]))
        class File:
            def __init__(self, *args): pass
            def __enter__(self): return {"mcmc": {}}
            def __exit__(self, *args): pass
        package = types.ModuleType("Kosmulator_main")
        package.Kosmulator_MCMC = types.SimpleNamespace(
            log_prior_all=lambda *args: np.array([-1., -1.]))
        modules = {
            "Kosmulator_main": package,
            "emcee": types.SimpleNamespace(backends=types.SimpleNamespace(
                HDFBackend=lambda *args, **kw: backend)),
            "h5py": types.SimpleNamespace(File=File),
        }
        with tempfile.NamedTemporaryFile() as file, patch.dict("sys.modules", modules):
            ll, origin = mc.saved_likelihood(samples, file.name, 0, {}, 0)
            np.testing.assert_equal(ll, [-1., -4.])
            self.assertIn("minus", origin)
            self.assertEqual(mc.saved_likelihood(samples+.01, file.name, 0, {}, 0),
                             (None, None))

    def test_plot_rd_passes_model_and_does_not_swallow_strict_failure(self):
        path = Path(__file__).resolve().parents[1]/"Plots/Plot_functions.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "resolve_plot_rd")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        package = types.ModuleType("Kosmulator_main")
        package.constants = types.SimpleNamespace(DERIVE_RD_WITH_MODEL_CLASS=True)
        package.rd_helpers = types.SimpleNamespace(
            _resolve_rd=lambda p, typ: p["__model_name__"])
        with patch.dict("sys.modules", {"Kosmulator_main": package}):
            self.assertEqual(namespace["resolve_plot_rd"]({}, "LCDM_v"), "LCDM_v")
            with self.assertRaises(ValueError):
                namespace["resolve_plot_rd"]({})
            def fail(*args): raise RuntimeError("CLASS failure")
            package.rd_helpers._resolve_rd = fail
            with self.assertRaises(RuntimeError):
                namespace["resolve_plot_rd"]({}, "LCDM_v")

    def test_dic_uses_mean_not_minimum_and_filters_aligned_samples(self):
        samples = np.array([[0.], [2.], [4.], [np.nan]])
        ll = np.array([0., -2., -8., np.nan])
        _, _, result = mc.posterior_statistics(samples, ll, lambda x: -.5*x[0]**2)
        self.assertAlmostEqual(result["D_at_mean"], 4.)
        self.assertAlmostEqual(result["D_bar"], 20/3)
        self.assertAlmostEqual(result["DIC"], 28/3)
        self.assertEqual(result["Excluded_samples"], 1)

    def test_selected_data_count_and_two_dimensional_calibration(self):
        cfg = {"observations": [["PantheonPS", "DESI_DR2"]],
               "observation_types": [["SNe", "DESI"]]}
        data = {"PantheonPS": {"m_b_corr": np.zeros(1657)},
                "DESI_DR2": {"redshift": np.zeros(13)}}
        self.assertEqual(mc.count_observations(data, cfg, 0)[0], 1670)
        cfg = {"observations": [["BBN_prior"]], "observation_types": [["BBN"]]}
        self.assertEqual(mc.count_observations(
            {"BBN_prior": {"cov": np.eye(2), "mu_Neff": 3.}}, cfg, 0)[0], 2)

    def test_boundary_search_is_separate_from_dic(self):
        samples = np.array([[1., .5], [.8, .3], [1.2, .4]])
        fn = lambda x: -.5*((x[0]-1)**2+x[1]) if x[1] >= 0 else -np.inf
        ll = np.array([fn(x) for x in samples])
        options = dict(mc.DEFAULTS, maxfev=800)
        best, value, runs, note = mc.polish_likelihood(
            samples, ll, fn, [(0., 2.), (0., 1.)], ["x", "delta"], options,
            positive_delta=True)
        self.assertEqual(best[1], 1e-10)
        self.assertAlmostEqual(-2*value, 0., places=6)
        self.assertTrue(any(r["stage"].startswith("delta=") for r in runs))
        self.assertIn("boundary", note)

    def test_stable_sequence_stays_inside_strict_domain(self):
        samples = np.array([[.8, -1.1], [1.2, -1.2], [1., -1.05]])
        fn = lambda x: -.5*((x[0]-1)**2+(-x[1]-1)) if x[1] < -1 else -np.inf
        ll = np.array([fn(x) for x in samples])
        best, value, runs, note = mc.polish_likelihood(
            samples, ll, fn, [(0., 2.), (-2., -.33)], ["x", "w"],
            dict(mc.DEFAULTS, maxfev=800), stable_w=True)
        self.assertLess(best[1], -1.)
        self.assertLess(-2*value, 1e-6)
        self.assertEqual(len([r for r in runs if r["stage"].startswith("w=")]), 3)
        self.assertIn("supremum", note)

    def test_missing_or_different_reference_never_fakes_zero(self):
        row = {"Data_counts": {"a": 10}, "Chi_squared": 3.,
               "AIC": 5., "AICc": 6., "DIC": 7., "BIC": 8.}
        result = mc.compare_models({"other": {"a": row.copy()}}, "LCDM")
        self.assertTrue(np.isnan(result["other"]["a"]["dAIC"]))
        different = dict(row, Data_counts={"a": 11})
        result = mc.compare_models({"other": {"a": row.copy()},
                                    "LCDM": {"a": different}}, "LCDM")
        self.assertTrue(np.isnan(result["other"]["a"]["dBIC"]))

    def test_complete_group_restores_flags_and_keeps_full_k(self):
        km = types.SimpleNamespace(
            log_prior_all=lambda x, *args: np.zeros(len(x)),
            log_likelihood_all=lambda x, *args: -.5*np.sum(x*x, axis=1))
        udm = types.SimpleNamespace(Get_model_function=lambda name: None,
            **{k: True for k in mc.SWITCHES})
        package = types.ModuleType("Kosmulator_main")
        package.Kosmulator_MCMC = km
        cfg = {"parameters": [["a", "b"]], "prior_limits": [{"a": (-3,3), "b": (-3,3)}],
               "observations": [["CC"]], "observation_types": [["CC"]], "burn": 0,
               "_postprocessing_switches": {k: False for k in mc.SWITCHES}}
        with patch.dict("sys.modules", {"Kosmulator_main": package, "User_defined_modules": udm}):
            result = mc.analyse_group(np.array([[0.,0.], [1.,1.], [2.,2.]]),
                {"CC": {"type_data": np.zeros(20)}}, cfg, "toy", 0)
        self.assertEqual(result["k"], 2)
        self.assertEqual(result["N"], 20)
        self.assertAlmostEqual(result["D_at_mean"], 2.)
        self.assertTrue(all(getattr(udm, key) for key in mc.SWITCHES))

    def test_export_is_full_precision_and_handles_undefined_deltas(self):
        row = {"N": 20, "k": 2, "p_D": 1.23456789, "Chi_squared": 4.,
               "D_bar": 6., "D_at_mean": 5., "AIC": 8., "AICc": 8.7,
               "DIC": 7., "BIC": 10., "MLE_note": "interior",
               "Data_counts": {"CC": 20}}
        results = mc.compare_models({"LCDM": {"CC": row.copy()},
                                     "IDE": {"CC": dict(row, AIC=7.)}}, "LCDM")
        with tempfile.TemporaryDirectory() as tmp:
            mc.export_comparison(results, tmp)
            parsed = json.loads((Path(tmp)/"model_comparison.json").read_text())
            self.assertEqual(parsed["IDE"]["CC"]["p_D"], 1.23456789)
            self.assertTrue((Path(tmp)/"information_criteria_1.pdf").is_file())
            self.assertTrue((Path(tmp)/"model_comparison.tex").is_file())


if __name__ == "__main__":
    unittest.main()
