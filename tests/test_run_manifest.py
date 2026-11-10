import json
import tempfile
import unittest
from pathlib import Path
from Kosmulator_main.run_manifest import validate_or_record


class ManifestTests(unittest.TestCase):
    def test_scientific_changes_rejected_operational_changes_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = dict(observations=[["DES_Dovekie", "DESI_DR2"]],
                            priors={"w": [-2, -.33]}, free_w_dm=False, walkers=120)
            validate_or_record(directory, settings)
            validate_or_record(directory, dict(settings, max_steps=200000))
            for key, value in [("observations", [["PantheonPS"]]),
                               ("priors", {"w": [-3, -.33]}),
                               ("free_w_dm", True), ("walkers", 64),
                               ("emcee_moves", "de_snooker"),
                               ("distance_policy", "dense-grid-dz0.01-v1")]:
                with self.assertRaises(ValueError):
                    validate_or_record(directory, dict(settings, **{key: value}))
            self.assertEqual(json.loads((Path(directory) / "run_settings.json").read_text()), settings)

    def test_legacy_chain_is_not_relabelled(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "old.h5").touch()
            with self.assertRaises(ValueError):
                validate_or_record(directory, {})
            self.assertFalse((Path(directory) / "run_settings.json").exists())

    def test_generic_settings_roundtrip_and_physics_change(self):
        from argparse import Namespace
        from types import SimpleNamespace
        from Kosmulator_main.run_manifest import generic_settings
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"User_defined_modules.py").write_text("# mock physics")
            namespace = dict(model_names=["IDE"], observations=[["CC"]],
                             prior_limits={"w": (-2, -.33)}, true_values={"w": -1},
                             nwalkers=120, burn=1000, convergence=.05)
            (root/"CC.dat").write_text("0.1 70 5")
            (root/"CC_corr.txt").write_text("# fixture")
            constants = SimpleNamespace(EMCEE_MOVES="stretch", OMEGA_B=.048, OBSERVATIONS_BASE=str(root))
            cli = Namespace(num_cores=8, resume=False, force_emcee=True)
            settings = generic_settings(namespace, constants, SimpleNamespace(), cli, root)
            validate_or_record(root, settings)
            cli.num_cores = 4
            cli.resume = True
            validate_or_record(root, generic_settings(namespace, constants, SimpleNamespace(), cli, root))
            (root/"CC.dat").write_text("0.1 70 6")
            with self.assertRaises(ValueError):
                validate_or_record(root, generic_settings(namespace, constants, SimpleNamespace(), cli, root))
            (root/"CC.dat").write_text("0.1 70 5")
            constants.OMEGA_B = .05
            with self.assertRaises(ValueError):
                validate_or_record(root, generic_settings(namespace, constants, SimpleNamespace(), cli, root))
            constants.OMEGA_B = .048
            (root/"User_defined_modules.py").write_text("# changed physics")
            with self.assertRaises(ValueError):
                validate_or_record(root, generic_settings(namespace, constants, SimpleNamespace(), cli, root))

    def test_analysis_rejects_legacy_distance_policy(self):
        from Kosmulator_main.run_manifest import require_chain_distance_policy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain = root / "model" / "chain.h5"
            chain.parent.mkdir()
            chain.touch()
            policy = "dense-grid-dz0.01-v1"
            with self.assertRaises(ValueError):
                require_chain_distance_policy(chain, policy)
            (root / "run_settings.json").write_text('{}')
            with self.assertRaises(ValueError):
                require_chain_distance_policy(chain, policy)
            (root / "run_settings.json").write_text(json.dumps({"distance_policy":policy}))
            require_chain_distance_policy(chain, policy)
