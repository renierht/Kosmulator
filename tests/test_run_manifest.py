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
                               ("emcee_moves", "de_snooker")]:
                with self.assertRaises(ValueError):
                    validate_or_record(directory, dict(settings, **{key: value}))
            self.assertEqual(json.loads((Path(directory) / "run_settings.json").read_text()), settings)

    def test_legacy_chain_is_not_relabelled(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "old.h5").touch()
            with self.assertRaises(ValueError):
                validate_or_record(directory, {})
            self.assertFalse((Path(directory) / "run_settings.json").exists())
