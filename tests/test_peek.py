import hashlib
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from Kosmulator_main.peek import inspect_chain


class PeekTests(unittest.TestCase):
    def test_saved_iteration_burn_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"chain.h5"
            with h5py.File(path, "w") as handle:
                group = handle.create_group("mcmc")
                group.attrs["iteration"] = 12
                chain = np.full((100, 4, 2), np.nan)
                chain[:12] = np.random.default_rng(5).normal(size=(12,4,2))
                group.create_dataset("chain", data=chain)
            before = hashlib.sha256(path.read_bytes()).digest()
            result = inspect_chain(path, 3)
            self.assertEqual(result["retained_steps"], 8)
            self.assertEqual(result["saved_steps"], 12)
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).digest())
            with self.assertRaises(ValueError):
                inspect_chain(path, 11)
