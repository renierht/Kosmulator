import multiprocessing as mp
import pickle
import tempfile
import unittest
import numpy as np
from pathlib import Path
from Kosmulator_main.pool_context import prepare, cleanup_after_run


def likelihood(theta, matrix):
    return -float(np.dot(theta, theta)), float(matrix.sum())


class PoolContextTests(unittest.TestCase):
    def test_spawn_agreement_payload_and_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            @cleanup_after_run
            def run():
                matrix = np.ones((512, 512))
                with mp.get_context("spawn").Pool(2) as pool:
                    fn, args = prepare(likelihood, (matrix,), pool, str(Path(folder)/"chain.h5"))
                    self.assertEqual(args, ())
                    self.assertLess(len(pickle.dumps(fn)), 1024)
                    points = [np.array([1., 2.]), np.array([3., 4.])]
                    self.assertEqual(pool.map(fn, points), [likelihood(p, matrix) for p in points])
            run()
            self.assertEqual(list(Path(folder).glob("*.pkl")), [])

    def test_cleanup_on_exception_and_serial_identity(self):
        fn, args = prepare(likelihood, (np.ones(2),), None, None)
        self.assertIs(fn, likelihood)
        with tempfile.TemporaryDirectory() as folder:
            @cleanup_after_run
            def fail():
                prepare(likelihood, args, object(), str(Path(folder)/"chain.h5"))
                raise RuntimeError("test")
            with self.assertRaises(RuntimeError):
                fail()
            self.assertEqual(list(Path(folder).glob("*.pkl")), [])
