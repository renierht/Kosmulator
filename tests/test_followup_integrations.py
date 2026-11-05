"""Regression checks for the non-IDE upstream ports."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
from Kosmulator_main import Class_run as C, Kosmulator_MCMC as M, utils as U
from Kosmulator_main import Model_comparison as MC

class FollowupTests(unittest.TestCase):
    def test_source_inputs_and_generated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'python').mkdir()
            src=root/'python/classy.pyx'; src.write_text('first')
            binary=root/'binary.so'; binary.touch(); os.utime(binary,(1,1))
            first=C._source_tree_hash(tmp)
            self.assertEqual(C._source_newer_than(tmp,str(binary)),str(src))
            (root/'python/classy.cpp').write_text('generated')
            self.assertEqual(C._source_tree_hash(tmp),first)
            src.write_text('second')
            self.assertNotEqual(C._source_tree_hash(tmp),first)

    def test_relink_forced_once(self):
        with patch.object(C,'_ensure_libclass'), patch.object(C,'_strip_mvec_from_setup'), patch.object(C,'_ensure_python_external_symlink'), patch.object(C.subprocess,'run') as run:
            C._build_python_extension('/tmp/model')
        calls=[c.args[0] for c in run.call_args_list if 'build_ext' in c.args[0]]
        self.assertEqual(len(calls),1)
        self.assertIn('--force',calls[0])

    def test_empty_unknown_and_populated_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'chain.h5'
            self.assertIsNone(M._saved_steps(p))
            p.write_text('not HDF'); self.assertIsNone(M._saved_steps(p))
            with h5py.File(p,'w'): pass
            self.assertEqual(M._saved_steps(p),0)
            with h5py.File(p,'w') as f: f.create_dataset('foreign',data=[1])
            self.assertIsNone(M._saved_steps(p))
            with h5py.File(p,'w') as f: f.create_group('mcmc').attrs['iteration']=7
            self.assertEqual(M._saved_steps(p),7)

    def test_zeus_alignment_resume_and_prior_subtraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=str(Path(tmp)/'chain.h5')
            chain=np.arange(12.).reshape(3,2,2); logs=np.arange(6.).reshape(3,2)
            cb=U.AppendProgressCallback(p,2)
            cb(2,chain[:2],logs[:2]); cb(3,chain,logs,force=True)
            cb=U.AppendProgressCallback(p,2)
            cb(1,chain[:1]+20,logs[:1]+20,force=True)
            with h5py.File(p) as f:
                np.testing.assert_array_equal(f['log_prob_chain'],np.concatenate([logs,logs[:1]+20]))
                samples=f['samples'][1:].reshape(-1,2)
            with patch.object(M,'log_prior_all',return_value=np.ones(6)*2):
                ll,_=MC.saved_likelihood(samples,p,1,{},0)
            np.testing.assert_array_equal(ll,np.concatenate([logs[1:].reshape(-1),logs[0]+20])-2)

    def test_legacy_zeus_history_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=str(Path(tmp)/'chain.h5')
            with h5py.File(p,'w') as f: f.create_dataset('samples',data=np.zeros((2,2,1)),maxshape=(None,2,1))
            U.AppendProgressCallback(p,1)(1,np.ones((1,2,1)),np.ones((1,2)))
            with h5py.File(p) as f: self.assertNotIn('log_prob_chain',f)
