"""Compile the actual production C density kernel in a temporary library."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from Kosmulator_main import ide_background as I


@unittest.skipUnless(shutil.which('gcc'), 'C kernel checks require gcc (run in WSL)')
class ClassDensityKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        root=Path(__file__).resolve().parents[1]
        src=Path(cls.tmp.name)/'wrapper.c'
        src.write_text('#include "ide_densities.h"\n'
            'int evaluate(int m,double a,double w,double v,double d,double dc,double dx,'
            'double c,double e,double *co,double *eo,double *q)'
            '{return ide_density_values(m,a,w,v,d,dc,dx,c,e,co,eo,q);}\n')
        so=Path(cls.tmp.name)/'kernel.so'
        subprocess.run(['gcc','-shared','-fPIC','-O2','-I'+str(root/'Class/IDE_background/include'),
                        str(src),'-lm','-o',str(so)],check=True,capture_output=True)
        cls.lib=ctypes.CDLL(str(so));cls.evaluate=cls.lib.evaluate
        cls.evaluate.argtypes=[ctypes.c_int]+[ctypes.c_double]*8+[ctypes.POINTER(ctypes.c_double)]*3
        cls.evaluate.restype=ctypes.c_int

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def call(self,m,a,p):
        dm,de,q=(ctypes.c_double() for _ in range(3))
        status=self.evaluate(m,a,p['w'],p['w_dm'],p['delta'],p['delta_dm'],p['delta_de'],
            p['Omega_m']-p['Omega_b'],1-p['Omega_m']-p['Omega_r'],
            ctypes.byref(dm),ctypes.byref(de),ctypes.byref(q))
        return status,np.array([dm.value,de.value]),q.value

    def test_python_c_agree_near_exceptional_branches(self):
        base=dict(Omega_m=.307,Omega_b=.048,Omega_r=8e-5,w=-1.08,w_dm=.025,
                  delta=.04,delta_dm=.03,delta_de=.02)
        cases=[(m,dict(base,delta=-.04 if m==10 else .04)) for m in range(1,11)]
        for eps in (0.,1e-8,1e-12):
            cases.extend((m,dict(base,delta=d)) for m,d in
                         [(4,1.105+eps),(5,1.105+eps),(6,1.105+eps),
                          (9,-2/3+eps),(10,2/3-eps)])
            cases.append((1,dict(base,w=-1.,w_dm=0.,delta_dm=.25-eps,delta_de=.25-eps)))
        for m,p in cases:
            for a in np.geomspace(.8,1.2,21):
                with self.subTest(model=m,a=a,p=p):
                    code,got,q=self.call(m,a,p)
                    self.assertEqual(code,0)
                    expected=np.array(I.densities(m,a,p))
                    np.testing.assert_allclose(got,expected,rtol=2e-10,atol=1e-12)
                    self.assertTrue(np.isfinite(q))

    def test_invalid_global_branches_rejected_by_c_and_python(self):
        p=dict(Omega_m=.307,Omega_b=.048,Omega_r=8e-5,w=-1.08,w_dm=.025,
               delta=.8,delta_dm=.03,delta_de=.02)
        for m,d in [(7,-2.),(8,-2.),(9,-.8),(10,.8)]:
            with self.subTest(model=m):
                q=dict(p,delta=d)
                self.assertFalse(I.domain(m,q))
                self.assertNotEqual(self.call(m,1.,q)[0],0)


class ClassCacheTests(unittest.TestCase):
    def test_stale_local_ide_binary_is_never_promoted(self):
        from Kosmulator_main import Class_run as C
        with patch.object(C,'_full_signature',return_value={'src':'new'}), \
             patch.object(C,'_current_class_model',None), \
             patch.object(C,'_find_cached_so',return_value=None), \
             patch.object(C,'find_classy_so',side_effect=AssertionError('stale local binary')):
            self.assertFalse(C.ensure_class_ready('SignSwitch_IDE_1',no_rebuild=True))
