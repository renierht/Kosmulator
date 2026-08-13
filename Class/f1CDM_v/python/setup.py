# Class/f1CDM_v/python/setup.py (concise, robust)
from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as nm, os, os.path as osp, subprocess as sbp, sys

# gcc libdir for linker
GCCPATH_STRING = sbp.Popen(['gcc','-print-libgcc-file-name'], stdout=sbp.PIPE).communicate()[0]
GCCPATH = osp.normpath(osp.dirname(GCCPATH_STRING.decode()))

liblist = ["class", "m"]  # <- important: only class + m

root_folder   = osp.join(osp.dirname(osp.abspath(__file__)), "..")
include_folder= osp.join(root_folder, "include")
classy_folder = osp.join(root_folder, "python")
heat_folder   = osp.join(root_folder, "external","heating")
recfast_folder= osp.join(root_folder, "external","RecfastCLASS")
hyrec_folder  = osp.join(root_folder, "external","HyRec2020")
hmcode_folder = osp.join(root_folder, "external","HMcode")
halofit_folder= osp.join(root_folder, "external","Halofit")

# discover VERSION from common.h
with open(osp.join(include_folder, 'common.h'), 'r') as fh:
    for line in fh:
        if "_VERSION_" in line:
            VERSION = line.split()[-1][2:-1]
            break

classy_ext = Extension(
    "classy",
    [osp.join(classy_folder, "classy.pyx")],
    include_dirs=[nm.get_include(), include_folder, heat_folder, recfast_folder, hyrec_folder, hmcode_folder, halofit_folder],
    libraries=liblist,
    library_dirs=[root_folder, GCCPATH],  # root has libclass.a
    language="c++",
    extra_compile_args=["-std=c++11"],
)
classy_ext.cython_directives = {'language_level': "3" if sys.version_info.major>=3 else "2"}

setup(
    name='classy',
    version=VERSION,
    description='Python interface to CLASS',
    url='http://www.class-code.net',
    cmdclass={'build_ext': build_ext},
    ext_modules=[classy_ext],
)

