from setuptools import setup, find_packages
from glob import glob
from distutils.extension import Extension
# from Cython.Distutils import build_ext
from os.path import pathsep
import numpy as np
from numba.pycc import CC

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = False

# # Cython extensions
# ext = '.pyx' if cythonize else '.c'
# ext_cpp = '.pyx' if cythonize else '.cpp'
# extensions = [
#     Extension('ncrf.opt', [f'ncrf/opt{ext}']),
#               include_dirs=['ncrf/dsyevh3C']),
# ]
# if cythonize:
#     extensions = cythonize(extensions)
# from purdonlabmeeg._temporal_dynamics_utils.numba_opt.kalman import cc
# extensions = [cc.distutils_extension()]
# print(extensions)

setup(
    name="purdonlabmeeg",
    description="blank",
    long_description='purdonlab meeg codes'
                     'GitHub: https://github.com/proloyd/neuro-currentRF',
    version="0.1",
    python_requires='>=3.6',

    install_requires=[
        'mne', 'numpy', 'scipy'
    ],

    # metadata for upload to PyPI
    author="Proloy DAS",
    author_email="proloy@umd.com",
    license="BSD 2-Clause (Simplified)",
    # cmdclass={'build_ext': build_ext},
    include_dirs=[np.get_include()],
    packages=find_packages(),
    # ext_modules=extensions,
    url='dummy-url',
    project_urls={
        "Source Code": "on gitlab",
    }
)
