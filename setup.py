from setuptools import setup, find_packages
import numpy as np


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
    include_dirs=[np.get_include()],
    packages=find_packages(),
    url='dummy-url',
    project_urls={
        "Source Code": "on gitlab",
    }
)
