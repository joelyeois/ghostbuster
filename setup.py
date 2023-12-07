from setuptools import setup, find_packages, Extension
import numpy as np

setup(
    name='ghostbuster',
    version='1.0',
    description='Ghostbuster: a diffraction tomography algorithm for cryo-EM single particle reconstruction',
    packages=find_packages(),
    ext_modules=[Extension('ghostbuster.ctools',
                           sources = ['./ghostbuster/ctools.c'],
                           include_dirs = [np.get_include()])],
    author="Joel Yeo",
    author_email="joelyeo@u.nus.edu",
    license="GPL v3.0",
    keywords="Phase Retrieval, cryo-EM, Diffraction Tomography",
)