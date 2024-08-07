#! /usr/bin/env python
"""
Setup for mrcnn
"""
import os
import sys
from setuptools import setup, find_packages


def read(fname):
	"""Read a file"""
	return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_version():
	""" Get the package version number """
	import caesar_yolo
	return caesar_yolo.__version__


PY_MAJOR_VERSION=sys.version_info.major
PY_MINOR_VERSION=sys.version_info.minor
print("PY VERSION: maj=%s, min=%s" % (PY_MAJOR_VERSION,PY_MINOR_VERSION))

reqs= []
reqs.append('numpy')
reqs.append('numpyencoder')
reqs.append('future')
reqs.append('astropy')
reqs.append('fitsio')
reqs.append('pandas')
reqs.append('scikit-image')
reqs.append('Pillow')
reqs.append('ultralytics')
reqs.append('matplotlib')
reqs.append('regions')
reqs.append('mpi4py')


data_dir = 'data'

setup(
	name="caesar_yolo",
	version=get_version(),
	author="Simone Riggi",
	author_email="simone.riggi@gmail.com",
	description="Tool to detect radio sources from astronomical FITS images using YOLO v8",
	license = "GPL3",
	url="https://github.com/SKA-INAF/caesar-yolo",
	long_description=read('README.md'),
	packages=find_packages(),
	include_package_data=True,
	install_requires=reqs,
	scripts=['scripts/run.py'],
)
