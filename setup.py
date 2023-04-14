#!/usr/bin/env python

"""
Clustering comparison between DBSCAN, HDBSCAN, Kmeans, APLoD
"""
import sys
from setuptools import setup, find_packages

short_description = __doc__.split("\n")


setup(name='clustering',
      version='1.0',
      description='Python Distribution Utilities',
      author='Ella Nguyen',
      author_email='tnguyen31@hawk.iit.edu',
      url='https://github.com/EllaNguyen1711/clustering',
      packages=find_packages(),
     )