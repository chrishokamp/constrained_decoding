#!/usr/bin/env python

import os
import setuptools

setuptools.setup(
    name='constrained_decoding',
    version='0.1',
    description='Lexically constrained decoding with Grid Beam Search',
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.md')).read(),
    license='BSD 3-clause',
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering'],
    packages=['constrained_decoding'],
)
