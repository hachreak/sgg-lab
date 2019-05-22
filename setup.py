# -*- coding: utf-8 -*-
#
# This file is part of sgg_lab.
# Copyright 2018 Leonardo Rossi <leonardo.rossi@studenti.unipr.it>.
#
# pysenslog is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# pysenslog is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with pysenslog.  If not, see <http://www.gnu.org/licenses/>.

"""setuptools."""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sgg_lab',
    version='0.1.0',
    description='SGG Lab',
    url='',
    author='Leonardo Rossi',
    author_email='leonardo.rossi@studenti.unipr.it',
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        #  'Programming Language :: Python :: 3',
        #  'Programming Language :: Python :: 3.4',
        #  'Programming Language :: Python :: 3.5',
        #  'Programming Language :: Python :: 3.6',
    ],

    keywords='machine learning',
    packages=find_packages(),
    install_requires=[
        'pillow>=6.0.0',
        'pycocotools>=2.0.0',
        #  'click>=7.0',
        #  # FIXME see keras-vis#141 keras-vis#119
        #  # install from git
        #  # 'keras-vis>=0.4.1',
        #  # 'keras-vis-temp>=0.4.2',
        #  'matplotlib>=2.2.2',
        #  #  'PySide2>=5.11.1',
        'numpy>=1.16.2',
        #  'Keras>=2.2.4',
        #  #  'tensorflow>=1.9.0',
        #  'dlib>=19.15.0',
        #  'scikit-learn>=0.19.2',
        #  'opencv-python>=3.4',
        #  'Pillow>=5.3.0',
    ],
    extras_require={  # Optional
        'gpu': ['tensorflow-gpu'],
        'cpu': ['tensorflow'],
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
    #  entry_points='''
    #      [console_scripts]
    #      cnn-kit-cli=sgg_lab.cli.main:cli
    #  ''',
)
