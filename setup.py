#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# description
with open(os.path.join(here, 'README.md'), mode='rt', encoding='utf-8') as f:
    long_description = f.read()

# version
version = {}
with open(os.path.join(here, 'semmap', 'version.py'), mode='rt', encoding='utf-8') as f:
    exec(f.read(), version)


setup(
    name="semmap",
    version=version["__version__"],
    description="Semantic Map",
    license='GNU General Public License v3 or later (GPLv3+)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Philipp Heinrich",
    author_email="philipp.heinrich@fau.de",
    url="https://github.com/ausgerechnet/semmap",
    packages=[
        'semmap'
    ],
    scripts=[
        'bin/semmap-create',
    ],
    python_requires='>=3.6.2',
    install_requires=[
        "pandas>=2.0",
        "pymagnitude-lite>=0.1.143",
        "scikit-learn>=1.3.0",
        "umap-learn>=0.5.5"
    ],
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Development Status :: 3 - Alpha",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
