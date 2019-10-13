#!/usr/bin/env python
# -*- coding: utf-8 -*-

# spvm setup.py

import io
import os
import json
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))
metaFileName = "pyp.json"

# Import the package meta data
with open(os.path.join(here, metaFileName)) as pmfile:
    meta = json.loads(pmfile.read())

# Use the README.md as the Long description
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

setup(
    name=meta['project_info']['name'],
    version=meta['project_vcs']['version'],
    description=meta['project_info']['description'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=meta['project_authors'][0]['name'],
    author_email=meta['project_authors'][0]['email'],
    python_requires=meta['project_requirements']['python_version'],
    url=meta['project_info']['url'],
    packages=find_packages(exclude=meta['project_vcs']['exclude_packages']),

    install_requires=meta['project_requirements']['python_packages'],
    include_package_data=True,
    license=meta['project_info']['license']
)
