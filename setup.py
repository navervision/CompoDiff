"""
CompoDiff
Copyright (c) 2023-present NAVER Corp.
Apache-2.0
"""
import os
import pkg_resources
from setuptools import setup, find_packages

setup(
    name='compodiff',
    version='0.1.1',
    description='Easy to use CompoDiff library',
    author='NAVER Corp.',
    author_email='dl_visionresearch@navercorp.com',
    url='https://github.com/navervision/CompoDiff',
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), 'requirements.txt'))
            )
        ],
    packages=find_packages(),
)
