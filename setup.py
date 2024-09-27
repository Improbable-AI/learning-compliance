from setuptools import find_packages
from distutils.core import setup

setup(
    name='b1_gym',
    version='1.0.0',
    author='Tifanny Portela',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='',
    description='Toolkit for deployment of sim-to-real RL on the Unitree B1+Z1.',
    install_requires=['jaynes==0.9.2',
                      'params-proto==2.10.9',
                      'gym',
                      'tqdm',
                      'matplotlib',
                      'numpy==1.23.5',
                      'wandb==0.15.0',
                      'wandb_osh',
                      'imageio'
                      ]
)
