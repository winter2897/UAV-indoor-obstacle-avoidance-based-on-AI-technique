from setuptools import setup
import os

setup(
    name = "gym_airsim",
    version = "4.0",
    install_requires=['keras=2.0.6'],
    extras_require={
          'gym': ['gym'],
      }
)
    