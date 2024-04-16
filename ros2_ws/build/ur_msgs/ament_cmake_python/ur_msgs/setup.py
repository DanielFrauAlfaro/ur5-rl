from setuptools import find_packages
from setuptools import setup

setup(
    name='ur_msgs',
    version='2.0.0',
    packages=find_packages(
        include=('ur_msgs', 'ur_msgs.*')),
)
