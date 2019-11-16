from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    raise RuntimeError("DeepTCR requires Python 3")

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="DeepTCR",
    description="Deep Learning Methods for Parsing T-Cell Receptor Sequencing (TCRSeq) Data",
    version="1.3.9",
    author="John-William Sidhom",
    author_email="jsidhom1@jhmi.edu",
    packages=find_packages(),
    install_requires = required,
    url="https://github.com/sidhomj/DeepTCR",
    license="LICENSE",
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown'
)

