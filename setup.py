from setuptools import setup, find_packages
import sys


if sys.version_info.major != 3:
    raise RuntimeError("DeepTCR requires Python 3")

setup(
    name="DeepTCR",
    description="Deep Learning Methods for Parsing T-Cell Receptor Sequencing (TCRSeq) Data",
    version="1.2.1",
    author="John-William Sidhom",
    author_email="jsidhom1@jhmi.edu",
    packages=find_packages(),
    url="https://github.com/sidhomj/DeepTCR",
    license="LICENSE",
    long_description=open("README.md").read()
)

