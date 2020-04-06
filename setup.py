from setuptools import setup, find_packages
import sys
import os

if sys.version_info.major != 3:
    raise RuntimeError("DeepTCR requires Python 3")

dir = os.path.dirname(os.path.abspath(__file__))
req_file = os.path.join(dir,'requirements.txt')
with open(req_file) as f:
    required = f.read().splitlines()
# required.remove('PhenoGraph==1.5.2')

setup(
    name="DeepTCR",
    description="Deep Learning Methods for Parsing T-Cell Receptor Sequencing (TCRSeq) Data",
    version="1.4.9",
    author="John-William Sidhom",
    author_email="jsidhom1@jhmi.edu",
    packages=find_packages(),
    install_requires = required,
    url="https://github.com/sidhomj/DeepTCR",
    license="LICENSE",
    long_description=open(os.path.join(dir,"README.md")).read(),
    long_description_content_type='text/markdown',
    package_data={'DeepTCR':[os.path.join('phenograph','louvain','*'),
                             os.path.join('functions','Supertype_Data_Dict.csv')]}
)