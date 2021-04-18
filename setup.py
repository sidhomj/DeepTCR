from setuptools import setup, find_packages
import sys
import os
from subprocess import check_output

if sys.version_info.major != 3:
    raise RuntimeError("DeepTCR requires Python 3")

dir = os.path.dirname(os.path.abspath(__file__))
req_file = os.path.join(dir,'requirements.txt')
with open(req_file) as f:
    required = f.read().splitlines()

try:
    num_gpus = len(check_output(['nvidia-smi', '--query-gpu=gpu_name',
                                 '--format=csv']).decode().strip().split('\n'))
    tf = 'tensorflow-gpu' if num_gpus > 1 else 'tensorflow'
except:
    tf = 'tensorflow'

if tf == 'tensorflow':
    sel = [x for x in required if x.startswith('tensorflow-gpu')]
    required.remove(sel[0])
    required.append(''.join(sel[0].split('-gpu')))

setup(
    name="DeepTCR",
    description="Deep Learning Methods for Parsing T-Cell Receptor Sequencing (TCRSeq) Data",
    version="2.0.10",
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