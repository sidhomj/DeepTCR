from setuptools import setup, find_packages
import sys
import os
import platform
# from subprocess import check_output

if sys.version_info.major != 3:
    raise RuntimeError("DeepTCR requires Python 3")

dir = os.path.dirname(os.path.abspath(__file__))

# Select requirements file based on operating system
if platform.system() == 'Darwin':  # macOS
    req_file = os.path.join(dir, 'requirements-macos.txt')
else:  # Other systems (Linux, Windows, etc.)
    req_file = os.path.join(dir, 'requirements.txt')

with open(req_file) as f:
    required = f.read().splitlines()


setup(
    name="DeepTCR",
    description="Deep Learning Methods for Parsing T-Cell Receptor Sequencing (TCRSeq) Data",
    version="2.1.28",
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