# DeepTCR

### Deep Learning Methods for Parsing T-Cell Receptor Sequencing (TCRSeq) Data

DeepTCR is a python package that has a collection of unsupervised and supervised 
deep learning methods to parse TCRSeq data. To see examples of how the algorithms can 
be used on an example datasets, see the subdirectory 'tutorials' at the [github repository](https://github.com/sidhomj/DeepTCR) for a collection of tutorial 
use cases across multiple datasets. 

While DeepTCR will run with Tensorflow-CPU versions, for optimal training times, we suggest training these algorithms on GPU's (requiring CUDA, cuDNN, and tensorflow-GPU). 

For questions or help, email: jsidhom1@jhmi.edu

## Publication

For full description of algorithm and methods behind DeepTCR, refer to the following manuscript:

[Sidhom, J. W., Larman, H. B., Pardoll, D. M., & Baras, A. S. (2021). DeepTCR is a deep learning framework for revealing sequence concepts within T-cell repertoires. Nat Commun 12, 1605](https://www.nature.com/articles/s41467-021-21879-w)

### Installation
In order to install DeepTCR:

```python
pip3 install DeepTCR

```

Or to install latest updated versions from Github repo:
 
Either download package, unzip, and run setup script:

```python
python3 setup.py install
```

Or use:

```python
pip3 install git+https://github.com/sidhomj/DeepTCR.git

```

## Dependencies

See [requirements.txt](https://github.com/sidhomj/DeepTCR/blob/master/requirements.txt) for all DeepTCR dependencies. Installing DeepTCR from Github repository or PyPi will install all required dependencies. It is recommended to create a virtualenv and installing DeepTCR within this environment to ensure proper versioning of dependencies.

In the most recent release (DeepTCR 2.0, fifth release), the package now uses python 3.8 & Tensorflow 2.0. Since this has required an overhaul in a lot of the code, there could be some bugs so we would greatly appreciate if you post any issues to the issues page and I will do my best to fix them as quickly as possible. One can find the latest DeepTCR 1.x version under the v1 branch if you still want to use that version. Or one can specifically pip install the specific version desired.

Instructions on how to create a virtual environment can be found [here.](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)


