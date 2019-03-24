# DeepTCR

## Deep Learning Methods for Parsing T-Cell Receptor Sequencing (TCRSeq) Data

DeepTCR is a python package that has a collection of unsupervised and supervised 
deep learning methods to parse TCRSeq data. To see examples of how the algorithms can 
be used on an example datasets, see the subdirectory 'tutorials' for a collection of tutorial 
use cases across multiple datasets. For complete documentation for all available methods,
see 'Supervised_Documentation.txt' and 'Unsupervised_Documentation.txt'. 
While DeepTCR will run with Tensorflow-CPU versions, for optimal training times, 
we suggest training these algorithms on GPU's (requiring CUDA, cuDNN, and tensorflow-GPU). 

DeepTCR now has the added functionality of being able to analyze paired alpha/beta chain inputs as well
as also being able to take in v/d/j gene usage if available. For detailed instructions on 
how to upload this type of data, refer to the documentation for loading data into DeepTCR.  

For questions or help, email: jsidhom1@jhmi.edu

## Publication

For full description of algorithm and methods behind DeepTCR, refer to the following manuscript:

Sidhom, J. W., Larman, H. B., Pardoll, D. M., & Baras, A. S. (2018). DeepTCR: a deep learning framework for revealing structural concepts within TCR Repertoire. bioRxiv, 464107.

https://www.biorxiv.org/content/early/2018/11/26/464107
## Dependencies

DeepTCR has the following python library dependencies:
1. numpy==1.14.5
2. pandas==0.23.1
3. tensorflow==1.11.0
4. scikit-learn==0.19.1
5. pickleshare==0.7.4
6. matplotlib==2.2.2
7. scipy==1.1.0
8. biopython==1.69
9. seaborn==0.9.0
10. PhenoGraph==1.5.2


## Installation


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






