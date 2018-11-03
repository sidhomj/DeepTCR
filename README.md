# DeepTCR

Deep Learning Methods for Parsing T-Cell Receptor Sequencing (TCRSeq) Data

DeepTCR is a python package that has a collection of unsupervsied and supervised 
deep learning approaches to parse TCRSeq data. To see an example of how the algorithms can 
be used on an example dataset, see Tutorial.ipnyb. For complete documentation for all available methods,
see 'Supervised_Documentation.txt' and 'Unsupervised_Documentation.txt'. While DeepTCR will run with Tensorflow-CPU versions,
for optimal training times, we suggest training these algorithms on GPU's (requiring CUDA, cuDNN, and tensorflow-GPU). 

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


In order to install DeepTCR, run setup script:

```python
python3 setup.py install
```

Or use:

```python
pip3 install git+https://github.com/sidhomj/DeepTCR.git

```

Or use:

```python
pip3 install DeepTCR

```




