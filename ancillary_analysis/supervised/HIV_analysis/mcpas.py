from DeepTCR.DeepTCR import DeepTCR_WF
import glob
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import pandas as pd

gpu = 3
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
folds=25

files = glob.glob('../../../Data/HIV/*.tsv')
samples = []
labels = []
for file in files:
    file = file.split('/')[-1]
    samples.append(file)
    labels.append(file.split('_')[1])

epitopes = np.unique(labels)
df_tcr = pd.read_csv('../../../Data/McPAS-TCR.csv')
mcpas_counts = []
for e in epitopes:
    temp = df_tcr[df_tcr['Epitope.peptide'] == e]
    temp = temp.groupby(['CDR3.beta.aa']).agg({'Epitope.peptide': 'first'}).reset_index()
    mcpas_counts.append(temp)

df_epitope_counts = pd.DataFrame()
df_epitope_counts['epitope'] = epitopes
df_epitope_counts['counts'] = mcpas_counts
df_epitope_counts.sort_values(by='counts',inplace=True,ascending=False)

df_tcr = pd.read_csv('../../../Data/VDJ.tsv',sep='\t')
vdj_counts = []
for e in epitopes:
    temp = df_tcr[df_tcr['Epitope'] == e]
    temp = temp.groupby(['CDR3']).agg({'Epitope': 'first'}).reset_index()
    vdj_counts.append(len(temp))

df_epitope_counts = pd.DataFrame()
df_epitope_counts['epitope'] = epitopes
df_epitope_counts['counts'] = vdj_counts
df_epitope_counts.sort_values(by='counts',inplace=True,ascending=False)

df_tcr = pd.read_csv('../../../Data/vdj_mcpas.csv')
counts = []
for e in epitopes:
    temp = df_tcr[df_tcr['epitope'] == e]
    counts.append(len(temp))

df_epitope_counts = pd.DataFrame()
df_epitope_counts['epitope'] = epitopes
df_epitope_counts['counts'] = counts
df_epitope_counts.sort_values(by='counts',inplace=True,ascending=False)

