"""
Figure 1B,C
Supplementary Figures 3-7
"""

"""This script is used to characterize the performance of various featurization
methods on TCRSeq data from 9 Murine Antigens."""

from DeepTCR.DeepTCR import DeepTCR_U
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import seaborn as sns
from NN_Assessment_utils import *
import pickle
import os
from scipy.stats import ttest_rel
import matplotlib

os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
matplotlib.rc('font', family='Arial')

load_prev_data = True
# Instantiate training object
DTCRU = DeepTCR_U('Murine_U',device='/device:GPU:1')
split_seed = 0
graph_seed = 0
# Load Data
DTCRU.Get_Data(directory='../../Data/Murine_Antigens', Load_Prev_Data=False, aggregate_by_aa=True,
               aa_column_beta=0, count_column=1, v_beta_column=2, j_beta_column=3)

if load_prev_data is False:
    #Get distances from various methods
    #VAE_- Genes
    DTCRU.Train_VAE(Load_Prev_Data=False,use_only_gene=True,split_seed=split_seed,graph_seed=graph_seed)
    distances_vae_gene = pdist(DTCRU.features, metric='euclidean')

    # #VAE_- Sequencs Alone+
    DTCRU.Train_VAE(Load_Prev_Data=False,use_only_seq=True,split_seed=split_seed,graph_seed=graph_seed)
    distances_vae_seq = pdist(DTCRU.features, metric='euclidean')

    #VAE_- Gene+Sequencs
    DTCRU.Train_VAE(Load_Prev_Data=False,split_seed=split_seed,graph_seed=graph_seed)
    distances_vae_seq_gene = pdist(DTCRU.features, metric='euclidean')

    #Hamming
    distances_hamming = pdist(np.squeeze(DTCRU.X_Seq_beta, 1), metric='hamming')

    #Kmer
    kmer_features = kmer_search(DTCRU.beta_sequences)
    distances_kmer = pdist(kmer_features, metric='euclidean')

    #Global Seq-Align
    # distances_seqalign = pairwise_alignment(DTCRU.beta_sequences)
    # with open('Murine_seqalign.pkl','wb') as f:
    #     pickle.dump(distances_seqalign,f)

    with open('Murine_seqalign.pkl','rb') as f:
        distances_seqalign = pickle.load(f)

    distances_seqalign = distances_seqalign + distances_seqalign.T
    distances_seqalign = squareform(distances_seqalign)

    distances_list = [distances_vae_seq,distances_vae_gene,distances_vae_seq_gene,distances_hamming,distances_kmer,distances_seqalign]
    names = ['VAE-Seq','VAE-VDJ','VAE-Seq-VDJ','Hamming','K-mer','Global-Seq-Align']
    with open('distances_murine.pkl','wb') as f:
        pickle.dump([distances_list,names],f,protocol=4)

else:
    with open('distances_murine.pkl','rb') as f:
        distances_list,names = pickle.load(f)

dir_results = 'Murine_Results'
if not os.path.exists(dir_results):
    os.makedirs(dir_results)

#Assess performance metrtics via K-Nearest Neighbors
file_write = os.path.join(dir_results,'data_fig1b.csv')
if load_prev_data is False:
    df_metrics = Assess_Performance_KNN(distances_list,names,DTCRU.class_id,dir_results,metrics=['AUC'])
    df_metrics.to_csv(file_write)
else:
    df_metrics = pd.read_csv(file_write)

Plot_Performance(df_metrics,dir_results)

subdir = 'Performance_Summary'
if not os.path.exists(os.path.join(dir_results,subdir)):
    os.makedirs(os.path.join(dir_results,subdir))

names = ['Global-Seq-Align','K-mer','Hamming','VAE-Seq','VAE-VDJ','VAE-Seq-VDJ']
for m in np.unique(df_metrics['Metric']):
    fig,ax = plt.subplots()
    sns.violinplot(data=df_metrics[df_metrics['Metric']==m],
                   x='Algorithm',y='Value',cut=0,
                   order=names,ax=ax)
    plt.ylabel(m)
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.subplots_adjust(bottom=0.25)
    plt.ylabel(m,fontsize=24)
    plt.xticks(fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(os.path.join(dir_results,subdir,m+'.eps'))

#PVal Comparisons
comp = [['Hamming','Global-Seq-Align'],
 ['Hamming','K-mer'],
 ['VAE-Seq-VDJ','Hamming'],
 ['VAE-Seq-VDJ','VAE-Seq'],
  ['VAE-Seq-VDJ','VAE-VDJ']]

method = 'AUC'
method_1 = []
method_2 = []
p_val_list = []
for c in comp:
    df_test = df_metrics[df_metrics['Metric']==method]
    idx_1 = df_test['Algorithm'] == c[0]
    idx_2 = df_test['Algorithm'] == c[1]
    t, p_val = ttest_rel(df_test[idx_1]['Value'], df_test[idx_2]['Value'])
    method_1.append(c[0])
    method_2.append(c[1])
    p_val_list.append(p_val)

df_pval = pd.DataFrame()
df_pval['Method 1'] = method_1
df_pval['Method 2'] = method_2
df_pval['P_Val'] = p_val_list

#Assess Length Dependency of various methods
SRCC = []
fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(8,10))
ax =  np.ndarray.flatten(ax)
for n,distances,a in zip(names,distances_list,ax):
    len_distance = pdist(np.sum(DTCRU.X_Seq_beta>0,-1))
    len_seq = np.sum(DTCRU.X_Seq_beta>0,-1)
    corr,_ = spearmanr(len_distance,distances)
    SRCC.append(corr)
    df = pd.DataFrame()
    df['D_len'] = len_distance.astype(int)
    df['D_features'] = distances
    sns.boxplot(x='D_len',y='D_features',data=df,ax=a)
    a.set_title(n)
    a.set_xlabel('')
    a.set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(dir_results,'LD.eps'))

df_out = pd.DataFrame()
df_out['Methods'] = names
df_out['SRCC'] = SRCC
df_out.to_csv(os.path.join(dir_results,'Length_Dep.csv'),index=False)
