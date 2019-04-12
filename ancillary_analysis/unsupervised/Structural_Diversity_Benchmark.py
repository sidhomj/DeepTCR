"""This script is used to assess the performance of various featurization
methods to describe the structural diversity of in-silico samples where various
number of antigen-specific sequences are mixed and then the diversity is measured by
number of clusters and the entropy across those clusters. Fig.2"""

from DeepTCR.DeepTCR import DeepTCR_U
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy, pearsonr
from scipy.spatial.distance import pdist, squareform
from NN_Assessment_utils import *
import pickle
import matplotlib.pyplot as plt

# Instantiate training object
DTCRU = DeepTCR_U('Structural_Diversity_U',device='/gpu:1')

# Assess ability for structural entropy to be of measure of number of antigens
classes_all = np.array(['Db-F2', 'Kb-M38', 'Db-M45', 'Db-NP', 'Db-PA', 'Db-PB1', 'Kb-m139','Kb-SIY','Kb-TRP2'])

DTCRU.Get_Data(directory='../../Data/Murine_Antigens', Load_Prev_Data=False, aggregate_by_aa=True,
               aa_column_beta=0, count_column=1, v_beta_column=2, j_beta_column=3)

# VAE-Gene
DTCRU.Train_VAE(use_only_gene=True)
d_vae_gene = squareform(pdist(DTCRU.features))

# VAE-Seq
DTCRU.Train_VAE(use_only_seq=True)
d_vae_seq = squareform(pdist(DTCRU.features))

# VAE-Seq-Gene
DTCRU.Train_VAE()
d_vae_seq_gene = squareform(pdist(DTCRU.features))


# Hamming
d_hamming = squareform(pdist(np.squeeze(DTCRU.X_Seq_beta, 1), metric='hamming'))

# Kmer
kmer_features = kmer_search(DTCRU.beta_sequences)
d_kmer = squareform(pdist(kmer_features, metric='euclidean'))

# Global Seq-Align
# distances_seqalign = pairwise_alignment(DTCRU.beta_sequences)
# with open('Murine_seqalign.pkl','wb') as f:
#     pickle.dump(distances_seqalign,f)

with open('Murine_seqalign.pkl', 'rb') as f:
    distances_seqalign = pickle.load(f)

distances_seqalign = distances_seqalign + distances_seqalign.T
d_seqalign = distances_seqalign


num = [1, 2, 3, 4, 5, 6, 7,8,9]
reps = [10, 10, 10, 10, 10, 10,10,10,1]

num_list = []
method_list = []
entropy_list = []
num_cluster_list = []

for n, rep in zip(num, reps):
    for r in range(rep):
        classes = np.random.choice(classes_all, n, replace=False)
        sel = np.isin(DTCRU.class_id, classes)

        #VAE-Gene
        d = d_vae_gene[sel,:]
        d = d[:,sel]
        c_freq = phenograph_clustering(d)
        entropy_list.append(entropy(c_freq))
        num_cluster_list.append(len(c_freq))
        num_list.append(n)
        method_list.append('VAE-VDJ')

        #VAE-Seq
        d = d_vae_seq[sel,:]
        d = d[:,sel]
        c_freq = phenograph_clustering(d)
        entropy_list.append(entropy(c_freq))
        num_cluster_list.append(len(c_freq))
        num_list.append(n)
        method_list.append('VAE-Seq')

        #VAE-Seq-Gene
        d = d_vae_seq_gene[sel,:]
        d = d[:,sel]
        c_freq = phenograph_clustering(d)
        entropy_list.append(entropy(c_freq))
        num_cluster_list.append(len(c_freq))
        num_list.append(n)
        method_list.append('VAE-Seq-VDJ')


        #Hamming
        d = d_hamming[sel,:]
        d = d[:,sel]
        c_freq = phenograph_clustering(d)
        entropy_list.append(entropy(c_freq))
        num_cluster_list.append(len(c_freq))
        num_list.append(n)
        method_list.append('Hamming')

        #Kmer
        d = d_kmer[sel,:]
        d = d[:,sel]
        c_freq = phenograph_clustering(d)
        entropy_list.append(entropy(c_freq))
        num_cluster_list.append(len(c_freq))
        num_list.append(n)
        method_list.append('K-mer')

        #Global-Seq-Align
        d = d_seqalign[sel,:]
        d = d[:,sel]
        c_freq = phenograph_clustering(d)
        entropy_list.append(entropy(c_freq))
        num_cluster_list.append(len(c_freq))
        num_list.append(n)
        method_list.append('Global-Seq-Align')


df = pd.DataFrame()
df['Number Of Antigens'] = num_list
df['Structural Entropy'] = entropy_list
df['Number of Clusters'] = num_cluster_list
df['Method'] = method_list

sns.catplot(data=df, x='Number Of Antigens', y='Structural Entropy', kind='point',hue='Method')
plt.xlabel('Number of Antigens',fontsize=16)
plt.ylabel('Structural Entropy',fontsize=16)
sns.catplot(data=df, x='Number Of Antigens', y='Number of Clusters', kind='point',hue='Method')
plt.xlabel('Number of Antigens',fontsize=16)
plt.ylabel('Number of Clusters',fontsize=16)

corr_SE = []
corr_NC = []
methods = []
for m in np.unique(method_list):
    corr,_ = pearsonr(df['Number Of Antigens'][df['Method']==m],df['Structural Entropy'][df['Method']==m])
    corr_SE.append(corr)
    corr,_ = pearsonr(df['Number Of Antigens'][df['Method']==m],df['Number of Clusters'][df['Method']==m])
    corr_NC.append(corr)
    methods.append(m)

df_corr = pd.DataFrame()
df_corr['Methods'] = methods
df_corr['Corr_SE'] = corr_SE
df_corr['Corr_NC'] = corr_NC
df_corr.to_csv('df_corr.csv',index=False)



