"""This script is used to assess the performance of various featurization methods
at the sample/repertoire level on TCRSeq data from tumor infiltrating lymphocytes (TIL)
taken from mice treated with different therapies."""

from DeepTCR.DeepTCR import DeepTCR_U
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from NN_Assessment_utils import *
import pickle
import os
import matplotlib.pyplot as plt

dir_results = 'Rudqvist_U_Sample_Results'
if not os.path.exists(dir_results):
    os.makedirs(dir_results)

# Instantiate training object
DTCRU = DeepTCR_U('Reperoire_Distances')

DTCRU.Get_Data(directory='../Data/Rudqvist',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21)
#
# # VAE-Gene
# DTCRU.Train_VAE(use_only_gene=True)
# d_vae_gene = squareform(pdist(DTCRU.features))
# prop_vae_gene,_ = phenograph_clustering_freq(d_vae_gene,DTCRU)
# #
# # #
# # VAE-Seq
# DTCRU.Train_VAE(use_only_seq=True)
# d_vae_seq = squareform(pdist(DTCRU.features))
# prop_vae_seq,_ = phenograph_clustering_freq(d_vae_seq,DTCRU)
# #
# #
# VAE-Seq-Gene
DTCRU.Train_VAE(Load_Prev_Data=False)
color_dict = {'Control':'g','RT':'r','9H10':'y','Combo':'b'}
DTCRU.Repertoire_Dendogram(distance_metric='KL',color_dict=color_dict,Load_Prev_Data=True)
# d_vae_seq_gene = squareform(pdist(DTCRU.features))
# prop_vae_seq_gene,_ = phenograph_clustering_freq(d_vae_seq_gene,DTCRU)
# #
# # Hamming
# d_hamming = squareform(pdist(np.squeeze(DTCRU.X_Seq_beta, 1), metric='hamming'))
# prop_hamming,_ = phenograph_clustering_freq(d_hamming,DTCRU)
#
# # #Kmer
# # # kmer_features = kmer_search(DTCRU.beta_sequences)
# # # d_kmer = squareform(pdist(kmer_features, metric='euclidean'))
# # #
# # # # with open('Rudqvist_kmer.pkl','wb') as f:
# # # #     pickle.dump(d_kmer,f,protocol=4)
# # #
# with open('Rudqvist_kmer.pkl','rb') as f:
#     d_kmer = pickle.load(f)
#
# prop_kmer,_ = phenograph_clustering_freq(d_kmer,DTCRU)
#
# # # Global Seq-Align
# # d_seqalign = pairwise_alignment(DTCRU.beta_sequences)
# # with open('Rudqvist_seqalign.pkl','wb') as f:
# #     pickle.dump(d_seqalign,f,protocol=4)
# #
# with open('Rudqvist_seqalign.pkl', 'rb') as f:
#     d_seqalign = pickle.load(f)
#
# d_seqalign = d_seqalign + d_seqalign.T
# prop_seqalign,_ = phenograph_clustering_freq(d_seqalign,DTCRU)
#
# #
# prop_list = [prop_vae_seq,prop_vae_gene,prop_vae_seq_gene,prop_hamming,prop_kmer,prop_seqalign]
# names = ['VAE-Seq','VAE-Gene','VAE-Seq-Gene','Hamming','K-mer','Global-Seq-Align']
# # #
# with open('Prop.pkl','wb') as f:
#     pickle.dump([prop_list,names],f)

with open('Prop.pkl','rb') as f:
    prop_list,names = pickle.load(f)


distances_list, distances_names, method_names = Get_Prop_Distances(prop_list,names,eps=1e-30)

labels = []
for i in prop_list[0].index:
    labels.append(DTCRU.class_id[np.where(DTCRU.sample_id == i)[0][0]])

df_metrics = Assess_Performance_KNN_Samples(distances_list,distances_names,method_names,dir_results,labels)
Plot_Performance_Samples(df_metrics,dir_results)

#Determine best distance metric across all algorithms
df_temp = df_metrics[df_metrics['Metric']=='AUC']
#df_temp = df_temp[df_temp['Classes']=='Combo']
fig,ax = plt.subplots()
sns.violinplot(data=df_temp,x='Distance Metric',y='Value',hue='Algorithm',ax=ax)
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
ax.get_legend().remove()

subdir = 'Performance_Summary'
if not os.path.exists(os.path.join(dir_results,subdir)):
    os.makedirs(os.path.join(dir_results,subdir))

for m in np.unique(df_metrics['Metric']):
    sns.catplot(data=df_metrics[df_metrics['Metric']==m],x='Algorithm',y='Value',kind='violin')
    plt.ylabel(m)
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.subplots_adjust(bottom=0.25)
    plt.ylabel(m,fontsize=14)
    plt.xticks(fontsize=12)
    plt.savefig(os.path.join(dir_results,subdir,m+'.eps'))

method = 'F1_Score'
from scipy.stats import ttest_rel
df_test = df_metrics[df_metrics['Metric']==method]
idx_1 = df_test['Algorithm'] == 'K-mer'
idx_2 = df_test['Algorithm'] == 'Hamming'
t,p_val = ttest_rel(df_test[idx_1]['Value'],df_test[idx_2]['Value'])
print(p_val)