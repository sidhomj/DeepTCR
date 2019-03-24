from DeepTCR.DeepTCR import DeepTCR_U
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from multiprocessing import Pool
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from NN_Assessment_utils import *


#Instantiate training object
DTCRU = DeepTCR_U('Reperoire_Distances')

#Assess ability for structural entropy to be of measure of number of antigens
classes_all = np.array(['F2', 'M38', 'M45', 'NP', 'PA', 'PB1', 'm139'])

p = Pool(40)
num = [1,2,3,4,5,6,7]
num = [1,3,7]
reps = [1,1,1]

num_list=[]
method_list = []
entropy_list = []
distances_vae_seq_l = []
distances_vae_gene_l = []
distances_vae_seq_gene_l = []
distances_hamming_l = []
distances_kmer_l = []

for n,rep in zip(num,reps):
    distances_vae_seq_l_temp = []
    distances_vae_gene_l_temp = []
    distances_vae_seq_gene_l_temp = []
    distances_hamming_l_temp = []
    distances_kmer_l_temp = []
    for r in range(rep):
        classes = np.random.choice(classes_all,n,replace=False)
        DTCRU = DeepTCR_U('Reperoire_Distances')
        DTCRU.Get_Data(directory='../Data/Dash/Traditional/Mouse',Load_Prev_Data=False,aggregate_by_aa=True,classes=classes,
                            aa_column_alpha=0,aa_column_beta=1,count_column=2,v_alpha_column=None,j_alpha_column=None,v_beta_column=None,j_beta_column=None,p=p)
        DTCRU.Train_VAE(accuracy_min=0.9)
        distances_vae_seq_l_temp.append(pdist(DTCRU.features))
        #entropy_list.append(entropy(pdist(DTCRU.features)))
        method_list.append('VAE-Seq')
        num_list.append(n)

        DTCRU.Get_Data(directory='../Data/Dash/Traditional/Mouse',Load_Prev_Data=False,aggregate_by_aa=True,classes=classes,
                            aa_column_alpha=0,aa_column_beta=1,count_column=2,v_alpha_column=3,j_alpha_column=4,v_beta_column=5,j_beta_column=6,p=p)
        DTCRU.Train_VAE(accuracy_min=0.9)
        distances_vae_seq_gene_l_temp.append(pdist(DTCRU.features))
        #entropy_list.append(entropy(pdist(DTCRU.features)))
        method_list.append('VAE-Seq-Gene')
        num_list.append(n)

        # Hamming
        distances_hamming_alpha = squareform(pdist(np.squeeze(DTCRU.X_Seq_alpha, 1), metric='hamming'))
        distances_hamming_beta = squareform(pdist(np.squeeze(DTCRU.X_Seq_beta, 1), metric='hamming'))
        distances_hamming = distances_hamming_alpha + distances_hamming_beta
        distances_hamming_l_temp.append(squareform(distances_hamming))
        #entropy_list.append(entropy(squareform(distances_hamming)))
        method_list.append('Hamming')
        num_list.append(n)


        # Kmer
        kmer_features_alpha = kmer_search(DTCRU.alpha_sequences)
        kmer_features_beta = kmer_search(DTCRU.beta_sequences)
        kmer_features = np.concatenate((kmer_features_alpha, kmer_features_beta), axis=1)
        distances_kmer_l_temp.append(pdist(kmer_features))
        #entropy_list.append(entropy(pdist(kmer_features, metric='euclidean')))
        method_list.append('K-Mer')
        num_list.append(n)

        DTCRU = DeepTCR_U('Reperoire_Distances')
        DTCRU.Get_Data(directory='../Data/Dash/Traditional/Mouse',Load_Prev_Data=False,aggregate_by_aa=False,classes=classes,
                            aa_column_alpha=None,aa_column_beta=None,count_column=2,v_alpha_column=3,j_alpha_column=4,v_beta_column=5,j_beta_column=6,p=p)
        DTCRU.Train_VAE(accuracy_min=0.9)
        distances_vae_gene_l_temp.append(pdist(DTCRU.features))
        method_list.append('VAE-Gene')
        num_list.append(n)

    distances_vae_seq_l.append(np.hstack(distances_vae_seq_l_temp))
    distances_vae_gene_l.append(np.hstack(distances_vae_gene_l_temp))
    distances_vae_seq_gene_l.append(np.hstack(distances_vae_seq_gene_l_temp))
    distances_hamming_l.append(np.hstack(distances_hamming_l_temp))
    distances_kmer_l.append(np.hstack(distances_kmer_l_temp))

#Plot distances

plt.figure()
for ii,n in enumerate(num,0):
    sns.distplot(distances_vae_seq_gene_l[ii],label=str(n),norm_hist=True)
plt.legend()
plt.title('VAE-Seq-Gene')

plt.figure()
for ii,n in enumerate(num,0):
    sns.distplot(distances_vae_seq_l[ii],label=str(n),norm_hist=True)
plt.legend()
plt.title('VAE-Seq')

plt.figure()
for ii,n in enumerate(num,0):
    sns.distplot(distances_vae_gene_l[ii],label=str(n),norm_hist=True)
plt.legend()
plt.title('VAE-Gene')

plt.figure()
for ii,n in enumerate(num,0):
    sns.distplot(distances_hamming_l[ii],label=str(n),norm_hist=True)
plt.legend()
plt.title('Hamming')

plt.figure()
for ii,n in enumerate(num,0):
    sns.distplot(distances_kmer_l[ii],label=str(n),norm_hist=True)
plt.legend()
plt.title('K-Mer')

df = pd.DataFrame()
df['Number Of Antigens'] = num_list
df['Structural Entropy'] = entropy_list
df['Method'] = method_list
sns.catplot(data=df,x='Number Of Antigens',y='Structural Entropy',kind='point',hue='Method')

#
# DTCRU.Get_Data(directory='../Data/Rudqvist',Load_Prev_Data=False,aggregate_by_aa=True,
#                aa_column_alpha=None,aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21)
# DTCRU.Train_VAE(seq_features_latent=True,accuracy_min=0.9)
# DTCRU.Structural_Entropy(plot=True)
# sns.boxplot(data=DTCRU.Entropy_DF,x='Label',y='Entropy',order=['Control','9H10','RT','Combo'])

DTCRU.Get_Data(directory='../Data/Ribas_Pre',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_alpha=None,aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21)
DTCRU.Train_VAE(seq_features_latent=True,accuracy_min=0.9)
DTCRU.Structural_Entropy(plot=True)
check=1



