from DeepTCR.DeepTCR_U import DeepTCR_U
from scipy.spatial.distance import pdist, squareform
from NN_Assessment_utils import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pickle
import umap
import time

#Instantiate training object
DTCRU = DeepTCR_U('Metrics')

"Sidhom"
dir_results = 'Sidhom_Figures'
DTCRU.Get_Data(directory='../Data/Sidhom',Load_Prev_Data=False,aggregate_by_aa=True,aa_column_beta=1,count_column=None,
               v_beta_column=7,d_beta_column=14,j_beta_column=21)

#VAE_GAN
distances_vae,distances_gan = VAE_GAN_Distances(DTCRU,Load_Prev_Data=True)

#Hamming
distances_hamming = squareform(pdist(np.squeeze(DTCRU.X_Seq_beta, 1), metric='hamming'))

#Kmer
kmer_features = kmer_search(DTCRU.beta_sequences)
distances_kmer = squareform(pdist(kmer_features, metric='euclidean'))

#Seq-Align
# start = time.time()
# distance_seqalign = pairwise_alignment(DTCRU.beta_sequences)
# end = time.time()
# total_time = end-start
# with open('Sidhom_seqalign.pkl','wb') as f:
#     pickle.dump([distance_seqalign,total_time],f)

with open('Sidhom_seqalign.pkl','rb') as f:
    distance_seqalign,total_time = pickle.load(f)

df = Assess_Performance(DTCRU,distances_vae, distances_gan, distances_hamming, distances_kmer,distance_seqalign,dir_results)
agg_dict = {'Recall':'mean','Precision':'mean','F1_Score':'mean','Accuracy':'mean','AUC':'mean'}
df_sum = df.groupby(['Algorithm']).agg(agg_dict)

#Plot Performance
Plot_Performance(df)

#Plot Latent Space
labels = LabelEncoder().fit_transform(DTCRU.label_id)
methods = [distances_gan, distances_vae, distances_hamming, distances_kmer]
Plot_Latent(labels,methods)

import pickle
with open('distances.pkl','wb') as f:
    pickle.dump([DTCRU.label_id,distances_vae, distances_gan, distances_hamming, distances_kmer],f)


# "Glanville"
# dir_results = 'Glanville_Figures'
# DTCRU.Get_Data(directory='../Data/Glanville',Load_Prev_Data=False,aggregate_by_aa=True,aa_column_beta=1,count_column=2)
#
# #VAE_GAN
# distances_vae,distances_gan = VAE_GAN_Distances(DTCRU,Load_Prev_Data=False)
#
# #Hamming
# distances_hamming = squareform(pdist(np.squeeze(DTCRU.X_Seq_beta, 1), metric='hamming'))
#
# #Kmer
# kmer_features = kmer_search(DTCRU.beta_sequences)
# distances_kmer = squareform(pdist(kmer_features, metric='euclidean'))
#
# df = Assess_Performance(DTCRU,distances_vae, distances_gan, distances_hamming, distances_kmer,dir_results)
#
# #Plot Performance
# Plot_Performance(df)
#
# #Plot Latent Space
# labels = LabelEncoder().fit_transform(DTCRU.label_id)
# methods = [distances_gan, distances_vae, distances_hamming, distances_kmer]
# Plot_Latent(labels,methods)


"Dash"
dir_results = 'Dash_Figures'
DTCRU.Get_Data(directory='../Data/Dash/Traditional/Mouse',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_alpha=0,aa_column_beta=1,count_column=2,v_alpha_column=None,j_alpha_column=None,v_beta_column=None,j_beta_column=None)

#VAE_GAN
distances_vae,distances_gan = VAE_GAN_Distances(DTCRU,Load_Prev_Data=False)

#Hamming
distances_hamming_alpha = squareform(pdist(np.squeeze(DTCRU.X_Seq_alpha, 1), metric='hamming'))
distances_hamming_beta = squareform(pdist(np.squeeze(DTCRU.X_Seq_beta, 1), metric='hamming'))
distances_hamming = distances_hamming_alpha+distances_hamming_beta

#Kmer
kmer_features_alpha = kmer_search(DTCRU.alpha_sequences)
kmer_features_beta = kmer_search(DTCRU.beta_sequences)
kmer_features =  np.concatenate((kmer_features_alpha,kmer_features_beta),axis=1)
distances_kmer = squareform(pdist(kmer_features, metric='euclidean'))

df = Assess_Performance(DTCRU,distances_vae, distances_gan, distances_hamming, distances_kmer,dir_results,use_genes_label='use_genes_true')

#Plot Performance
Plot_Performance(df)

#Plot Latent Space
labels = LabelEncoder().fit_transform(DTCRU.label_id)
methods = [distances_gan, distances_vae, distances_hamming, distances_kmer]
Plot_Latent(labels,methods)



