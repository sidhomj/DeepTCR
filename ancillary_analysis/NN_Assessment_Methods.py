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
from multiprocessing import Pool
from sklearn.manifold import MDS

#Instantiate training object
DTCRU = DeepTCR_U('Metrics')

"Sidhom"
dir_results = 'Sidhom_Figures'

p = Pool(40)
#VAE_- Sequencs Alone
DTCRU.Get_Data(directory='../Data/Sidhom',Load_Prev_Data=False,aggregate_by_aa=True,aa_column_beta=1,count_column=None,
               v_beta_column=None,d_beta_column=None,j_beta_column=None,p=p)
DTCRU.Train_VAE(accuracy_min=0.9, Load_Prev_Data=False)
features_vae_seq = DTCRU.features
distances_vae_seq = squareform(pdist(DTCRU.features, metric='euclidean'))

#VAE_- Gene+Sequencs
DTCRU.Get_Data(directory='../Data/Sidhom',Load_Prev_Data=False,aggregate_by_aa=True,aa_column_beta=1,count_column=None,
               v_beta_column=7,d_beta_column=14,j_beta_column=21,p=p)
DTCRU.Train_VAE(accuracy_min=0.9, Load_Prev_Data=False,seq_features_latent=True)
features_vae_seq_gene = DTCRU.features
distances_vae_seq_gene = squareform(pdist(DTCRU.features, metric='euclidean'))
p.close()
p.join()

#Hamming
distances_hamming = squareform(pdist(np.squeeze(DTCRU.X_Seq_beta, 1), metric='hamming'))

#Kmer
kmer_features = kmer_search(DTCRU.beta_sequences)
distances_kmer = squareform(pdist(kmer_features, metric='euclidean'))

#Seq-Align
# start = time.time()
# distances_seqalign = pairwise_alignment(DTCRU.beta_sequences)
# end = time.time()
# total_time = end-start
# with open('Sidhom_seqalign.pkl','wb') as f:
#     pickle.dump([distances_seqalign,total_time],f)

with open('Sidhom_seqalign.pkl','rb') as f:
    distances_seqalign,total_time = pickle.load(f)
#make symmetric
distances_seqalign = distances_seqalign + distances_seqalign.T

#
# methods = [distances_vae_seq, distances_vae_seq_gene, distances_hamming, distances_kmer,distances_seqalign]
# names = ['VAE-Seq','VAE-Seq-Gene','Hamming','K-mer','Global Seq-Align']
#
# import pandas as pd
# from scipy.stats import wasserstein_distance
# df_distances = pd.DataFrame()
# intra = []
# inter = []
# w = []
# for distances,n in zip(methods,names):
#     labels = DTCRU.label_id
#     intra_distances = []
#     inter_distances = []
#     for c in np.unique(labels):
#         idx = labels==c
#         d = distances[idx,:]
#         d = d[:,idx]
#         intra_distances.append(np.ndarray.flatten(d))
#         d=distances[idx,:]
#         d = d[:,~idx]
#         inter_distances.append(np.ndarray.flatten(d))
#
#     intra_distances = np.hstack(intra_distances)
#     inter_distances = np.hstack(inter_distances)
#     w.append(wasserstein_distance(intra_distances, inter_distances))
#
#     intra.append(np.average(intra_distances))
#     inter.append(np.average(inter_distances))
#
# df_distances['Methods'] = names
# df_distances['Intra'] = intra
# df_distances['Inter'] = inter
# df_distances['Waserstein'] = w
#
#
#
#
# from scipy.stats import kstest
#
# kstest(intra_distances,inter_distances)
#
# plt.figure()
# plt.hist(inter_distances,50,alpha=0.5)
# plt.hist(intra_distances,50,alpha=0.5)
# plt.legend()
#
# break
# distances[]

df = Assess_Performance_KNN(DTCRU,distances_vae_seq, distances_vae_seq_gene, distances_hamming, distances_kmer,distances_seqalign,dir_results)

# #Plot Performance
Plot_Performance(df,dir_results)

#Plot Latent Space
methods = [distances_vae_seq, distances_vae_seq_gene, distances_hamming, distances_kmer,distances_seqalign]
Plot_Latent(DTCRU.label_id,methods,dir_results)

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



