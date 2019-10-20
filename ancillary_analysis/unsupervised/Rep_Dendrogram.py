"""Figure 1F"""

"""This script is used to create the Repertoire Dendrogram of the Rudqvist_2017 dataset."""

from DeepTCR.DeepTCR import DeepTCR_U

# Instantiate training object
DTCRU = DeepTCR_U('Rep_Dendrogram',device='/device:GPU:1')

#Load Data from directories
DTCRU.Get_Data(directory='../../Data/Rudqvist',Load_Prev_Data=True,aggregate_by_aa=True,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21)

#Train VAE
DTCRU.Train_VAE(accuracy_min=0.9,Load_Prev_Data=True)

#Create Repertoire Dendrogram
color_dict = {'Control':'limegreen','9H10':'red','RT':'darkorange','Combo':'magenta'}
DTCRU.Repertoire_Dendrogram(n_jobs=40,distance_metric='KL',log_scale=True,
                           dendrogram_radius=0.28,repertoire_radius=0.35,Load_Prev_Data=True,gridsize=60,
                            color_dict=color_dict,lw=4,gaussian_sigma=1.0,vmax=0.001)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(0,0,s=5000,edgecolors='magenta',facecolors='none',linewidths=8)
