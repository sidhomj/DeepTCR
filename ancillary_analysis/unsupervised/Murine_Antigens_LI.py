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
import matplotlib
matplotlib.rc('font', family='Arial')

#Instantiate training object
DTCRU = DeepTCR_U('Murine_U_LI')
#Load Data
DTCRU.Get_Data(directory='../../Data/Murine_Antigens',Load_Prev_Data=True,
               aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3)

# # #VAE_- Gene+Sequencs
# DTCRU.Train_VAE(Load_Prev_Data=False,length_invariance=False,use_only_seq=True,accuracy_min=0.80)
# distances_vae = pdist(DTCRU.features, metric='euclidean')

DTCRU.Train_VAE(Load_Prev_Data=False,length_invariance=True,use_only_seq=True,accuracy_min=0.80)
distances_vae_li = pdist(DTCRU.features, metric='euclidean')

# distances_list = [distances_vae,distances_vae_li]
# names = ['VAE-Seq','VAE-Seq-LI']

distances_list = [distances_vae_li]
names = ['VAE-Seq-LI']

dir_results = 'Murine_Results_LI'
if not os.path.exists(dir_results):
    os.makedirs(dir_results)


#Assess Clustering Quality of Various Methods
df_cq = Clustering_Quality(distances_list,names,DTCRU.class_id)
df_cq.to_csv(os.path.join(dir_results,'data_fig1b.csv'))
fig,ax = plt.subplots()
sns.scatterplot(data=df_cq,x='Variance Ratio Criteria',y='Adjusted Mutual Information',s=200,
                hue='Algorithm',alpha=0.5,linewidth=.25,ax=ax)
plt.xlabel('Variance Ratio Criterion',fontsize=18)
plt.ylabel('Adjusted Mutual Information',fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Clustering Quality',fontsize=22)
plt.tight_layout()
plt.savefig(os.path.join(dir_results,'Clutering_Quality.eps'))

#Assess performance metrtics via K-Nearest Neighbors
df_metrics = Assess_Performance_KNN(distances_list,names,DTCRU.class_id,dir_results)
df_metrics.to_csv(os.path.join(dir_results,'data_fig1c.csv'))
df_metrics = pd.read_csv(os.path.join(dir_results,'data_fig1c.csv'))
# Plot_Performance(df_metrics,dir_results)

subdir = 'Performance_Summary'
if not os.path.exists(os.path.join(dir_results,subdir)):
    os.makedirs(os.path.join(dir_results,subdir))

order = ['VAE-Seq','VAE-Seq-LI']
for m in np.unique(df_metrics['Metric']):
    sns.catplot(data=df_metrics[df_metrics['Metric']==m],x='Algorithm',y='Value',kind='violin',order=order)
    plt.ylabel(m)
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.subplots_adjust(bottom=0.25)
    plt.ylabel(m,fontsize=14)
    plt.xticks(fontsize=12)
    plt.savefig(os.path.join(dir_results,subdir,m+'.eps'))

method = 'AUC'
for ii in range(len(names)):
    from scipy.stats import ttest_rel
    df_test = df_metrics[df_metrics['Metric']==method]
    idx_1 = df_test['Algorithm'] == names[ii]
    idx_2 = df_test['Algorithm'] == names[ii+1]
    t,p_val = ttest_rel(df_test[idx_1]['Value'],df_test[idx_2]['Value'])
    print(p_val)

from scipy.stats import ttest_rel
df_test = df_metrics[df_metrics['Metric']==method]
idx_1 = df_test['Algorithm'] == names[5]
idx_2 = df_test['Algorithm'] == names[0]
t,p_val = ttest_rel(df_test[idx_1]['Value'],df_test[idx_2]['Value'])
print(p_val)


#Assess Length Dependency of various methods
SRCC = []
for n,distances in zip(names,distances_list):
    len_distance = pdist(np.sum(DTCRU.X_Seq_beta>0,-1))
    len_seq = np.sum(DTCRU.X_Seq_beta>0,-1)
    corr,_ = spearmanr(len_distance,distances)
    SRCC.append(corr)
    df = pd.DataFrame()
    df['D_len'] = len_distance.astype(int)
    df['D_features'] = distances
    plt.figure()
    sns.boxplot(x='D_len',y='D_features',data=df)
    plt.title(n)
    plt.savefig(os.path.join(dir_results,n+'_box_LD.eps'))

df_out = pd.DataFrame()
df_out['Methods'] = names
df_out['SRCC'] = SRCC
df_out.to_csv(os.path.join(dir_results,'Length_Dep.csv'),index=False)

DTCRU.Cluster(clustering_method='kmeans',t=100)
