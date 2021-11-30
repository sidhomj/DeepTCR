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
DTCRU = DeepTCR_U('Murine_U')
#Load Data
DTCRU.Get_Data(directory='../../Data/Murine_Antigens',Load_Prev_Data=False,
               aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3)

DTCRU.Train_VAE(Load_Prev_Data=False,use_only_seq=True)
distances_vae = pdist(DTCRU.features, metric='euclidean')

DTCRU.Train_VAE(Load_Prev_Data=False,masked_input=True,mask_rate=0.10,use_only_seq=True)
distances_vae_mask = pdist(DTCRU.features, metric='euclidean')

distances_list = [distances_vae,distances_vae_mask]
names = ['VAE','Masked VAE']

dir_results = 'Murine_Results_Masked'
if not os.path.exists(dir_results):
    os.makedirs(dir_results)

#Assess performance metrtics via K-Nearest Neighbors
df_metrics = Assess_Performance_KNN(distances_list,names,DTCRU.class_id,dir_results)
df_metrics.to_csv(os.path.join(dir_results,'data_fig1c.csv'))
df_metrics = pd.read_csv('data_fig1c.csv')
Plot_Performance(df_metrics,dir_results)

subdir = 'Performance_Summary_Masked'
if not os.path.exists(os.path.join(dir_results,subdir)):
    os.makedirs(os.path.join(dir_results,subdir))

order =['VAE','Masked VAE']
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


