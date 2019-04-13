from DeepTCR.DeepTCR import DeepTCR_SS, DeepTCR_U
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import seaborn as sns
from NN_Assessment_utils import *
import pickle
import os

#Run VAE
DTCRU = DeepTCR_U('Sequence_C',device='/gpu:1')
DTCRU.Get_Data(directory='../../Data/Murine_Antigens',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3)

DTCRU.Train_VAE(Load_Prev_Data=False,use_only_gene=True)
distances_vae_gene = pdist(DTCRU.features, metric='euclidean')

# #VAE_- Sequencs Alone+
DTCRU.Train_VAE(Load_Prev_Data=False,use_only_seq=True)
distances_vae_seq = pdist(DTCRU.features, metric='euclidean')

DTCRU.Train_VAE(Load_Prev_Data=False)
distances_vae_seq_gene = pdist(DTCRU.features, metric='euclidean')

distances_list = [distances_vae_seq,distances_vae_gene,distances_vae_seq_gene]
names = ['VAE-Seq','VAE-VDJ','VAE-Seq-VDJ']

dir_results = 'sup_v_unsup_results'
if not os.path.exists(dir_results):
    os.makedirs(dir_results)

df_metrics = Assess_Performance_KNN(distances_list,names,DTCRU.class_id,dir_results,metrics=['AUC'])

df_u = pd.DataFrame()
df_u['Class'] = df_metrics['Classes']
df_u['AUC'] = df_metrics['Value']
df_u['Method'] = df_metrics['Algorithm']
df_u['Type'] = 'Unsupervised'

#Run Supervised Sequence Classifier
DTCRS = DeepTCR_SS('Sequence_C')
DTCRS.Get_Data(directory='../../Data/Murine_Antigens',Load_Prev_Data=True,aggregate_by_aa=True,
               aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3)

AUC = []
Class = []
Method = []
for i in range(10):
    DTCRS.Get_Train_Valid_Test()

    DTCRS.Train(use_only_seq=True,num_fc_layers=1,units_fc=256)
    DTCRS.AUC_Curve(plot=False)
    AUC.extend(DTCRS.AUC_DF['AUC'].tolist())
    Class.extend(DTCRS.AUC_DF['Class'].tolist())
    Method.extend(['Sup-Seq']*len(DTCRS.AUC_DF))

    DTCRS.Train(use_only_gene=True,num_fc_layers=1,units_fc=256)
    DTCRS.AUC_Curve(plot=False)
    AUC.extend(DTCRS.AUC_DF['AUC'].tolist())
    Class.extend(DTCRS.AUC_DF['Class'].tolist())
    Method.extend(['Sup-VDJ']*len(DTCRS.AUC_DF))

    DTCRS.Train(num_fc_layers=1,units_fc=256)
    DTCRS.AUC_Curve(plot=False)
    AUC.extend(DTCRS.AUC_DF['AUC'].tolist())
    Class.extend(DTCRS.AUC_DF['Class'].tolist())
    Method.extend(['Sup-Seq-VDJ']*len(DTCRS.AUC_DF))



df_s = pd.DataFrame()
df_s['Class'] = Class
df_s['AUC'] = AUC
df_s['Method'] = Method
df_s['Type'] = 'Supervised'


df_comp = pd.concat((df_u,df_s),axis=0)

sns.violinplot(data=df_comp,x='Class',y='AUC',hue='Method')
plt.xticks(rotation=45)
plt.xlabel('')
plt.ylabel('AUC',fontsize=28)
plt.xticks(fontsize=18)
plt.subplots_adjust(bottom=0.15)
ax = plt.gca()
ax.legend().remove()
plt.legend(fontsize=24)





