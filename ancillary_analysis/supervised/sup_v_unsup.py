"""Supplementary Fig.16"""

"""This script it sued to direclty compare the performance for sequence classification
from the unsupervised VAE + KNN vs a supervised deep learning sequence classifier."""

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
from scipy.stats import ttest_ind
import pandas as pd
import matplotlib
matplotlib.rc('font', family='Arial')

load_prev_data = False

#Get VAE Data
dir_results = '../unsupervised/Murine_Results'
file_write = os.path.join(dir_results,'data_fig1b.csv')
df_metrics = pd.read_csv(file_write)
df_metrics = df_metrics[df_metrics['Algorithm']=='VAE-Seq-VDJ']

df_u = pd.DataFrame()
df_u['Class'] = df_metrics['Classes']
df_u['AUC'] = df_metrics['Value']
df_u['Method'] = df_metrics['Algorithm']
df_u['Type'] = 'Unsupervised'

dir_results = 'Sup_V_Unsup_Results'
if not os.path.exists(dir_results):
    os.makedirs(dir_results)

#Run Supervised Sequence Classifier
if load_prev_data is False:
    DTCRS = DeepTCR_SS('Sequence_C')
    DTCRS.Get_Data(directory='../../Data/Murine_Antigens',Load_Prev_Data=True,aggregate_by_aa=True,
                   aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3)

    AUC = []
    Class = []
    Method = []
    for i in range(100):
        DTCRS.Get_Train_Valid_Test()
        DTCRS.Train()
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
    df_comp.to_csv(os.path.join(dir_results,'df_comp.csv'))
else:
    df_comp = pd.read_csv(os.path.join(dir_results,'df_comp.csv'))

fig,ax = plt.subplots(figsize=(15,10))
sns.violinplot(data=df_comp,x='Class',y='AUC',hue='Method',cut=0,ax=ax)
plt.xticks(rotation=45)
plt.xlabel('')
plt.ylabel('AUC',fontsize=28)
plt.xticks(fontsize=22)
plt.yticks(fontsize=18)
plt.subplots_adjust(bottom=0.15)
ax.legend().remove()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(fontsize=24)

antigens =['Db-F2', 'Db-M45', 'Db-NP', 'Db-PA', 'Db-PB1', 'Kb-M38', 'Kb-SIY','Kb-TRP2', 'Kb-m139']
p_val_list = []
for antigen in antigens:
    df_test = df_comp[df_comp['Class'] == antigen]
    idx_1 = df_test['Method'] == 'VAE-Seq-VDJ'
    idx_2 = df_test['Method'] == 'Sup-Seq-VDJ'
    t,p_val = ttest_ind(df_test[idx_1]['AUC'],df_test[idx_2]['AUC'])
    print(p_val)
    p_val_list.append(p_val)

df_pval = pd.DataFrame()
df_pval['antigen'] = antigens
df_pval['P_Val'] = p_val_list

