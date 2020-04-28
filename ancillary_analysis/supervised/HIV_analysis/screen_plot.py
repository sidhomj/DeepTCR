from DeepTCR.DeepTCR import DeepTCR_WF
import glob
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import pandas as pd

files = glob.glob('../../../Data/HIV/*.tsv')
samples = []
labels = []
for file in files:
    file = file.split('/')[-1]
    samples.append(file)
    labels.append(file.split('_')[1])

label_dict = dict(zip(samples,labels))

DTCR = DeepTCR_WF('load')
DTCR.Get_Data('../../../Data/HIV',aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,
              type_of_data_cut='Read_Cut',data_cut=10)

idx = np.isin(DTCR.sample_id,np.array(list(label_dict.keys())))
beta_sequences = DTCR.beta_sequences[idx]
v_beta = DTCR.v_beta[idx]
d_beta = DTCR.d_beta[idx]
j_beta = DTCR.j_beta[idx]
sample_labels = DTCR.sample_id[idx]
counts = DTCR.counts[idx]
class_labels  = np.array([label_dict[x] for x in sample_labels])

group_1 = ['TSNLQEQIAW', 'TSNLQEQIGW', 'TSTLAEQIAW', 'TSTLAEQMAW',
       'TSTLAEQVAW', 'TSTLQEQIEW', 'TSTLQEQIGW', 'TSTLSEQIAW',
       'TSTLSEQVAW', 'TSTLTEQIAW', 'TSTLTEQVAW', 'TSTLVEQIAW']
group_2 = ['ISPRTLNAW', 'MSPRTLNAW']
group_3 =  ['KIRLRPGGKKKYKLK', 'KIRLRPGGKKRYKLK']
group_4 = ['KAALDLSHF','KAAVDLSHF', 'KGALDLSHF','KSALDLSHF','TAALDMSHF']
group_5 = ['HTQGYFPDW','NTQGYFPDW']
group_6 =  ['FFPDWQNYT','YFPDWQNYT']
group = np.hstack([group_1,group_2,group_3,group_4,group_5,group_6])
df_screen = pd.read_csv('screen.csv')
df_screen = df_screen.set_index('epitope').loc[group].reset_index()

group = [group_1,group_2,group_3,group_4,group_5,group_6]
group_name = ['GAG TW10','GAG IW9', 'GAG KK15','NEF KF9','NEF HW9','NEF YT9']
class_dict = {}
for n,g in zip(group_name,group):
    for e in g:
        class_dict[e] = n
df_screen['Epitope Family'] = df_screen['epitope'].map(class_dict)
df_screen.to_csv('screen_plot.csv',index=False)

ax = sns.scatterplot(data=df_screen,x='pred_diff',y='auc',hue='Epitope Family',s=100,hue_order=group_name)
plt.axhline(0.90,color='grey',linewidth=3,linestyle='dashed')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:])
plt.setp(plt.gca().get_legend().get_texts(), fontsize='18')
ax.get_legend().draw_frame(False)
plt.xlabel('Delta Prediction', fontsize=24)
plt.ylabel('AUC', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.savefig('screen.eps')

# df_screen.sort_values(by='pred_diff',inplace=True,ascending=False)
df_screen.sort_values(by='pred_diff',inplace=True,ascending=False)
df_screen.reset_index(drop=True,inplace=True)

df_sel = df_screen[df_screen['auc'] > 0.90]