from DeepTCR.DeepTCR import DeepTCR_WF
import glob
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import pandas as pd

gpu = 2
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
folds=100
graph_seed=0
seeds= np.array(range(folds))

files = glob.glob('../../../Data/HIV/*.tsv')
files = files[0:-1]
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

group = ['ISPRTLNAW', 'MSPRTLNAW']

aucs = np.zeros([len(group),len(group)])
preds =  np.zeros([len(group),len(group)])
p = Pool(40)
#pairwise
for ii in range(len(group)):
    for jj in range(ii,len(group)):
        if ii != jj:
            label_keep = np.array([group[ii],group[jj]])
            idx = np.isin(class_labels, label_keep)
            DTCR = DeepTCR_WF('train', device=gpu)
            DTCR.Load_Data(beta_sequences=beta_sequences[idx],
                           counts=counts[idx],
                           class_labels=class_labels[idx],
                           sample_labels=sample_labels[idx],p=p)
            DTCR.Monte_Carlo_CrossVal(folds=folds, LOO=len(label_keep), combine_train_valid=True, num_concepts=64,
                                      convergence='training', train_loss_min=0.10,graph_seed=graph_seed,seeds=seeds)
            aucs[ii,jj] = roc_auc_score(DTCR.y_test,DTCR.y_pred)
            c = 1
            idx_pos = DTCR.y_test[:, c] == 1
            mag = np.mean(DTCR.y_pred[idx_pos, c]) - np.mean(DTCR.y_pred[~idx_pos, c])
            preds[ii,jj] = mag

p.close()
p.join()

import pickle
with open('aucs_gagiw9.pkl','wb') as f:
    pickle.dump([aucs,preds,group],f,protocol=4)

ii = 0
import seaborn as sns
#Preds Distribution
import matplotlib.pyplot as plt
test_peptide = group[jj]
c = np.where(DTCR.lb.classes_ == test_peptide)[0][0]
color_dict = {1:'b',0:'r'}
label_dict = {1:test_peptide,0:group[ii]}
df_preds = pd.DataFrame()
df_preds['y_test'] = DTCR.y_test[:,c]
df_preds['y_pred'] = DTCR.y_pred[:,c]
df_preds['color'] = df_preds['y_test'].map(color_dict)
df_preds['label'] = df_preds['y_test'].map(label_dict)
plt.figure()
sns.violinplot(data=df_preds,x='label',y='y_pred',cut=0,palette=['r','b'])
plt.xlabel('')
plt.ylabel('P('+test_peptide+')',fontsize=24)
plt.xticks(size=18)
plt.yticks(size=18)
plt.ylim([0,1])
plt.subplots_adjust(left=0.15)
plt.savefig('GAGIW9_preds.png',dpi=1200)