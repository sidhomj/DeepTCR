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

gpu = 1
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
folds=25
graph_seed=0
seeds= np.array(range(folds))

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

p = Pool(40)
aucs = []
pred_diff = []
for ii in range(len(group)):
    label_keep = np.array([group[ii], 'CEF','NoPeptide'])
    idx = np.isin(class_labels, label_keep)
    DTCR = DeepTCR_WF('train', device=gpu)
    DTCR.Load_Data(beta_sequences=beta_sequences[idx],
                   v_beta=v_beta[idx],
                   d_beta=d_beta[idx],
                   j_beta=j_beta[idx],
                   counts=counts[idx],
                   class_labels=class_labels[idx],
                   sample_labels=sample_labels[idx], p=p)
    DTCR.Monte_Carlo_CrossVal(folds=folds, graph_seed=graph_seed,seeds=seeds,
                              LOO=len(label_keep), combine_train_valid=True, num_concepts=64,
                              convergence='training', epochs_min=100,train_loss_min=0.25)
    c = np.where(DTCR.lb.classes_ == group[ii])[0][0]
    aucs.append(roc_auc_score(DTCR.y_test[:,c],DTCR.y_pred[:,c]))
    idx_pos = DTCR.y_test[:, c] == 1
    mag = np.mean(DTCR.y_pred[idx_pos, c]) - np.mean(DTCR.y_pred[~idx_pos, c])
    pred_diff.append(mag)

p.close()
p.join()
df_auc = pd.DataFrame()
df_auc['epitope'] = group
df_auc['auc'] = aucs
df_auc['pred_diff'] = pred_diff
df_auc.to_csv('screen.csv',index=False)
check=1

