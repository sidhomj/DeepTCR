from DeepTCR.DeepTCR import DeepTCR_WF
import glob
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool

gpu = 3
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
folds=25

files = glob.glob('Data/*.tsv')
files = files[0:-1]
samples = []
labels = []
for file in files:
    file = file.split('/')[-1]
    samples.append(file)
    labels.append(file.split('_')[1])

label_dict = dict(zip(samples,labels))

DTCR = DeepTCR_WF('load')
DTCR.Get_Data('Data',aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,
              type_of_data_cut='Read_Cut',data_cut=10)

idx = np.isin(DTCR.sample_id,np.array(list(label_dict.keys())))
beta_sequences = DTCR.beta_sequences[idx]
v_beta = DTCR.v_beta[idx]
d_beta = DTCR.d_beta[idx]
j_beta = DTCR.j_beta[idx]
sample_labels = DTCR.sample_id[idx]
counts = DTCR.counts[idx]
class_labels  = np.array([label_dict[x] for x in sample_labels])

group = ['FFPDWQNYT','YFPDWQNYT','NoPeptide']

aucs = np.zeros([len(group),len(group)])
p = Pool(40)
#pairwise
for ii in range(len(group)):
    for jj in range(ii,len(group)):
        if ii != jj:
            label_keep = np.array([group[ii],group[jj]])
            idx = np.isin(class_labels, label_keep)
            DTCR = DeepTCR_WF('train', device='/device:GPU:3')
            DTCR.Load_Data(beta_sequences=beta_sequences[idx],
                           v_beta=v_beta[idx],
                           d_beta=d_beta[idx],
                           j_beta=j_beta[idx],
                           counts=counts[idx],
                           class_labels=class_labels[idx],
                           sample_labels=sample_labels[idx],p=p)
            hinge_loss_t = -np.log(1 / len(label_keep)) / 2
            DTCR.Monte_Carlo_CrossVal(folds=folds, LOO=len(label_keep), combine_train_valid=True, num_concepts=64,
                                      convergence='training', epochs_min=100)
            aucs[ii,jj] = roc_auc_score(DTCR.y_test,DTCR.y_pred)

p.close()
p.join()

import pickle
with open('aucs_nefyt9.pkl','wb') as f:
    pickle.dump([aucs,group],f,protocol=4)