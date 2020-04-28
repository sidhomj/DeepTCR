from DeepTCR.DeepTCR import DeepTCR_WF
import glob
import os
import numpy as np
from multiprocessing import Pool
import pickle

gpu = 0
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
folds=100
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

group = ['TSTLAEQIAW', 'TSTLAEQMAW',
       'TSTLAEQVAW', 'TSTLQEQIEW', 'TSTLQEQIGW', 'TSTLSEQIAW',
       'TSTLSEQVAW', 'TSTLTEQIAW', 'TSTLTEQVAW', 'TSTLVEQIAW']
p = Pool(40)
sequences = []
seq_class_labels = []
predicted = []
seq_counts = []
for ii in range(len(group)):
    label_keep = np.array([group[ii], 'CEF','NoPeptide','AY9'])
    idx = np.isin(class_labels, label_keep)
    class_dict = {group[ii]:group[ii],'CEF':'non-Cognate','NoPeptide':'non-Cognate','AY9':'non-Cognate'}
    class_relabel = class_labels[idx]
    class_relabel = np.array([class_dict[x] for x in class_relabel])
    DTCR = DeepTCR_WF('screen', device=gpu)
    DTCR.Load_Data(beta_sequences=beta_sequences[idx],
                   counts=counts[idx],
                   class_labels=class_relabel,
                   sample_labels=sample_labels[idx], p=p)
    DTCR.Monte_Carlo_CrossVal(folds=folds, graph_seed=graph_seed,seeds=seeds,
                              LOO=2, combine_train_valid=True, num_concepts=64,
                              convergence='training',train_loss_min=0.1)
    c = np.where(DTCR.lb.classes_ == group[ii])[0][0]
    sel_idx = (group[ii] == DTCR.class_id)
    seq = DTCR.beta_sequences[sel_idx]
    pred = DTCR.predicted[sel_idx,c]
    sequences.append(seq)
    seq_class_labels.append(DTCR.class_id[sel_idx])
    predicted.append(pred)
    seq_counts.append(DTCR.counts[sel_idx])

with open('tw10_seq.pkl','wb') as f:
    pickle.dump([sequences,seq_class_labels,predicted,seq_counts],f,protocol=4)