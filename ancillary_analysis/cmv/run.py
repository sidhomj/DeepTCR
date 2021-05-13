import pandas as pd
import numpy as np
from DeepTCR.DeepTCR import DeepTCR_WF, DeepTCR_SS
import matplotlib.pyplot as plt
import glob
from copy import deepcopy
import seaborn as sns

import os
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

DTCR_l = DeepTCR_WF('load')
DTCR_l.Get_Data('../../NatGen_Data/natgen-cohort1',Load_Prev_Data=True,type_of_data_cut='Num_Seq',data_cut=10000,
              aa_column_beta=1,count_column=5,v_beta_column=10,d_beta_column=13,j_beta_column=16)

beta_sequences = DTCR_l.beta_sequences
sample_labels = DTCR_l.sample_id
counts = DTCR_l.counts

idx = np.isin(sample_labels,np.random.choice(np.unique(sample_labels),100,replace=False))
beta_sequences = beta_sequences[idx]
sample_labels = sample_labels[idx]
counts = counts[idx]

df_meta = pd.read_csv('../../NatGen_Data/cohort1_cmv.csv')
df_meta = df_meta[df_meta['CMV']!='Unknown']
df_meta['Subject'] = df_meta['Subject']  + '.tsv'
label_dict = dict(zip(df_meta['Subject'],df_meta['CMV']))

idx = np.isin(sample_labels,df_meta['Subject'])
beta_sequences = beta_sequences[idx]
sample_labels = sample_labels[idx]
counts = counts[idx]
class_labels = np.array(list(map(label_dict.get,sample_labels)))

DTCR = DeepTCR_WF('model',device=3)
DTCR.Load_Data(beta_sequences=beta_sequences,counts=counts,class_labels=class_labels,sample_labels=sample_labels)
DTCR.Monte_Carlo_CrossVal(folds=10,train_loss_min=0.3,trainable_embedding=False,l2_reg=3e-3,size_of_net='large',
                          subsample=1000,subsample_valid_test=True,subsample_by_freq=True)
DTCR.AUC_Curve()