import pandas as pd
import numpy as np
from DeepTCR.DeepTCR import DeepTCR_WF

meta = pd.read_csv('../../Data/natgen/cohort1_meta.csv')
meta[meta.columns[0]] = meta[meta.columns[0]]+'.tsv'
meta[meta.columns[4]] = 'CMV'+meta[meta.columns[4]]
meta = meta[meta[meta.columns[4]]!='CMVUnknown']
label_dict = dict(zip(meta[meta.columns[0]],meta[meta.columns[4]]))

DTCR_l = DeepTCR_WF('load')
DTCR_l.Get_Data(directory='../../Data/natgen/data/',Load_Prev_Data=True,
                aa_column_beta=1,v_beta_column=7,d_beta_column=14,j_beta_column=21,
                type_of_data_cut='Num_Seq',
                data_cut=500)

idx = np.isin(DTCR_l.sample_id,meta[meta.columns[0]])
beta_sequences = DTCR_l.beta_sequences[idx]
v_beta = DTCR_l.v_beta[idx]
d_beta = DTCR_l.d_beta[idx]
j_beta = DTCR_l.j_beta[idx]
counts = DTCR_l.counts[idx]
sample_labels = DTCR_l.sample_id[idx]
class_labels = np.array(list(map(label_dict.get,sample_labels)))

DTCR = DeepTCR_WF('ng')
DTCR.Load_Data(beta_sequences=beta_sequences,v_beta=v_beta,d_beta=d_beta,j_beta=j_beta,counts=counts,
               sample_labels=sample_labels,class_labels=class_labels)
folds=1
graph_seed=0
seeds= np.array(range(folds))
DTCR.Monte_Carlo_CrossVal(folds=folds,seeds=seeds,graph_seed=graph_seed,l2_reg=0.0,
                          test_size=0.20,combine_train_valid=True,train_loss_min=0.3)
DTCR.AUC_Curve()