import numpy as np
import pandas as pd
import os
import utils
from DeepTCR.DeepSynapse import DeepSynapse
from sklearn.metrics import roc_auc_score

df = utils.load_mhc_binding()
df = df[df['AA Sequence'].str.len() <= 12]
df['Y'][df['Y']>1]=1
df['Y'][df['Y']<0]=0
df = df.sample(1000)

DTCR = DeepSynapse('mhc_binding')
DTCR.Load_Data(epitope_sequences = np.array(df['AA Sequence']),
               hla=np.array(df['Allele Name']),
                Y=np.array(df['Y']),
               use_hla_supertype=False,
               use_hla_seq=False
               )

DTCR.Monte_Carlo_CrossVal(folds=1,batch_size=50000,epochs_min=10,
                          units_fc=[1024,1024,1024],
                          embedding_dim_hla=512,
                          units_epitope=[1024],kernel_epitope=[10],stride_epitope=[1],
                          units_hla=[12,12,12],kernel_hla=[30,30,30],stride_hla=[5,5,5])

# DTCR.SRCC()
threshold = 1 - np.log10(500) / np.log10(50000)
DTCR.AUC_Curve_CV(threshold=threshold)

y_test = DTCR.y_test >= threshold
roc_auc_score(y_test,DTCR.y_pred)