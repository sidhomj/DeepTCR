import numpy as np
import pandas as pd
import os
import utils
from DeepTCR.DeepSynapse import DeepSynapse
from sklearn.metrics import roc_auc_score

df = utils.load_foreign_v_self()

DTCR = DeepSynapse('foreign_v_self')
DTCR.Load_Data(epitope_sequences = np.array(df['AA Sequence']),
                class_labels=np.array(df['Y']),
               )

DTCR.Monte_Carlo_CrossVal(folds=1,batch_size=50000,epochs_min=10,
                          num_fc_layers=3,units_fc=256,
                          units_hla=[12,12,12],kernel_hla=[30,30,30],stride_hla=[5,5,5])

DTCR.SRCC()
threshold = 1 - np.log10(500) / np.log10(50000)
DTCR.AUC_Curve_CV(threshold=threshold)
