from DeepTCR.DeepTCR import DeepTCR_SS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import seaborn as sns
import pickle
import os
import matplotlib
matplotlib.rc('font', family='Arial')

#Instantiate training object
DTCRU = DeepTCR_SS('Murine_Sup')
#Load Data
# DTCRU.Get_Data(directory='../../Data/Murine_Antigens',Load_Prev_Data=False,
#                aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3,
#                classes=['Db-F2', 'Db-M45', 'Db-NP', 'Db-PA', 'Db-PB1'])
# DTCRU.Monte_Carlo_CrossVal(folds=5)

DTCR_inf = DeepTCR_SS('load')
DTCR_inf.Get_Data(directory='../../Data/Murine_Antigens',Load_Prev_Data=False,
               aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3,
               classes=['Kb-M38', 'Kb-SIY','Kb-TRP2', 'Kb-m139'])

beta_sequences = DTCR_inf.beta_sequences
v_beta = DTCR_inf.v_beta
j_beta = DTCR_inf.j_beta

out = DTCRU.Sequence_Inference(beta_sequences=beta_sequences,v_beta=v_beta,j_beta=j_beta)
out2 = DTCRU.Sequence_Inference(beta_sequences=beta_sequences,v_beta=v_beta,j_beta=j_beta)

