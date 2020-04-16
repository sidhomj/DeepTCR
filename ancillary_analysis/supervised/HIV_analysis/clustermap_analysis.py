import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


file = 'aucs_gagtw10.pkl'
file = 'aucs_gagiw9.pkl'
file = 'aucs_gagkk15.pkl'
file = 'aucs_nefkf9.pkl'
file = 'aucs_nefhw9.pkl'
file = 'aucs_nefyt9.pkl'

with open(file,'rb') as f:
    aucs,group = pickle.load(f)

aucs = aucs.T + aucs
aucs[aucs<0.5] = 0.5
aucs = 1 - aucs
aucs = 2*aucs

df = pd.DataFrame(aucs)
df.index = group
df.columns = group
# df.drop(columns='NoPeptide',inplace=True)
# df.drop(labels='NoPeptide',inplace=True)
sns.clustermap(data=df,cmap='bwr',row_cluster=True,col_cluster=True)
plt.subplots_adjust(bottom=0.2,right=0.8)