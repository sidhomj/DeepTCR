import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

#Assess correlation between models on per sample basis
df_TCR = pd.read_csv('sample_tcr.csv')
df_HLA = pd.read_csv('sample_hla.csv')
df_TCR_HLA = pd.read_csv('sample_tcr_hla.csv')

df_plot = pd.DataFrame()
df_plot['TCR'] = df_TCR['y_pred']
df_plot['HLA'] = df_HLA['y_pred']
df_plot['TCR+HLA'] = df_TCR_HLA['y_pred']
df_plot['y_test'] = df_TCR['y_test']
label_dict = {1.0:'crpr',0.0:'sdpd'}
df_plot['Label'] = df_plot['y_test'].map(label_dict)
df_plot['Sample'] = df_TCR['Samples']

#average over sample
df_plot = df_plot.groupby(['Sample']).agg({'TCR':'mean','HLA':'mean','TCR+HLA':'mean','y_test':'mean','Label':'first'})

#HLA vs TCR
sns.scatterplot(data=df_plot,x='TCR',y='HLA',linewidth=0,hue='Label')
plt.xlim([0,1])
plt.ylim([0,1])
spearmanr(df_plot['TCR'],df_plot['HLA'])

#HLA vs TCR+HLA
sns.scatterplot(data=df_plot,x='HLA',y='TCR+HLA',linewidth=0,hue='Label')
plt.xlim([0,1])
plt.ylim([0,1])
spearmanr(df_plot['HLA'],df_plot['TCR+HLA'])

#TCR vs TCR_HLA
sns.scatterplot(data=df_plot,x='TCR',y='TCR+HLA',linewidth=0,hue='Label')
plt.xlim([0,1])
plt.ylim([0,1])
spearmanr(df_plot['TCR'],df_plot['TCR+HLA'])