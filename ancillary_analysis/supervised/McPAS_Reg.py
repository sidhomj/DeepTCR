import pandas as pd
from DeepTCR.DeepTCR import DeepTCR_SS
import numpy as np
from multiprocessing import Pool
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

df = pd.read_csv('../../Data/10x_Data/Data_Regression.csv')

antigens = np.array(df.columns)[2:]
epitopes = []
for a in antigens:
    epitopes.append(a.split('_')[1])

df_tcr = pd.read_csv('../../Data/McPAS-TCR.csv')
mcpas_counts = []
for e in epitopes:
    temp = df_tcr[df_tcr['Epitope.peptide'] == e]
    temp = temp.groupby(['CDR3.beta.aa']).agg({'Epitope.peptide': 'first'}).reset_index()
    c = np.sum(np.isin(np.asarray(df['beta'].tolist()), temp['CDR3.beta.aa']))
    mcpas_counts.append(c)

df_epitope_counts = pd.DataFrame()
df_epitope_counts['antigen'] = antigens
df_epitope_counts['epitope'] = epitopes
df_epitope_counts['counts'] = mcpas_counts
df_epitope_counts.sort_values(by='counts',inplace=True,ascending=False)

DTCRS = DeepTCR_SS('reg_bm',device=2)

z=0
antigen = df_epitope_counts['antigen'].iloc[z]
epitope = df_epitope_counts['epitope'].iloc[z]

#Get alpha/beta sequences
alpha = np.asarray(df['alpha'].tolist())
beta = np.asarray(df['beta'].tolist())
i = np.where(df.columns==antigen)[0][0]
sel = df.iloc[:, i]
Y = np.log2(np.asarray(sel.tolist()) + 1)
DTCRS.Load_Data(alpha_sequences=alpha, beta_sequences=beta, Y=Y)
folds = 5
seeds = np.array(range(folds))
graph_seed = 0
DTCRS.K_Fold_CrossVal(split_by_sample=False, folds=folds,seeds=seeds,graph_seed=graph_seed)

dir  = 'density_plots'
if not os.path.exists(dir):
    os.makedirs(dir)

x = np.squeeze(DTCRS.predicted)
y = np.squeeze(DTCRS.Y)
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
r = np.argsort(z)
x, y, z = x[r], y[r], z[r]
plt.figure()
plt.scatter(x, y, s=15, c=z, cmap=plt.cm.jet)
plt.title(antigen, fontsize=12)
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Log2(counts+1)', fontsize=12)
plt.savefig(os.path.join(dir, antigen + '.eps'))

temp = df_tcr[df_tcr['Epitope.peptide']==epitope]
temp = temp.groupby(['CDR3.beta.aa']).agg({'Epitope.peptide':'first'}).reset_index()
label = np.isin(DTCRS.beta_sequences,temp['CDR3.beta.aa'])

df_plot = pd.DataFrame()
df_plot['beta'] = DTCRS.beta_sequences
df_plot['counts'] = Y
df_plot['preds'] = DTCRS.predicted
df_plot['GT'] = label
df_plot = df_plot[df_plot['GT'] == 1]

x = np.array(df_plot['preds'])
y = np.array(df_plot['counts'])
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
r = np.argsort(z)
x, y, z = x[r], y[r], z[r]
plt.figure()
plt.scatter(x, y, s=15, c=z, cmap=plt.cm.jet)
plt.title(antigen+' Validated', fontsize=12)
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Log2(counts+1)', fontsize=12)
plt.savefig(os.path.join(dir, antigen + '_validated.eps'))


