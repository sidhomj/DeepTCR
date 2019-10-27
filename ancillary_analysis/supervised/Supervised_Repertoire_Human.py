"""
Figure 3C
"""

"""This script is used to generate the ROC plot from the previously trained HLA, TCR, TCR+HLA
models showing the comparative performance of these models on the CheckMate-038 clinical trial data."""

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#Load Data
df_tcr = pd.read_csv('sample_tcr.csv')
df_hla = pd.read_csv('sample_hla.csv')
df_tcr_hla = pd.read_csv('sample_tcr_hla.csv')

#Concat Data
df = pd.DataFrame()
df['Samples'] = df_tcr['Samples']
df['TCR'] = df_tcr['y_pred']
df['HLA'] = df_hla['y_pred']
df['TCR+HLA'] = df_tcr_hla['y_pred']
df['Label'] = df_tcr_hla['y_test']
models = ['TCR','HLA','TCR+HLA']

#Run Bootstrapping to assess differneces between
n_boots=5000
auc_list = []
model_list = []
for n in range(n_boots):
    idx = np.random.choice(range(len(df)), len(df), replace=True)
    auc_list.append(roc_auc_score(df['Label'].iloc[idx],df['TCR'].iloc[idx]))
    model_list.append('TCR')
    auc_list.append(roc_auc_score(df['Label'].iloc[idx],df['HLA'].iloc[idx]))
    model_list.append('HLA')
    auc_list.append(roc_auc_score(df['Label'].iloc[idx],df['TCR+HLA'].iloc[idx]))
    model_list.append('TCR+HLA')

df_bootstrap = pd.DataFrame()
df_bootstrap['model'] = model_list
df_bootstrap['auc'] = auc_list

#Draw figure
f, ax1 = plt.subplots()
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
for m in models:
    roc_score = roc_auc_score(df['Label'], df[m])
    fpr, tpr, th = roc_curve(df['Label'], df[m])
    class_name = m
    ax1.plot(fpr, tpr, lw=2, label='%s (%0.2f)' % (class_name, roc_score))
ax1.legend(loc='upper left',frameon=False)

ax2 = f.add_axes([0.48, .2, .4, .4])
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
sns.violinplot(data=df_bootstrap,x='model',y='auc',ax=ax2,cut=0)
ax2.set_xlabel('')

#Get P-Values
df_pval = pd.DataFrame()

model_1 = []
model_2 = []
p_val = []

models_test = ['TCR','HLA']
m1 = np.array(df_bootstrap[df_bootstrap['model']==models_test[0]]['auc'])
m2 = np.array(df_bootstrap[df_bootstrap['model']==models_test[1]]['auc'])
model_1.append(models_test[0])
model_2.append(models_test[1])
p_val.append(np.where(np.sort(m2 - m1)>=0)[0][0]/n_boots)

models_test = ['TCR','TCR+HLA']
m1 = np.array(df_bootstrap[df_bootstrap['model']==models_test[0]]['auc'])
m2 = np.array(df_bootstrap[df_bootstrap['model']==models_test[1]]['auc'])
model_1.append(models_test[0])
model_2.append(models_test[1])
p_val.append(np.where(np.sort(m2 - m1)>=0)[0][0]/n_boots)

models_test = ['HLA','TCR+HLA']
m1 = np.array(df_bootstrap[df_bootstrap['model']==models_test[0]]['auc'])
m2 = np.array(df_bootstrap[df_bootstrap['model']==models_test[1]]['auc'])
model_1.append(models_test[0])
model_2.append(models_test[1])
p_val.append(np.where(np.sort(m2 - m1)>=0)[0][0]/n_boots)

df_pval['Model_1'] = model_1
df_pval['Model_2'] = model_2
df_pval['P_Val'] = p_val
