"""
Supplementary Fig. 21
"""

"""This script is used to benchmark various state-of-the-art biomarkers of response to immunotherapy
agasint DeepTCR."""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('CM038_BM.csv')
models = ['Clonality','PDL1','TMB','DeepTCR']

#Run Bootstrapping to assess differneces between biomarkers
n_boots=5000
auc_list = []
model_list = []
for n in range(n_boots):
    for m in models:
        df_test = df.dropna(subset=[m])
        idx = np.random.choice(range(len(df_test)), len(df_test), replace=True)
        auc_list.append(roc_auc_score(df_test['Response_Num'].iloc[idx], df_test[m].iloc[idx]))
        model_list.append(m)

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
    df_test = df.dropna(subset=[m])
    roc_score = roc_auc_score(df_test['Response_Num'], df_test[m])
    fpr, tpr, th = roc_curve(df_test['Response_Num'], df_test[m])
    class_name = m
    ax1.plot(fpr, tpr, lw=2, label='%s (%0.2f)' % (class_name, roc_score))

ax1.legend(loc = 'upper left',prop={'size':10},frameon=False)

ax2 = f.add_axes([0.52, .18, .35, .35])
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
sns.violinplot(data=df_bootstrap,x='model',y='auc',order=models,ax=ax2,cut=0)
ax2.set_xlabel('')
plt.show()

#Bootstrap statistics
model_1 = []
model_2 = []
p_val = []

models_test = ['Clonality','DeepTCR']

df_test = df.dropna(subset=models_test)
auc_diff = []
for n in range(n_boots):
    idx = np.random.choice(range(len(df_test)),len(df_test),replace=True)
    auc_1 = roc_auc_score(df_test['Response_Num'].iloc[idx],df_test[models_test[0]].iloc[idx])
    auc_2 = roc_auc_score(df_test['Response_Num'].iloc[idx],df_test[models_test[1]].iloc[idx])
    auc_diff.append(auc_2-auc_1)

model_1.append(models_test[0])
model_2.append(models_test[1])
p_val.append(np.where(np.sort(auc_diff)>=0)[0][0]/n_boots)

models_test = ['PDL1','DeepTCR']

df_test = df.dropna(subset=models_test)
auc_diff = []
for n in range(n_boots):
    idx = np.random.choice(range(len(df_test)),len(df_test),replace=True)
    auc_1 = roc_auc_score(df_test['Response_Num'].iloc[idx],df_test[models_test[0]].iloc[idx])
    auc_2 = roc_auc_score(df_test['Response_Num'].iloc[idx],df_test[models_test[1]].iloc[idx])
    auc_diff.append(auc_2-auc_1)

model_1.append(models_test[0])
model_2.append(models_test[1])
p_val.append(np.where(np.sort(auc_diff)>=0)[0][0]/n_boots)

models_test = ['TMB','DeepTCR']

df_test = df.dropna(subset=models_test)
auc_diff = []
for n in range(n_boots):
    idx = np.random.choice(range(len(df_test)),len(df_test),replace=True)
    auc_1 = roc_auc_score(df_test['Response_Num'].iloc[idx],df_test[models_test[0]].iloc[idx])
    auc_2 = roc_auc_score(df_test['Response_Num'].iloc[idx],df_test[models_test[1]].iloc[idx])
    auc_diff.append(auc_2-auc_1)

model_1.append(models_test[0])
model_2.append(models_test[1])
p_val.append(np.where(np.sort(auc_diff)>=0)[0][0]/n_boots)

df_pval = pd.DataFrame()
df_pval['Model_1'] = model_1
df_pval['Model_2'] = model_2
df_pval['P_Val'] = p_val




