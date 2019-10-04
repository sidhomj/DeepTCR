"""Figure 4C"""

"""This script is used to generate the plot from the previously trained HLA, TCR, TCR+HLA
models showing the comparative performance of these models."""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc
from scipy.stats import rankdata
from scipy.stats import wilcoxon

# AUC plot axes
f, ax1 = plt.subplots()

# data files to process and their label
model_info = {'TCR': dict(file='sample_tcr.csv', alpha=1.),
              'HLA': dict(file='sample_hla.csv', alpha=1.),
              'TCR+HLA': dict(file='sample_tcr_hla.csv', alpha=1.0)}

#
# mc pred approach
#

# number of boot straps
n_boot = 5000
# number of samples per boot
s_boot = 300

s = list()
for model_name in model_info:
    model_info[model_name]['mc_preds'] = pd.read_csv(model_info[model_name]['file'])
    # d['mc_set'] = np.repeat(np.arange(100), 6)
    # d['ranks'] = d.groupby('mc_set')['y_pred'].transform(lambda x: rankdata(x))
    s.append(model_info[model_name]['mc_preds'].groupby('Samples').apply(lambda x: pd.Series([np.mean(x.y_test), np.mean(x.y_pred)], index=['y_test', model_name])))
    fpr, tpr, _ = roc_curve(model_info[model_name]['mc_preds'].y_test.values, model_info[model_name]['mc_preds'].y_pred.values)
    model_info[model_name]['full'] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}
    model_info[model_name]['bootstrap'] = {'auc': list()}
s = pd.concat([s[0].iloc[:, 0]] + [_.iloc[:, 1] for _ in s], axis=1)

for model_name in model_info:
    ax1.plot(model_info[model_name]['full']['fpr'], model_info[model_name]['full']['tpr'], alpha=model_info[model_name]['alpha'], label='%s (%.2f)' % (model_name,  model_info[model_name]['full']['auc']))
ax1.legend()

for i in range(n_boot):
    idx = np.random.choice(model_info['TCR']['mc_preds'].shape[0], s_boot)
    for model_name in model_info:
        fpr, tpr, _ = roc_curve(model_info[model_name]['mc_preds']['y_test'].loc[idx].values, model_info[model_name]['mc_preds']['y_pred'].loc[idx].values)
        model_info[model_name]['bootstrap']['auc'].append(auc(fpr, tpr))

for model_name in model_info:
    model_info[model_name]['bootstrap']['auc'] = np.array(model_info[model_name]['bootstrap']['auc'])

ax2 = f.add_axes([0.48, .2, .4, .4])
sns.violinplot(data=[model_info[model_name]['bootstrap']['auc'] for model_name in model_info], orient='v', cut=0, ax=ax2)
ax2.set(xticklabels=model_info.keys(), frame_on=True)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')

# empirical p-value, probability of null hypothesis
np.mean((model_info['TCR+HLA']['bootstrap']['auc'] - model_info['HLA']['bootstrap']['auc']) <= 0)
np.mean((model_info['TCR+HLA']['bootstrap']['auc'] - model_info['TCR']['bootstrap']['auc']) <= 0)
np.mean((model_info['TCR']['bootstrap']['auc'] - model_info['HLA']['bootstrap']['auc']) <= 0)