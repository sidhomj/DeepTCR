"""Figure 3C
Supplementary Figure 19
"""

"""This script is used to generate the plot from the previously trained HLA, TCR, TCR+HLA
models showing the comparative performance of these models."""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc
from scipy.stats import wilcoxon, rankdata, ranksums

# AUC plot axes
f, ax1 = plt.subplots()

# data files to process and their label
model_info = {'TCR': dict(file='sample_tcr_2.csv', alpha=1.),
              'HLA': dict(file='sample_hla_2.csv', alpha=1.),
              'TCR+HLA': dict(file='sample_tcr_hla_2.csv', alpha=1.0)}

#
# mc pred approach

# number of boot straps
n_boot = 5000
for model_name in model_info:
    model_info[model_name]['mc_preds'] = pd.read_csv(model_info[model_name]['file'])
    model_info[model_name]['mean'] = model_info[model_name]['mc_preds'].groupby('Samples')[['y_test', 'y_pred']].agg(np.mean)
    fpr, tpr, _ = roc_curve(model_info[model_name]['mc_preds'].y_test.values, model_info[model_name]['mc_preds'].y_pred.values)
    model_info[model_name]['full'] = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}
    model_info[model_name]['bootstrap'] = {'auc': list()}

for model_name in model_info:
    ax1.plot(model_info[model_name]['full']['fpr'], model_info[model_name]['full']['tpr'], alpha=model_info[model_name]['alpha'], label='%s (%.2f)' % (model_name,  model_info[model_name]['full']['auc']))
ax1.legend()

for i in range(n_boot):
    idx = np.random.choice(model_info['TCR']['mc_preds'].shape[0], model_info['TCR']['mc_preds'].shape[0],replace=True)

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

# 95 Bootstrap CI of test statistic - AUC, null hyopthesis is that diff = 0
np.quantile(model_info['TCR+HLA']['bootstrap']['auc'] - model_info['HLA']['bootstrap']['auc'], [0.025, 0.975])
np.quantile(model_info['TCR+HLA']['bootstrap']['auc'] - model_info['TCR']['bootstrap']['auc'], [0.025, 0.975])
np.quantile(model_info['HLA']['bootstrap']['auc'] - model_info['TCR']['bootstrap']['auc'], [0.025, 0.975])

# checking lower bounds of 99.9, 99, and 95 CIs to see where they cross 0
np.quantile(model_info['TCR+HLA']['bootstrap']['auc'] - model_info['HLA']['bootstrap']['auc'], [0.0005, 0.005, 0.025])
np.quantile(model_info['TCR+HLA']['bootstrap']['auc'] - model_info['TCR']['bootstrap']['auc'], [0.0005, 0.005, 0.025])

# mc_preds per sample distribution plot sorted by sample mean in combo model
_, ax = plt.subplots(ncols=3)
i = 0
idx = model_info[model_name]['mean']['y_pred'].sort_values().index
for model_name in model_info:
    sns.violinplot(x=model_info[model_name]['mc_preds']['y_pred'].values, y=model_info[model_name]['mc_preds']['Samples'].values, linewidth=.5, order=idx, ax=ax[i], cut=0)
    ax[i].set(yticklabels=[], xlim=[0, 1], title=model_name)
    i += 1
label_dict = {0: 'sdpd', 1: 'crpr'}
ax[0].set(yticklabels=[label_dict[_] for _ in model_info[model_name]['mean'].loc[idx, ]['y_test'].values])
ax[1].set(yticks=[])
ax[2].set(yticks=[])
plt.tight_layout()

for model_name in model_info:
    model_info[model_name]['mean']['90 CI'] = model_info[model_name]['mc_preds'].groupby('Samples')['y_pred'].apply(lambda x: np.diff(np.quantile(x, [.1, .9]))[0])

# paired tests
wilcoxon(model_info['TCR+HLA']['mean']['90 CI'], model_info['HLA']['mean']['90 CI'])
wilcoxon(model_info['TCR+HLA']['mean']['90 CI'], model_info['TCR']['mean']['90 CI'])
wilcoxon(model_info['TCR']['mean']['90 CI'], model_info['HLA']['mean']['90 CI'])

# non paired
ranksums(model_info['TCR+HLA']['mean']['90 CI'], model_info['HLA']['mean']['90 CI'])
ranksums(model_info['TCR+HLA']['mean']['90 CI'], model_info['TCR']['mean']['90 CI'])
ranksums(model_info['TCR']['mean']['90 CI'], model_info['HLA']['mean']['90 CI'])
