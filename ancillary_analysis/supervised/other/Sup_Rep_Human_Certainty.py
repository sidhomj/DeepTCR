"""
Supplementary Figure 20
"""

"""This script is used to generate the plot from the previously trained HLA, TCR, TCR+HLA
models showing the certainty of the predictions for each sample for these models on the 
CheckMate-038 clinical trial data. """

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon, rankdata, ranksums

# data files to process and their label
model_info = {'TCR': dict(file='sample_tcr.csv', alpha=1.),
              'HLA': dict(file='sample_hla.csv', alpha=1.),
              'TCR+HLA': dict(file='sample_tcr_hla.csv', alpha=1.0)}

# mc pred approach
for model_name in model_info:
    model_info[model_name]['mc_preds'] = pd.read_csv(model_info[model_name]['file'])
    model_info[model_name]['mean'] = model_info[model_name]['mc_preds'].groupby('Samples')[['y_test', 'y_pred']].agg(np.mean)

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