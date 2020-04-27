"""
Supplementary Fig. 22
"""

"""This script is used to benchmark various state-of-the-art biomarkers of response to immunotherapy
agasint DeepTCR."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, ranksums
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit


def resampling_distribution(fn, d, method='bootstrap', n=1000, prop=0.5, rand_seeds=None):
    n_d = len(d)
    one_hot = np.eye(len(np.unique(d[:, 0])))[d[:, 0].astype(int)]

    estimates = list()

    if method == 'bootstrap':
        for i in range(n):
            if rand_seeds is not None:
                np.random.seed(rand_seeds[i])
            y_check = True
            while y_check:
                idx_rand = np.random.choice(n_d, n_d, replace=True)
                if np.all(np.sum(np.eye(2)[d[idx_rand, 0].astype(int)], axis=0) > 1):
                    y_check = False
            estimates.append(fn(d, idx_rand))
    elif method == 'subsampling':
        for i in range(n):
            if rand_seeds is not None:
                np.random.seed(rand_seeds[i])
            idx_rand = np.random.choice(n_d, np.floor(n_d * prop).astype(int), replace=False)
            estimates.append(fn(d, idx_rand))
    elif method == 'mc_pred':
        for i in range(n):
            if rand_seeds is not None:
                np.random.seed(rand_seeds[i])
            idx_rand = np.repeat(False, n_d)
            idx_rand[np.concatenate([np.random.choice(np.where(one_hot[:, i])[0], np.floor(np.sum(one_hot[:, i]) * prop).astype(int), replace=False) for i in range(one_hot.shape[1])])] = True
            estimates.append(fn(d, idx_rand))

    return estimates


d = pd.read_csv('CM038_BM.csv')
d['log2_TTC'] = np.log2(d['TCR_Reads'])
d['log2_TMB'] = np.log2(d['TMB'])
d['Response'] = d['Response_cat']
auc_fn = lambda samp, idx: (lambda roc: [auc(roc[0], roc[1]), roc])(roc_curve(samp[idx, 0], samp[idx, 1]))
# seeds = np.random.choice(1000, 1000, replace=False)
# lr = LogisticRegression(solver='lbfgs', max_iter=1000)

mods = dict()
for mod_name, col in zip(['PD-L1 (TPS)', 'TMB', 'TCR Clonality', 'TTC', 'DeepTCR'], ['PDL1', 'log2_TMB', 'Clonality', 'log2_TTC', 'DeepTCR']):
    mods[mod_name] = dict()
    mods[mod_name]['cols'] = ['Response_Num', col]
    sample_idx = ~d[mods[mod_name]['cols']].isna().any(axis=1)
    # mods[mod_name]['ds'] = resampling_distribution(lambda a, b: lr_fn(a, b, lr=lr, pred='test'), d[mods[mod_name]['cols']].loc[sample_idx].values, method='mc_pred', rand_seeds=seeds, prop=0.75)
    mods[mod_name]['ds'] = resampling_distribution(auc_fn, d[mods[mod_name]['cols']].loc[sample_idx].values, method='bootstrap', rand_seeds=None)
    mods[mod_name]['aucs'] = np.array([s[0] for s in mods[mod_name]['ds']])
    mods[mod_name]['med_aucs'] = np.median(mods[mod_name]['aucs'])
    mods[mod_name]['m_fpr'] = np.sort(np.unique(np.concatenate([s[1][0] for s in mods[mod_name]['ds']], axis=0)))
    mods[mod_name]['m_tpr'] = np.median(np.stack([np.interp(mods[mod_name]['m_fpr'], s[1][0], s[1][1]) for s in mods[mod_name]['ds']], axis=1), axis=1)
    mods[mod_name]['m_tpr'][[0, -1]] = [0, 1]

_, ax = plt.subplots(nrows=2, ncols=2,figsize=(10,8))

ax[0, 1].cla()
sns.violinplot(data=[mods[key]['aucs'] for key in mods.keys()], cut=0, orient='h', ax=ax[0, 1])
ax[0, 1].set(yticklabels=mods.keys(), xlabel='Area Under Curve (AUC)', title='Bootstrap AUC Distributions')

ax[0, 0].cla()
for mod_name in mods:
    ax[0, 0].plot(mods[mod_name]['m_fpr'], mods[mod_name]['m_tpr'])
ax[0, 0].legend(['%s (%.3f)' % (key, mods[key]['med_aucs']) for key in mods.keys()])
ax[0, 0].set(xlabel='FPR', ylabel='TPR', title='Model AUCs')

ax[1, 0].cla()
sns.scatterplot(d['log2_TTC'].values, d['DeepTCR'].values, hue=d['Response'], edgecolor=None, ax=ax[1, 0])
ax[1, 0].set(xlabel='log2 TCR Read counts', ylabel='DeepTCR Likelihood of Response', title='DeepTCR vs. TTC')
spearmanr(d['TCR_Reads'].values, d['DeepTCR'].values)

lr = LogisticRegression(solver='lbfgs', max_iter=1000)

coef = list()
# mc_pred = list()
# for idx_train, idx_test in StratifiedShuffleSplit(500, 10).split(np.zeros(d.shape[0]), d['Response_Num'].values):
for idx_train in np.random.choice(d.shape[0], [500, d.shape[0]], replace=True):
    # lr.fit(np.concatenate([d.loc[idx_train, ['DeepTCR', 'log2_TTC']].values, np.prod(d.loc[idx_train, ['DeepTCR', 'log2_TTC']].values, axis=1)[:, np.newaxis]], axis=1), d.loc[idx_train, 'Response_Num'].values)
    # lr.fit(d.loc[idx_train, ['DeepTCR', 'TCR_Reads']].values, d.loc[idx_train, 'Response_Num'].values)
    lr.fit(d.loc[idx_train, ['DeepTCR', 'log2_TTC']].values, d.loc[idx_train, 'Response_Num'].values)
    coef.append(lr.coef_[0])
    # mc_pred.append(np.stack([lr.predict_proba(d.loc[idx_test, ['DeepTCR', 'log2_TTC']].values)[:, 1], d.loc[idx_test, 'Response_Num'].values], axis=1))
# mc_pred = np.concatenate(mc_pred, axis=0)
coef = np.stack(coef, axis=0)

# fpr, tpr, _ = roc_curve(mc_pred[:, 1], mc_pred[:, 0])
# auc(fpr, tpr)

sns.violinplot(data=coef, ax=ax[1, 1])
ax[1, 1].set(ylabel='Logistic regression coefficient', xticklabels=['DeepTCR', 'TTC'], ylim=[0, ax[1, 1].get_ylim()[1]], title='Multivariate Logistic Regression Coefficients')
plt.subplots_adjust(top = 0.95, bottom=0.1, hspace=0.25,wspace=0.35)
plt.show()
np.quantile(coef, [0.025, 0.5, 0.975], axis=0)
np.mean(coef > 0, axis=0)