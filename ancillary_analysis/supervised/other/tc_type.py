"""
Supplementary Figure 19
"""

"""This script is used to generate the plots that look at performance metrics of the DeepTCR models
 stratified by treatment arm in the CheckMate-038 trial data."""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd

def create_figure(df_model,model_name):
    df_model['sample_2'] = df_model['Samples'].str[:-9]
    df_combo = df_model[df_model['sample_2'].isin(combo)]
    df_mono = df_model[df_model['sample_2'].isin(mono)]

    models = [df_mono, df_combo]
    names = ['nivo', 'nivo+ipi']

    plt.figure()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)

    for m, n in zip(models, names):
        roc_score = roc_auc_score(m['y_test'], m['y_pred'])
        fpr, tpr, _ = roc_curve(m['y_test'], m['y_pred'])
        plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (n, roc_score))
    plt.legend(loc="lower right")
    plt.title(model_name)
    plt.show()

df = pd.read_csv('CM038_BM.csv')
combo = np.array(df['sample'][df['Tx'].str.contains('IPI')])
mono = np.array(df['sample'][~df['Tx'].str.contains('IPI')])

models = ['sample_tcr.csv','sample_hla.csv','sample_tcr_hla.csv']
model_names = ['TCR','HLA','TCR+HLA']

for m,n in zip(models,model_names):
    df_model = pd.read_csv(m)
    create_figure(df_model,n)


