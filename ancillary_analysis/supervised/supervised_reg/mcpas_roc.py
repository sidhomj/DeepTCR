import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib
matplotlib.rc('font', family='Arial')

flu = pd.read_csv('flu_mcpas_val.csv')
ebv = pd.read_csv('ebv_mcpas_val.csv')
mart1 = pd.read_csv('mart1_mcpas_val.csv')

names = ['Flu-MP','BMLF1_EBV','MART-1']
antigens = [flu,ebv,mart1]

plt.figure(figsize=(6,5))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
for n,a in zip(names,antigens):
    fpr, tpr, _ = roc_curve(a['label'], a['pred'])
    roc_score = roc_auc_score(a['label'], a['pred'])
    label = '%s = %0.3f (n=%d)'  % (n,roc_score,np.sum(a['label']))
    plt.plot(fpr, tpr, lw=2, label=label)
plt.legend(loc='lower right', frameon=False,prop={'size': 14})
ax = plt.gca()
ax.xaxis.label.set_size(24)
ax.yaxis.label.set_size(24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.savefig('mcpas_roc.eps')

check=1