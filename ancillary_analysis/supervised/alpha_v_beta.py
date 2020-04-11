"""Figure 2D"""

"""This script is used to benchmark the performance of the Supervised Sequence Classifier
with either the alpha chain, beta chain, or both provided to the model."""

from DeepTCR.DeepTCR import DeepTCR_SS
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import os
import numpy as np

p = Pool(80)
dir_results = 'alpha_v_beta_results'
if not os.path.exists(dir_results):
    os.makedirs(dir_results)

DTCR = DeepTCR_SS('alpha_v_beta_SS')

antigens = ['ATP6AP1-KLG_G3W',
 'GNL3L-R4C',
 'MART1-A2L',
 'YFV-LLW']

opt = ['alpha','beta','alpha_beta']

folds = 100
graph_seed = 0
seeds = np.array(range(folds))
for a in antigens:
    y_pred_list = []
    y_test_list = []
    for o in opt:
        if o == 'alpha':
            DTCR = DeepTCR_SS('alpha_v_beta_SS')
            DTCR.Get_Data(directory='../../Data/Zhang/'+a,aa_column_alpha=0,p=p)
        elif o == 'beta':
            DTCR = DeepTCR_SS('alpha_v_beta_SS')
            DTCR.Get_Data(directory='../../Data/Zhang/'+a,aa_column_beta=1,p=p)
        elif o == 'alpha_beta':
            DTCR = DeepTCR_SS('alpha_v_beta_SS')
            DTCR.Get_Data(directory='../../Data/Zhang/'+a,aa_column_alpha=0,aa_column_beta=1,p=p)

        DTCR.Monte_Carlo_CrossVal(folds=folds,weight_by_class=True,graph_seed=graph_seed,seeds=seeds)
        y_pred_list.append(DTCR.y_pred)
        y_test_list.append(DTCR.y_test)

    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=24)
    plt.ylabel('True Positive Rate',fontsize=24)
    for ii, o in enumerate(opt, 0):
        y_test = y_test_list[ii]
        y_pred = y_pred_list[ii]
        roc_score = roc_auc_score(y_test[:, 1], y_pred[:, 1])
        fpr, tpr, _ = roc_curve(y_test[:, 1], y_pred[:, 1])
        plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (o, roc_score))

    plt.legend(loc="lower right",fontsize=14)
    plt.title(a,fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_results,a+ '_AUC.eps'))
    plt.close()

