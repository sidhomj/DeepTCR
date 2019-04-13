from DeepTCR.DeepTCR import DeepTCR_SS
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import os

p = Pool(80)
dir_results = 'alpha_v_beta_results'
if not os.path.exists(dir_results):
    os.makedirs(dir_results)

DTCR = DeepTCR_SS('alpha_v_beta_SS')

antigens = ['GANAB-S5F',
 'ATP6AP1-KLG_G3W',
 'CMV-MLN',
 'GNL3L-R4C',
 'MART1-A2L',
 'YFV-LLW']

opt = ['alpha','beta','alpha_beta']

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
        elif o == 'alaph_beta':
            DTCR = DeepTCR_SS('alpha_v_beta_SS')
            DTCR.Get_Data(directory='../../Data/Zhang/'+a,aa_column_alpha=0,aa_column_beta=1,p=p)

        DTCR.Monte_Carlo_CrossVal(folds=50)
        y_pred_list.append(DTCR.y_pred)
        y_test_list.append(DTCR.y_test)

    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    for ii, o in enumerate(opt, 0):
        y_test = y_test_list[ii]
        y_pred = y_pred_list[ii]
        roc_score = roc_auc_score(y_test[:, 1], y_pred[:, 1])
        fpr, tpr, _ = roc_curve(y_test[:, 1], y_pred[:, 1])
        plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (o, roc_score))

    plt.legend(loc="lower right",fontsize=14)
    plt.title(a,fontsize=22)
    plt.savefig(os.path.join(dir_results,a+ '_AUC.tif'))
    plt.close()

check=1


