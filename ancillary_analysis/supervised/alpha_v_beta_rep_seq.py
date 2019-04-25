from DeepTCR.DeepTCR import DeepTCR_SS
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import os

p = Pool(80)
dir_results = 'alpha_v_beta_results'
if not os.path.exists(dir_results):
    os.makedirs(dir_results)

antigens = ['GANAB-S5F',
 'ATP6AP1-KLG_G3W',
 'CMV-MLN',
 'GNL3L-R4C',
 'MART1-A2L',
 'YFV-LLW']

for a in antigens:
    DTCR = DeepTCR_SS(a+'Rep')
    DTCR.Get_Data(directory='../../Data/Zhang/'+a,aa_column_alpha=0,aa_column_beta=1,p=p)
    DTCR.Monte_Carlo_CrossVal(folds=50,weight_by_class=True)
    DTCR.Representative_Sequences()