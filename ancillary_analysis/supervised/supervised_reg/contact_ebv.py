import numpy as np
import pandas as pd
from DeepTCR.DeepTCR import DeepTCR_SS
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Arial')
from sklearn.metrics import roc_auc_score, roc_curve

DTCRS = DeepTCR_SS('reg_ebv',device=2)

alpha = 'CAEDNNARLMF'
beta = 'CSARDGTGNGYTF'
contacts_alpha = [0,0,0,1,1,1,1,0,0,0,0]
contacts_beta = [0,0,0,1,1,1,1,1,1,0,0,0,0]

input_alpha = np.array([alpha,alpha])
input_beta = np.array([beta,beta])
fig_rsl,ax_rsl = DTCRS.Residue_Sensitivity_Logo(input_alpha,input_beta,background_color='black',Load_Prev_Data=False)

df_alpha = pd.DataFrame()
df_alpha['seq'] = list(alpha)
df_alpha['mag'] = DTCRS.mag_alpha
df_alpha['label'] = contacts_alpha

df_beta = pd.DataFrame()
df_beta['seq'] = list(beta)
df_beta['mag'] = DTCRS.mag_beta
df_beta['label'] = contacts_beta

df = pd.concat([df_alpha,df_beta])
roc_auc_score(df['label'],df['mag'])

plt.figure()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=24)
plt.ylabel('True Positive Rate',fontsize=24)
y_test = df['label']
y_pred = df['mag']
roc_score = roc_auc_score(y_test,y_pred)
fpr, tpr, th = roc_curve(y_test, y_pred)
id = 'Contact Residue'
plt.plot(fpr, tpr, lw=2, label='%s (%0.3f)' % (id, roc_score),c='grey')
plt.legend(loc="lower right",prop={'size':16})
ax = plt.gca()
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.title('3O4L',fontsize=24)
plt.tight_layout()
