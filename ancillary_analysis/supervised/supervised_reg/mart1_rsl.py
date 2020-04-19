import numpy as np
import pandas as pd
from DeepTCR.DeepTCR import DeepTCR_SS
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Arial')

DTCRS = DeepTCR_SS('reg_mart1',device=2)

alpha = 'CAVNFGGGKLIF'
beta = 'CASSWSFGTEAFF'
input_alpha = np.array([alpha,alpha])
input_beta = np.array([beta,beta])
pred = DTCRS.Sequence_Inference(input_alpha,input_beta)
fig_rsl,ax_rsl = DTCRS.Residue_Sensitivity_Logo(input_alpha,input_beta,background_color='black',Load_Prev_Data=False)

fig_rsl.savefig('mart1_rsl.png',dpi=1200,facecolor='black')

fig,ax = plt.subplots(1,2,figsize=(10,5))
sns.swarmplot(data=DTCRS.df_alpha_list[0],x='pos',y='high',ax=ax[0])
i = 0
ax[i].set_xlabel('')
ax[i].set_ylabel('')
ax[i].set_xticklabels(list(alpha),size=24)
ax[i].tick_params(axis='y',labelsize=18)
ax[i].spines['right'].set_visible(False)
ax[i].spines['top'].set_visible(False)
ax[i].spines['bottom'].set_visible(False)
ax[i].tick_params(axis='x',length=0)
ylim_alpha = ax[i].get_ylim()

sns.swarmplot(data=DTCRS.df_beta_list[0],x='pos',y='high',ax=ax[1])
i = 1
ax[i].set_xticklabels(list(beta),size=24)
ax[i].tick_params(axis='y',labelsize=18)
ax[i].set_xlabel('')
ax[i].set_ylabel('')
ax[i].spines['right'].set_visible(False)
ax[i].spines['top'].set_visible(False)
ax[i].spines['bottom'].set_visible(False)
ax[i].tick_params(axis='x',length=0)
ylim_beta = ax[i].get_ylim()
ylim = np.vstack([ylim_alpha,ylim_beta])
ylim_min = np.min(ylim)
ylim_max = np.max(ylim)
ax[0].set_ylim([ylim_min,ylim_max])
ax[1].set_ylim([ylim_min,ylim_max])
ax[0].axhline(pred[0],color='black')
ax[1].axhline(pred[0],color='black')
fig.savefig('mart1_rsl_dist.eps')