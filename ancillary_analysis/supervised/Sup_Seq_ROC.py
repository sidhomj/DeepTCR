"""Figure 2B"""

"""This script is used to create the ROC curves for assessing the ability
of supervised sequence classifier to correctly predict the antigen-specificity of 
the 9 murine antigens in the manuscript.."""

from DeepTCR.DeepTCR import DeepTCR_SS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Arial')

#Run Supervised Sequence Classifier
DTCRS = DeepTCR_SS('Sequence_C',device=2)


DTCRS.Get_Data(directory='../../Data/Murine_Antigens',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3)

folds = 10
seeds = np.array(range(folds))
graph_seed = 0
DTCRS.Monte_Carlo_CrossVal(folds=folds,seeds=seeds,graph_seed=graph_seed)
DTCRS.AUC_Curve(xlabel_size=24,ylabel_size=24,xtick_size=18,ytick_size=18,legend_font_size=14,frameon=False,
                diag_line=False)
DTCRS.Representative_Sequences(top_seq=100,make_seq_logos=False)


seq = np.array(DTCRS.Rep_Seq['Db-PA']['beta'])
seq_sel = seq[90]
seq_run = list(seq_sel)
ref = list(DTCRS.aa_idx.keys())
seq_run_list = []
pos = []
ref_list = []
alt_list = []
for ii,c in enumerate(seq_run,0):
    seq_run_temp = seq_run.copy()
    for r in ref:
        seq_run_temp[ii] = r
        seq_run_list.append(''.join(seq_run_temp))
        pos.append(ii)
        ref_list.append(seq_run[ii])
        alt_list.append(r)

import pandas as pd
df_out = pd.DataFrame()
df_out['seq'] = seq_run_list
df_out['pos']= pos
df_out['ref'] = ref_list
df_out['alt'] = alt_list
out_orig,_ = DTCRS.Sequence_Inference(beta_sequences=np.array([seq_sel]))
df_out['pred_baseline'] = out_orig[:,3][0]
out,_ = DTCRS.Sequence_Inference(beta_sequences=np.array(df_out['seq']))
df_out['pred'] = out[:,3]
df_out['pred_diff'] = df_out['pred_baseline']- df_out['pred']
df_out.sort_values(by='pred_diff',inplace=True)

import seaborn as sns
ax = sns.swarmplot(data=df_out,x='pos',y='pred_diff')
locs,labels = plt.xticks()
plt.xticks(locs,seq_run)
plt.axhline()

max = df_out.groupby(['pos']).agg({'pred_diff':'max'})
min = df_out.groupby(['pos']).agg({'pred_diff':'min'})
sens = max-min