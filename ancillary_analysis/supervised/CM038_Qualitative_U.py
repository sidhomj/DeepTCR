"""
Figure 3f-h, Supplementary Figure 23
"""

"""This script is used to provide a descriptive analysis of the distribution of TCR sequences
within the CheckMate-038 clinical trial.
"""

import pickle
import numpy as np
import pandas as pd
import umap
from DeepTCR.DeepTCR import DeepTCR_WF,DeepTCR_U
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import gaussian_kde

os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

DTCR = DeepTCR_WF('Human_TIL',device='/device:GPU:0')
DTCR.Get_Data(directory='../../Data/CheckMate_038',Load_Prev_Data=False,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data/CheckMate_038/HLA_Ref_sup_AB.csv')

with open('cm038_ft_pred_perc.pkl','rb') as f:
    features,predicted,perc = pickle.load(f)

win = 10
cut_bottom = np.percentile(predicted[:,0],win)
cut_top = np.percentile(predicted[:,0],100-win)

df_plot = pd.DataFrame()
df_plot['beta'] = DTCR.beta_sequences
df_plot['sample'] = DTCR.sample_id
df_plot['pred'] = predicted[:,0]
df_plot['gt'] = DTCR.class_id
df_plot['freq'] = DTCR.freq

plt.figure()
ax = sns.distplot(df_plot['pred'],1000,color='k',kde=False)
N,bins= np.histogram(df_plot['pred'],1000)
for p,b in zip(ax.patches,bins):
    if b < cut_bottom:
        p.set_facecolor('r')
    elif b > cut_top:
        p.set_facecolor('b')
y_min,y_max = plt.ylim()
plt.xlim([0,1])
plt.xticks(np.arange(0.0,1.1,0.1))
plt.yticks([])
plt.xlabel('')
plt.ylabel('')
plt.show()

beta_sequences = DTCR.beta_sequences
v_beta = DTCR.v_beta
j_beta = DTCR.j_beta
d_beta = DTCR.d_beta
hla = DTCR.hla_data_seq
sample_id = DTCR.sample_id

file = 'cm038_x2_u.pkl'
featurize = False
if featurize:
    DTCR_U = DeepTCR_U('test_hum', device='/device:GPU:6')
    DTCR_U.Load_Data(beta_sequences=beta_sequences, v_beta=v_beta, d_beta=d_beta, j_beta=j_beta, hla=hla)
    DTCR_U.Train_VAE(Load_Prev_Data=False, latent_dim=64,stop_criterion=0.01)
    X_2 = umap.UMAP().fit_transform(DTCR_U.features)
    with open(file, 'wb') as f:
        pickle.dump(X_2, f, protocol=4)
else:
    with open(file,'rb') as f:
        X_2 = pickle.load(f)


df_plot['x'] = X_2[:,0]
df_plot['y'] = X_2[:,1]

idx_crpr = predicted[:,0] >= cut_top
idx_sdpd = predicted[:,0] <= cut_bottom
df_plot['label'] = None
df_plot['label'].iloc[idx_crpr] = 'crpr'
df_plot['label'].iloc[idx_sdpd] = 'sdpd'
df_plot['label'] = df_plot['label'].fillna(value='out')
label_dict = {'crpr':'b','sdpd':'r','out':'darkgrey'}
df_plot['c'] = df_plot['label'].map(label_dict)

#Plot crpr sequences
df_plot_crpr = df_plot[df_plot['label']!='sdpd']
plt.figure()
df_plot_crpr.sort_values(by='pred',ascending=True,inplace=True)
plt.scatter(df_plot_crpr['x'],df_plot_crpr['y'],c=df_plot_crpr['c'],s=1)
plt.xticks([])
plt.yticks([])
x_min,x_max = plt.xlim()
y_min,y_max = plt.ylim()

#Plot sdpd sequences
df_plot_sdpd = df_plot[df_plot['label']!='crpr']
plt.figure()
df_plot_sdpd.sort_values(by='pred',ascending=False,inplace=True)
plt.scatter(df_plot_sdpd['x'],df_plot_sdpd['y'],c=df_plot_sdpd['c'],s=1)
plt.xticks([])
plt.yticks([])
plt.xlim([x_min,x_max])
plt.ylim([y_min,y_max])

ref = df_plot.groupby(['sample']).agg({'gt':'first'}).reset_index()
ref.sort_values(by='gt',ascending=False,inplace=True)
ref_pred = pd.read_csv('sample_tcr_hla.csv')
ref_pred = ref_pred.groupby(['Samples']).agg({'y_pred':'mean'}).reset_index()
ref_dict = dict(zip(ref_pred['Samples'],ref_pred['y_pred']))
ref['pred'] = ref['sample'].map(ref_dict)
ref.sort_values(by='pred',inplace=True)

n_rows = 4
n_cols = 11
fig,ax = plt.subplots(n_rows,n_cols,figsize=(13,5))
ax = np.ndarray.flatten(ax)
for s,l,r,a in zip(ref['sample'],ref['gt'],ref['pred'],ax):
    df_temp =df_plot[df_plot['sample']==s]
    df_temp = df_temp[df_temp['label']!='out']
    a.scatter(df_temp['x'], df_temp['y'], c=df_temp['c'], s=0.1)
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlim([x_min,x_max])
    a.set_ylim([y_min,y_max])
    a.set_title(np.round(r,3))
    if l == 'crpr':
        c = 'b'
    else:
        c = 'r'
    for axis in ['top', 'bottom', 'left', 'right']:
        a.spines[axis].set_linewidth(3)

    for axis in ['top', 'bottom', 'left', 'right']:
        a.spines[axis].set_color(c)

lef = n_cols*n_rows-43
for ii in range(43,43+lef):
    ax[ii].remove()
plt.subplots_adjust(top = 0.9, bottom=0.1, hspace=0.5)

#Plot unfiltered sample repertoires
def gaussian_density(x,y,w=None):
    xy = np.vstack([x, y,w])
    z = gaussian_kde(xy)(xy)
    r = np.argsort(z)
    x ,y, z = x[r], y[r], z[r]
    return x,y,z

n_rows = 4
n_cols = 11
fig,ax = plt.subplots(n_rows,n_cols,figsize=(13,5))
ax = np.ndarray.flatten(ax)
for s,l,r,a in zip(ref['sample'],ref['gt'],ref['pred'],ax):
    df_temp =df_plot[df_plot['sample']==s]
    x = np.array(df_temp['x'])
    y = np.array(df_temp['y'])
    x,y,z = gaussian_density(x,y,df_temp['freq'])
    a.scatter(x, y, s=0.1,c=z)
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlim([x_min,x_max])
    a.set_ylim([y_min,y_max])
    a.set_title(np.round(r,3))
    if l == 'crpr':
        c = 'b'
    else:
        c = 'r'
    for axis in ['top', 'bottom', 'left', 'right']:
        a.spines[axis].set_linewidth(3)

    for axis in ['top', 'bottom', 'left', 'right']:
        a.spines[axis].set_color(c)

lef = n_cols*n_rows-43
for ii in range(43,43+lef):
    ax[ii].remove()
plt.subplots_adjust(top = 0.9, bottom=0.1, hspace=0.5)

# df_out = pd.DataFrame(DTCR_U.features)
# df_out['pred'] = predicted[:,0]
# df_out['label'] = DTCR.class_id
# df_out['sample'] = DTCR.sample_id
# df_out['freq'] = DTCR.freq
# df_out['counts'] = DTCR.counts
# df_out.to_csv('cm038_ft_u.csv',index=False)