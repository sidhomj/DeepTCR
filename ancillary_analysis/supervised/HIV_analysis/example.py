from DeepTCR.DeepTCR import DeepTCR_WF
import glob
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import pandas as pd
import matplotlib
import pickle
matplotlib.rc('font', family='Arial')


gpu = 1
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
folds=100
graph_seed=0
seeds= np.array(range(folds))

files = glob.glob('../../../Data/HIV/*.tsv')
samples = []
labels = []
for file in files:
    file = file.split('/')[-1]
    samples.append(file)
    labels.append(file.split('_')[1])

label_dict = dict(zip(samples,labels))

DTCR = DeepTCR_WF('load')
DTCR.Get_Data('../../../Data/HIV',aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,
              type_of_data_cut='Read_Cut',data_cut=10)

idx = np.isin(DTCR.sample_id,np.array(list(label_dict.keys())))
beta_sequences = DTCR.beta_sequences[idx]
v_beta = DTCR.v_beta[idx]
d_beta = DTCR.d_beta[idx]
j_beta = DTCR.j_beta[idx]
sample_labels = DTCR.sample_id[idx]
counts = DTCR.counts[idx]
class_labels  = np.array([label_dict[x] for x in sample_labels])

test_peptide = 'TSTLQEQIGW'
label_keep = np.array([test_peptide, 'CEF', 'NoPeptide','AY9'])
idx = np.isin(class_labels, label_keep)
class_dict = {test_peptide: test_peptide, 'CEF': 'non-Cognate', 'NoPeptide': 'non-Cognate', 'AY9': 'non-Cognate'}
class_relabel = class_labels[idx]
class_relabel = np.array([class_dict[x] for x in class_relabel])
DTCR = DeepTCR_WF('screen', device=gpu)
DTCR.Load_Data(beta_sequences=beta_sequences[idx],
               counts=counts[idx],
               class_labels=class_relabel,
               sample_labels=sample_labels[idx])
DTCR.Monte_Carlo_CrossVal(folds=folds,graph_seed=graph_seed,seeds=seeds,
                          LOO=2, combine_train_valid=True, num_concepts=64,
                          convergence='training',train_loss_min=0.10)
DTCR.Representative_Sequences(top_seq=100,make_seq_logos=False)
rep_seq = DTCR.Rep_Seq[test_peptide]
import pickle
with open('rep_seq.pkl','wb') as f:
    pickle.dump(rep_seq,f,protocol=4)
with open('rep_seq.pkl','rb') as f:
    rep_seq = pickle.load(f)

# agg_dict = {test_peptide:'mean','v_beta':'first','d_beta':'first','j_beta':'first'}
# rep_seq = rep_seq.groupby(['beta']).agg(agg_dict).reset_index().sort_values(by=test_peptide,ascending=False)
rep_seq = rep_seq.iloc[0:10]
models = np.random.choice(range(25),1,replace=False)
models = ['model_'+str(x) for x in models]
models = None
DTCR.Residue_Sensitivity_Logo(beta_sequences=np.array(rep_seq['beta']),
                                models=models,class_sel=test_peptide,Load_Prev_Data=True,background_color='black',
                              edgewidth=0.0,figsize=(3,5),min_size=0.25,norm_to_seq=True)
# v_beta = np.array(rep_seq['v_beta']), d_beta = np.array(rep_seq['d_beta']), j_beta = np.array(rep_seq['j_beta']),

c = np.where(DTCR.lb.classes_ == test_peptide)[0][0]
idx_pos = DTCR.y_test[:,c]==1
mag = np.mean(DTCR.y_pred[idx_pos,c])-np.mean(DTCR.y_pred[~idx_pos,c])

DTCR.AUC_Curve(xlabel_size=24,ylabel_size=24,xtick_size=18,ytick_size=18,legend_font_size=14,frameon=False,
                diag_line=False,title=test_peptide,title_font=24)
c = np.where(DTCR.lb.classes_ == test_peptide)[0][0]
idx = DTCR.class_id==test_peptide
x = np.squeeze(DTCR.predicted[idx,c])
y = np.log2(DTCR.counts[idx]+1)
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
r = np.argsort(z)
x, y, z = x[r], y[r], z[r]
plt.figure()
plt.scatter(x, y, s=15, c=z, cmap=plt.cm.jet)
plt.title(test_peptide,fontsize=24)
plt.xlabel('Predicted',fontsize=24)
plt.ylabel('Log2(counts+1)',fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()

roc_auc_score(DTCR.y_test[:,2],DTCR.y_pred[:,2])
color_dict = {1:'b',0:'r'}
df_preds = pd.DataFrame()
df_preds['y_test'] = DTCR.y_test[:,c]
df_preds['y_pred'] = DTCR.y_pred[:,c]
df_preds['color'] = df_preds['y_test'].map(color_dict)
plt.figure()
sns.violinplot(data=df_preds,x='y_test',y='y_pred',cut=0,palette=['r','b'])

idx_pos = np.where(DTCR.class_id==test_peptide)[0]