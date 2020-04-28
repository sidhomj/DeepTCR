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

test_peptide = 'TSNLQEQIGW'
label_keep = np.array([test_peptide, 'CEF', 'NoPeptide','AY9'])
idx = np.isin(class_labels, label_keep)
class_dict = {test_peptide: 'Cognate', 'CEF': 'non-Cognate', 'NoPeptide': 'non-Cognate', 'AY9': 'non-Cognate'}
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

#AUC
DTCR.AUC_Curve(xlabel_size=24,ylabel_size=24,xtick_size=18,ytick_size=18,legend_font_size=18,frameon=False,
                diag_line=False,title=None,title_font=24,by='Cognate')
plt.savefig('auc_example.eps')

#Preds Distribution
test_peptide = 'Cognate'
c = np.where(DTCR.lb.classes_ == test_peptide)[0][0]
color_dict = {1:'b',0:'r'}
label_dict = {1:test_peptide,0:'Non-'+test_peptide}
df_preds = pd.DataFrame()
df_preds['y_test'] = DTCR.y_test[:,c]
df_preds['y_pred'] = DTCR.y_pred[:,c]
df_preds['color'] = df_preds['y_test'].map(color_dict)
df_preds['label'] = df_preds['y_test'].map(label_dict)
plt.figure()
sns.violinplot(data=df_preds,x='label',y='y_pred',cut=0,palette=['r','b'])
plt.xlabel('')
plt.ylabel('P('+test_peptide+')',fontsize=24)
plt.xticks(size=18)
plt.yticks(size=18)
plt.ylim([0,1])
plt.subplots_adjust(left=0.15)
plt.savefig('preds_example.eps')

#Preds by counts
idx = DTCR.class_id==test_peptide
x = np.squeeze(DTCR.predicted[idx,c])
y = np.log2(DTCR.counts[idx]+1)
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
r = np.argsort(z)
x, y, z = x[r], y[r], z[r]
plt.figure()
plt.scatter(x, y, s=15, c=z, cmap=plt.cm.jet)
plt.xlabel('Predicted',fontsize=24)
plt.ylabel('Log2(counts+1)',fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig('preds_v_counts.eps')

#Representative Sequences
DTCR.Representative_Sequences(top_seq=100,make_seq_logos=False)
rep_seq = DTCR.Rep_Seq[test_peptide][0:10]
models = None
DTCR.Residue_Sensitivity_Logo(beta_sequences=np.array(rep_seq['beta']),
                                models=models,class_sel=test_peptide,Load_Prev_Data=False,background_color='black',
                              edgewidth=0.0,figsize=(3,4),min_size=0.25,norm_to_seq=True)
plt.savefig('example_logo.png',dpi=1200)