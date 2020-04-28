from DeepTCR.DeepTCR import DeepTCR_WF, DeepTCR_SS
import glob
import os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import pandas as pd
import pickle
import umap
import matplotlib.pyplot as plt
gpu = 0
folds=100
graph_seed=0
seeds= np.array(range(folds))

def GKDE(x,y,z=None):
    xy = np.vstack([x, y])
    kernel = gaussian_kde(xy,weights=z)
    z = kernel(xy)
    r = np.argsort(z)
    x ,y, z = x[r], y[r], z[r]
    return x,y,z,kernel,r

with open('tw10_seq.pkl','rb') as f:
    sequences,seq_class_labels,predicted,counts = pickle.load(f)

#Collect predicted antigen-specific sequences
thresh = 0.99
seq_train = []
label_train = []
count_train = []
for s,seq_cl,p,c in zip(sequences,seq_class_labels,predicted,counts):
    sel_idx = p > thresh
    seq_train.append(s[sel_idx])
    label_train.append(seq_cl[sel_idx])
    count_train.append(c[sel_idx])

seq_train = np.hstack(seq_train)
label_train = np.hstack(label_train)
count_train = np.hstack(count_train)

#Train Sequence Classifier
DTCR = DeepTCR_SS('tw10_seq',device=gpu)
DTCR.Load_Data(beta_sequences=seq_train,class_labels=label_train)
DTCR.Monte_Carlo_CrossVal(folds=folds,graph_seed=graph_seed,seeds=seeds,convergence='training')
y_pred = DTCR.predicted
y_test = DTCR.Y
plt.figure(figsize=(6,5))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
for ii,cl in enumerate(DTCR.lb.classes_,0):
    fpr,tpr,_ = roc_curve(y_test[:,ii],y_pred[:,ii])
    roc_score = roc_auc_score(y_test[:,ii],y_pred[:,ii])
    label = '%s = %0.3f'  % (cl,roc_score)
    plt.plot(fpr,tpr,lw=2,label=label)
plt.legend(loc='lower right', frameon=False,prop={'size': 10})
ax = plt.gca()
ax.xaxis.label.set_size(24)
ax.yaxis.label.set_size(24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.subplots_adjust(bottom=0.15,left=0.15)
plt.savefig('multiclass_roc.png',dpi=1200)

#Learn UMAP
X_2 = umap.UMAP().fit_transform(DTCR.predicted)
plt.scatter(X_2[:,0],X_2[:,1])
ylim = plt.ylim()
xlim = plt.xlim()
plt.close()

#Plot in 1 figure
fig,ax = plt.subplots(4,3,figsize=(10,10))
ax = np.ndarray.flatten(ax)
for ii,l in enumerate(DTCR.lb.classes_,0):
    sel_idx = DTCR.class_id==l
    x = X_2[sel_idx,0]
    y = X_2[sel_idx,1]
    x,y,c,_,_ = GKDE(x,y)
    ax[ii].scatter(x,y,c=c,cmap='jet',s=5)
    ax[ii].set_xlim(xlim)
    ax[ii].set_ylim(ylim)
    ax[ii].set_title(l)
    ax[ii].set_xticks([])
    ax[ii].set_yticks([])
plt.tight_layout()

dir_write = 'umap_tw10'
if not os.path.exists(dir_write):
    os.makedirs(dir_write)

#Plot in separate figures
for ii,l in enumerate(DTCR.lb.classes_,0):
    sel_idx = DTCR.class_id==l
    fig,ax = plt.subplots(figsize=(5,5))
    x = X_2[sel_idx,0]
    y = X_2[sel_idx,1]
    x,y,c,_,_ = GKDE(x,y)
    ax.scatter(x,y,c=c,cmap='jet',s=100)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(l,fontsize=36)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(os.path.join(dir_write,l+'.png'),dpi=1200)
    plt.close()

#Get Residue Sensitivity Logo for select epitopes
DTCR.Representative_Sequences(top_seq=100,make_seq_logos=False)
test_peptide = 'TSTLQEQIGW'
rep_seq = DTCR.Rep_Seq[test_peptide]['beta'][0:10]
models = np.random.choice(range(100),5,replace=False)
models = ['model_'+str(x) for x in models]
models = None
DTCR.Residue_Sensitivity_Logo(beta_sequences=np.array(rep_seq),models=models,
                                class_sel=test_peptide,Load_Prev_Data=False,background_color='black',
                              edgewidth=0.0,figsize=(3,4),min_size=0.25,norm_to_seq=True)
plt.savefig(test_peptide+'.png',dpi=1200)

test_peptide = 'TSTLTEQVAW'
rep_seq = DTCR.Rep_Seq[test_peptide]['beta'][0:10]
DTCR.Residue_Sensitivity_Logo(beta_sequences=np.array(rep_seq),models=models,
                                class_sel=test_peptide,Load_Prev_Data=False,background_color='black',
                              edgewidth=0.0,figsize=(3,4),min_size=0.25,norm_to_seq=True)
plt.savefig(test_peptide+'.png',dpi=1200)