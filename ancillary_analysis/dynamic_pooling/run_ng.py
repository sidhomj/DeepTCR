import pandas as pd
import numpy as np
from DeepTCR.DeepTCR import DeepTCR_WF
import matplotlib.pyplot as plt
import os
import pickle

meta = pd.read_csv('../../Data/natgen/cohort1_meta.csv')
meta[meta.columns[0]] = meta[meta.columns[0]]+'.tsv'
meta[meta.columns[4]] = 'CMV'+meta[meta.columns[4]]
meta = meta[meta[meta.columns[4]]!='CMVUnknown']
# meta = meta.sample(10)
label_dict = dict(zip(meta[meta.columns[0]],meta[meta.columns[4]]))

DTCR_l = DeepTCR_WF('load_5000')
DTCR_l.Get_Data(directory='../../Data/natgen/data/cohort1/',Load_Prev_Data=True,
                aa_column_beta=1,v_beta_column=10,d_beta_column=13,j_beta_column=16,count_column=5,
                type_of_data_cut='Num_Seq',
                data_cut=5000)

idx = np.isin(DTCR_l.sample_id,meta[meta.columns[0]])
beta_sequences = DTCR_l.beta_sequences[idx]
v_beta = DTCR_l.v_beta[idx]
d_beta = DTCR_l.d_beta[idx]
j_beta = DTCR_l.j_beta[idx]
counts = DTCR_l.counts[idx]
sample_labels = DTCR_l.sample_id[idx]
class_labels = np.array(list(map(label_dict.get,sample_labels)))

DTCR = DeepTCR_WF('ng')
DTCR.Load_Data(beta_sequences=beta_sequences,v_beta=v_beta,d_beta=d_beta,j_beta=j_beta,counts=counts,
               sample_labels=sample_labels,class_labels=class_labels)

folds = 1
epochs_min = 5
size_of_net = 'medium'
num_concepts = 12
hinge_loss_t = 0.0
train_loss_min=0.2
seeds = np.array(range(folds))
graph_seed = 0
DTCR.Monte_Carlo_CrossVal(folds=folds,seeds=seeds,graph_seed=graph_seed,l2_reg=0.0,epochs_min=epochs_min,
                          num_concepts=num_concepts,size_of_net=size_of_net,
                          test_size=0.40,combine_train_valid=True,train_loss_min=train_loss_min,
                          hinge_loss_t=hinge_loss_t,subsample=500)
                          #num_agg_layers=1,units_agg=12)
# DTCR.AUC_Curve()
with open(os.path.join(DTCR.Name, 'seq_features.pkl'), 'rb') as f:
    DTCR.features = pickle.load(f)

with open(os.path.join(DTCR.Name, 'seq_preds.pkl'), 'rb') as f:
    DTCR.predicted = pickle.load(f)

with open(os.path.join(DTCR.Name, 'split_indices.pkl'), 'rb') as f:
    DTCR.train_idx, DTCR.valid_idx, DTCR.test_idx = pickle.load(f)

# DTCR.UMAP_Plot(set='train',by_class=True,sample_per_class=1000,Load_Prev_Data=False,plot_by_class=True,scale=10)
# DTCR.UMAP_Plot(set='train',by_sample=True,sample_per_class=10000,Load_Prev_Data=True,plot_by_class=True,scale=10)
DTCR.HeatMap_Sequences(set='train',by_sample=True,sample_num_per_class=10000,figsize=(7,7),color_dict={'CMV+':'b','CMV-':'r'})
DTCR.HeatMap_Samples(set='train',figsize=(7,7),color_dict={'CMV+':'b','CMV-':'r'})

from scipy.spatial.distance import pdist, squareform
features = DTCR.features[DTCR.test_idx]
sel = np.random.choice(range(len(features)), 1000, replace=False)
features = features[sel]
pw = squareform(pdist(features))
import seaborn as sns
sns.clustermap(pw,figsize=(7,7),cmap='jet')
plt.subplots(figsize=(7,7))
sns.heatmap(pw)

sel = np.random.choice(DTC)

