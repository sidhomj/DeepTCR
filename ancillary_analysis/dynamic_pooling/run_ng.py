import pandas as pd
import numpy as np
from DeepTCR.DeepTCR import DeepTCR_WF

meta = pd.read_csv('../../Data/natgen/cohort1_meta.csv')
meta[meta.columns[0]] = meta[meta.columns[0]]+'.tsv'
meta[meta.columns[4]] = 'CMV'+meta[meta.columns[4]]
meta = meta[meta[meta.columns[4]]!='CMVUnknown']
# meta = meta.sample(10)
label_dict = dict(zip(meta[meta.columns[0]],meta[meta.columns[4]]))

DTCR_l = DeepTCR_WF('load_500')
DTCR_l.Get_Data(directory='../../Data/natgen/data/cohort1/',Load_Prev_Data=True,
                aa_column_beta=1,v_beta_column=10,d_beta_column=13,j_beta_column=16,count_column=5,
                type_of_data_cut='Num_Seq',
                data_cut=1000)

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
                          hinge_loss_t=hinge_loss_t,subsample=None)
                          #num_agg_layers=1,units_agg=12)
# DTCR.AUC_Curve()
DTCR.UMAP_Plot(set='train',by_class=True,sample_per_class=100,prob_plot='CMV+')

import umap
import warnings
import os
import pickle
idx = None
features = DTCR.features
class_id = DTCR.class_id
sample_id = DTCR.sample_id
freq = DTCR.freq
predicted = DTCR.predicted

features_temp = []
class_temp = []
sample_temp = []
freq_temp = []
predicted_temp = []
cluster_temp = []

for i in DTCR.lb.classes_:
    sel = np.where(class_id == i)[0]
    sel = np.random.choice(sel, 1000, replace=False)
    features_temp.append(features[sel])
    class_temp.append(class_id[sel])
    sample_temp.append(sample_id[sel])
    freq_temp.append(freq[sel])
    predicted_temp.append(predicted[sel])

features = np.vstack(features_temp)
class_id = np.hstack(class_temp)
sample_id = np.hstack(sample_temp)
freq = np.hstack(freq_temp)
predicted = np.hstack(predicted_temp)
IDX = None
umap_obj = umap.UMAP()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    X_2 = umap_obj.fit_transform(features)
with open(os.path.join(DTCR.Name, 'umap.pkl'), 'wb') as f:
    pickle.dump([X_2, features, class_id, sample_id, freq, IDX, idx], f, protocol=4)

df_plot = pd.DataFrame()
df_plot['x'] = X_2[:, 0]
df_plot['y'] = X_2[:, 1]
df_plot['Class'] = class_id
df_plot['Sample'] = sample_id
