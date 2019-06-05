from DeepTCR.DeepTCR import DeepTCR_WF
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

#Train Sequence Classifier
DTCR = DeepTCR_WF('Human_TIL',device='/gpu:4')
dir = 'Topalian/beta/pre_crpr_sdpd'
DTCR.Get_Data(directory='../../Data/Topalian',Load_Prev_Data=False,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=0.25,
              hla='../../Data/Topalian/HLA_Ref.csv')

folds = 500
LOO = 6
epochs_min = 50
weight_by_class = True
size_of_net = 'small'
stop_criterion = 0.25

y_pred_list = []
y_test_list = []

auc_list = []
names_list = []

names = ['Seq','VDJ','HLA','Seq+VDJ','Seq+HLA','VDJ+HLA','Seq+VDJ+HLA']
#
# #Just train w/ Sequence Information
# DTCR.use_hla = False
# DTCR.use_v_beta = False
# DTCR.use_d_beta = False
# DTCR.use_j_beta = False
# DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,
#                           weight_by_class=weight_by_class,size_of_net=size_of_net,stop_criterion=stop_criterion)
# y_pred_list.append(DTCR.y_pred)
# y_test_list.append(DTCR.y_test)
#
# for ii in range(folds):
#     auc_list.append(roc_auc_score(DTCR.y_test[ii*LOO:ii*LOO+LOO],DTCR.y_pred[ii*LOO:ii*LOO+LOO]))
#     names_list.append('Seq')
#
# #Just train w/ VDJ Information
# DTCR.use_hla = False
# DTCR.use_beta = False
# DTCR.use_v_beta = True
# DTCR.use_d_beta = True
# DTCR.use_j_beta = True
# DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,
#                           weight_by_class=weight_by_class,size_of_net=size_of_net,stop_criterion=stop_criterion)
# y_pred_list.append(DTCR.y_pred)
# y_test_list.append(DTCR.y_test)
#
# for ii in range(folds):
#     auc_list.append(roc_auc_score(DTCR.y_test[ii*LOO:ii*LOO+LOO],DTCR.y_pred[ii*LOO:ii*LOO+LOO]))
#     names_list.append('VDJ')
#
# #Just train w/HLA
# DTCR.use_hla = True
# DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,use_only_hla=True,
#                           weight_by_class=weight_by_class,size_of_net=size_of_net,stop_criterion=stop_criterion)
# y_pred_list.append(DTCR.y_pred)
# y_test_list.append(DTCR.y_test)
#
# for ii in range(folds):
#     auc_list.append(roc_auc_score(DTCR.y_test[ii*LOO:ii*LOO+LOO],DTCR.y_pred[ii*LOO:ii*LOO+LOO]))
#     names_list.append('HLA')
#
# #Just train Seq + VDJ
# DTCR.use_hla = False
# DTCR.use_beta = True
# DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,
#                           weight_by_class=weight_by_class,size_of_net=size_of_net,stop_criterion=stop_criterion)
# y_pred_list.append(DTCR.y_pred)
# y_test_list.append(DTCR.y_test)
#
# for ii in range(folds):
#     auc_list.append(roc_auc_score(DTCR.y_test[ii*LOO:ii*LOO+LOO],DTCR.y_pred[ii*LOO:ii*LOO+LOO]))
#     names_list.append('Seq+VDJ')
#
# #Just train Seq + HLA
# DTCR.use_hla = True
# DTCR.use_v_beta = False
# DTCR.use_d_beta = False
# DTCR.use_j_beta = False
# DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,
#                           weight_by_class=weight_by_class,size_of_net=size_of_net,stop_criterion=stop_criterion)
# y_pred_list.append(DTCR.y_pred)
# y_test_list.append(DTCR.y_test)
#
# for ii in range(folds):
#     auc_list.append(roc_auc_score(DTCR.y_test[ii*LOO:ii*LOO+LOO],DTCR.y_pred[ii*LOO:ii*LOO+LOO]))
#     names_list.append('Seq+HLA')
#
# #Just train VDJ + HLA
# DTCR.use_beta = False
# DTCR.use_v_beta = True
# DTCR.use_d_beta = True
# DTCR.use_j_beta = True
# DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,
#                           weight_by_class=weight_by_class,size_of_net=size_of_net,stop_criterion=stop_criterion)
# y_pred_list.append(DTCR.y_pred)
# y_test_list.append(DTCR.y_test)
#
# for ii in range(folds):
#     auc_list.append(roc_auc_score(DTCR.y_test[ii*LOO:ii*LOO+LOO],DTCR.y_pred[ii*LOO:ii*LOO+LOO]))
#     names_list.append('VDJ+HLA')

#Train with both Seq + VDJ+ HLA
DTCR.use_beta = True
epochs_min=100
folds = 25
DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,
                          weight_by_class=weight_by_class,size_of_net=size_of_net,stop_criterion=stop_criterion,
                          on_graph_clustering=True,num_clusters=12,lr_c=0.01)
y_pred_list.append(DTCR.y_pred)
y_test_list.append(DTCR.y_test)

for ii in range(folds):
    auc_list.append(roc_auc_score(DTCR.y_test[ii*LOO:ii*LOO+LOO],DTCR.y_pred[ii*LOO:ii*LOO+LOO]))
    names_list.append('Seq+VDJ+HLA')

plt.figure()
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)

for y_pred,y_test,c in zip(y_pred_list,y_test_list,names):
    roc_score = roc_auc_score(y_test[:, 0], y_pred[:, 0])
    fpr, tpr, _ = roc_curve(y_test[:, 0], y_pred[:, 0])
    plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (c, roc_score))

plt.legend(loc="lower right")
plt.savefig('Human_TIL_AUC_2.eps')


df_plot = pd.DataFrame()
df_plot['Method'] = names_list
df_plot['AUC'] = auc_list
#sns.swarmplot(data=df_plot,x='Method',y='AUC',order=['Seq','HLA','Seq+HLA'])
#sns.boxplot(data=df_plot,x='Method',y='AUC',order=['Seq','HLA','Seq+HLA'])
sns.violinplot(data=df_plot,x='Method',y='AUC',order=names)

for ii,n in enumerate(names,0):
    print(mannwhitneyu(df_plot[df_plot['Method']==names[ii]]['AUC'],df_plot[df_plot['Method']==names[ii+1]]['AUC'])[1])

df_plot.groupby(['Method']).agg({'AUC':'mean'})

DTCR.UMAP_Plot(by_class=True,freq_weight=True,scale=5000,Load_Prev_Data=False,alpha=0.5,prob_plot='crpr')

import numpy as np
from sklearn.preprocessing import MinMaxScaler

indices = DTCR.beta_indices
max_len = np.sum(DTCR.X_Seq_beta>0,-1)
loc = indices/max_len

features_norm = MinMaxScaler().fit_transform(DTCR.beta_features)
max_ft = np.argmax(features_norm,axis=-1)
loc_max = []
ind_max = []
perc_max = []
for i,c,f,ft in zip(indices,loc,max_ft,features_norm):
    loc_max.append(c[f])
    ind_max.append(i[f])
    perc_max.append(ft[f])

loc_max = np.asarray(loc_max)
ind_max = np.asarray(ind_max)
perc_max = np.asarray(perc_max)

top = 100
sel = 'sdpd'
DTCR.Representative_Sequences(top)
Rep_Seq = DTCR.Rep_Seq
index = np.asarray(Rep_Seq[sel].index[0:top])
loc_sel = loc_max[index]
ind_sel = ind_max[index]
perc_sel = perc_max[index]

motifs = []
for i,ind in zip(ind_sel,index):
    motifs.append(DTCR.beta_sequences[ind][int(i):int(i)+5])

Rep_Seq[sel]['Motifs'] = motifs
Rep_Seq[sel]['Percentile'] = perc_sel

Rep_Seq['crpr']['vdj'] = Rep_Seq['crpr']['v_beta'] + Rep_Seq['crpr']['d_beta'] + Rep_Seq['crpr']['j_beta']
Rep_Seq['sdpd']['vdj'] = Rep_Seq['sdpd']['v_beta'] + Rep_Seq['sdpd']['d_beta'] + Rep_Seq['sdpd']['j_beta']


Rep_Seq['crpr']['j_beta'].value_counts()
Rep_Seq['sdpd']['j_beta'].value_counts()


plt.hist(np.ndarray.flatten(loc))
plt.xlim([0,1])
DTCR.Representative_Sequences(100)

perc_all = []
cohort = []

index = np.asarray(DTCR.Rep_Seq['crpr'].index[0:100])
perc_sel = perc_max[index]
perc_all.append(perc_sel)
cohort.append(['crpr']*len(perc_sel))

index = np.asarray(DTCR.Rep_Seq['sdpd'].index[0:100])
perc_sel = perc_max[index]
perc_all.append(perc_sel)
cohort.append(['sdpd']*len(perc_sel))

perc_all = np.hstack(perc_all)
cohort = np.hstack(cohort)

df_perc = pd.DataFrame()
df_perc['Ft'] = perc_all
df_perc['cohort'] = cohort

sns.swarmplot(data=df_perc,x='cohort',y='Ft')


DTCR_HA = DeepTCR_WF('HA')
DTCR_HA.Get_Data(directory='../../Data/Human_Antigens',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3)

beta_sequences = DTCR_HA.beta_sequences
v_beta = DTCR.v_beta
d_beta = DTCR.d_beta
j_beta = DTCR.j_beta
counts = np.ones(len(j_beta))
sample_id = np.asarray(range(len(counts)))
sample_id = np.asarray(['seq'+str(s) for s in sample_id])


DTCR.Sample_Inference(sample_labels=sample_id,beta_sequences=beta_sequences,counts=counts)