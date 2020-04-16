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
matplotlib.rc('font', family='Arial')


gpu = 1
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
folds=10
graph_seed=0
seeds=seeds = np.array(range(folds))

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

group_1 = ['TSNLQEQIAW', 'TSNLQEQIGW', 'TSTLAEQIAW', 'TSTLAEQMAW',
       'TSTLAEQVAW', 'TSTLQEQIEW', 'TSTLQEQIGW', 'TSTLSEQIAW',
       'TSTLSEQVAW', 'TSTLTEQIAW', 'TSTLTEQVAW', 'TSTLVEQIAW']
group_2 = ['ISPRTLNAW', 'MSPRTLNAW']
group_3 =  ['KIRLRPGGKKKYKLK', 'KIRLRPGGKKRYKLK']
group_4 = ['KAALDLSHF','KAAVDLSHF', 'KGALDLSHF','KSALDLSHF','TAALDMSHF']
group_5 = ['HTQGYFPDW','NTQGYFPDW']
group_6 =  ['FFPDWQNYT','YFPDWQNYT']
group = [group_1,group_2,group_3,group_4,group_5,group_6]
group_name = ['GAG TW10','GAG IW9', 'GAG KK15','NEF KF9','NEF HW9','NEF YT9']
class_dict = {}
for n,g in zip(group_name,group):
    for e in g:
        class_dict[e] = n
df_screen = pd.read_csv('screen.csv')
df_screen['group'] = df_screen['epitope'].map(class_dict)
df_screen.sort_values(by='pred_diff',inplace=True,ascending=False)

pos_1 = df_screen.iloc[0]['epitope']
# pos_1 = 'KIRLRPGGKKKYKLK'
# neg_1 = 'KIRLRPGGKKRYKLK'
# pos_2 = 'KSALDLSHF'
# neg_2 = 'TAALDMSHF'
#positive examples
test_peptide = pos_1
label_keep = np.array([test_peptide, 'CEF', 'NoPeptide'])
idx = np.isin(class_labels, label_keep)
DTCR = DeepTCR_WF('train_ex', device=gpu)
DTCR.Load_Data(beta_sequences=beta_sequences[idx],
               v_beta=v_beta[idx],
               d_beta=d_beta[idx],
               j_beta=j_beta[idx],
               counts=counts[idx],
               class_labels=class_labels[idx],
               sample_labels=sample_labels[idx])
DTCR.Monte_Carlo_CrossVal(folds=folds,graph_seed=graph_seed,seeds=seeds,
                          LOO=len(label_keep), combine_train_valid=True, num_concepts=64,
                          convergence='training', epochs_min=100,train_loss_min=0.25)
DTCR.Representative_Sequences(top_seq=100,motif_seq=10,color_scheme='hydrophobicity',make_seq_logos=False)
rep_seq = DTCR.Rep_Seq[pos_1].iloc[0:]
models = ['model_0']
import pickle
# with open('rep_seq.pkl','wb') as f:
#     pickle.dump(rep_seq,f,protocol=4)

with open('rep_seq.pkl','rb') as f:
    rep_seq = pickle.load(f)

rep_seq = rep_seq.iloc[0:10]
models = np.random.choice(range(10),5,replace=False)
models = ['model_'+str(x) for x in models]
DTCR.Residue_Sensitivity_Logo(alpha_sequences=np.array(rep_seq['alpha']),beta_sequences=np.array(rep_seq['beta']),
                              v_beta = np.array(rep_seq['v_beta']),d_beta=np.array(rep_seq['d_beta']),j_beta=np.array(rep_seq['j_beta']),
                                models=models,class_sel=test_peptide)

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
df_preds = pd.DataFrame()
df_preds['y_test'] = DTCR.y_test[:,1]
df_preds['y_pred'] = DTCR.y_pred[:,1]
plt.figure()
sns.violinplot(data=df_preds,x='y_test',y='y_pred',cut=0)

idx_pos = np.where(DTCR.class_id==test_peptide)[0]
