from DeepTCR.DeepTCR import DeepTCR_SS, DeepTCR_WF

folds = 25
LOO = 4
epochs_min = 100

# #Train Sequence Classifier
# DTCR_SS = DeepTCR_SS('Rudqvist_SS')
# DTCR_SS.Get_Data(directory='../../Data/Rudqvist',Load_Prev_Data=False,
#                aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21)
#
# DTCR_SS.Monte_Carlo_CrossVal(folds=folds,test_size=0.25)
# DTCR_SS.AUC_Curve(filename='AUC.eps')

#Train Repertoire Classifier without on-graph clustering
import os
os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DTCR_WF = DeepTCR_WF('Rudqvist_WF',device='/device:GPU:1')
DTCR_WF.Get_Data(directory='../../Data/Rudqvist',Load_Prev_Data=False,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21)

DTCR_WF.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,stop_criterion=0.01,num_clusters=64)
# DTCR_WF.AUC_Curve(filename='Rep_AUC.eps')

#Train Repertoire Classifier with on-graph clustering
#DTCR_WF.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,on_graph_clustering=True,epochs_min=epochs_min)
DTCR_WF.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,gcn=True,epochs_min=epochs_min,num_clusters=12)
DTCR_WF.AUC_Curve(filename='Rep_AUC_clustering.eps')
DTCR_WF.AUC_Curve()

#Visualize Latent Space
DTCR_WF.UMAP_Plot(by_class=True,freq_weight=True,show_legend=True,scale=5000,Load_Prev_Data=False,
                  alpha=0.5)

import pandas as pd
import seaborn as sns
data = pd.DataFrame(DTCR_WF.weights)
data.columns = DTCR_WF.lb.classes_
sns.clustermap(data=data,cmap='bwr',standard_scale=1,yticklabels=False)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

i = 3
sel = DTCR_WF.lb.classes_[i]
idx = np.where(DTCR_WF.class_id==sel)[0]
x = DTCR_WF.freq[idx]
y = DTCR_WF.weights[idx,i]
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
r = np.argsort(z)
x, y, z = x[r], y[r], z[r]
plt.scatter(x,y,c=z,s=5)
plt.xlabel('Freq')
plt.ylabel('Weights')
plt.title(DTCR_WF.lb.classes_[i])
plt.xlim((-0.1,1.1))
plt.ylim((-0.1,1.1))

