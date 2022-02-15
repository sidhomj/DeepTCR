"""Figure 3B"""

"""This script is used to train both the sequence and repertoire classifier on the
Rudqvist_2017 dataset and compare their performances."""

from DeepTCR.DeepTCR import DeepTCR_SS, DeepTCR_WF
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from matplotlib import pyplot as plt

#Train Sequence Classifier
# DTCR_SS = DeepTCR_SS('Rudqvist_SS')
# DTCR_SS.Get_Data(directory='../../../Data/Rudqvist',Load_Prev_Data=False,
#                aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,
#                  type_of_data_cut='Num_Seq',
#                  data_cut=100)


# #Train Sequence Classifier
# DTCR_SS = DeepTCR_SS('help_model')
# DTCR_SS.Get_Data(directory='../../../Data/help',Load_Prev_Data=False,aggregate_by_aa=True,
#                aa_column_beta=0,count_column=4,v_beta_column=1,j_beta_column=3)
#
# DTCR_SS.Get_Train_Valid_Test()
# DTCR_SS.Train()
# DTCR_SS.Monte_Carlo_CrossVal(folds=3)
# DTCR_SS.K_Fold_CrossVal(folds=3)

#Train Repertoire Classifier
folds = 25
LOO = 4
epochs_min = 10
size_of_net = 'small'
num_concepts=64
hinge_loss_t = 0.1
train_loss_min=1.0
seeds = np.array(range(folds))
graph_seed=0

DTCR_WF = DeepTCR_WF('Rudqvist_WF2')
DTCR_WF.Get_Data(directory='../../../Data/Rudqvist',Load_Prev_Data=False,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,
                 type_of_data_cut='Num_Seq',
                 data_cut=100)

DTCR_WF.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,num_concepts=num_concepts,size_of_net=size_of_net,
                             train_loss_min=train_loss_min,hinge_loss_t=hinge_loss_t,combine_train_valid=True,seeds=seeds,
                             graph_seed=graph_seed,multisample_dropout=False,multisample_dropout_rate=0.50,multisample_dropout_num_masks=64)

DTCR_WF.AUC_Curve()

#Create plot to compare SS/WF performance
plt.figure()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

classes = DTCR_WF.lb.classes_
color_dict = {'Control':'limegreen','9H10':'red','RT':'darkorange','Combo':'magenta'}
for ii,c in enumerate(classes,0):
    roc_score = roc_auc_score(DTCR_SS.y_test[:,ii], DTCR_SS.y_pred[:,ii])
    fpr, tpr, _ = roc_curve(DTCR_SS.y_test[:,ii], DTCR_SS.y_pred[:,ii])
    plt.plot(fpr, tpr, lw=2, label='Seq. %s (area = %0.4f)' % (c, roc_score),c=color_dict[c],linestyle='-.')

    roc_score = roc_auc_score(DTCR_WF.y_test[:,ii], DTCR_WF.y_pred[:,ii])
    fpr, tpr, _ = roc_curve(DTCR_WF.y_test[:,ii], DTCR_WF.y_pred[:,ii])
    plt.plot(fpr, tpr, lw=2, label='Rep. %s (area = %0.4f)' % (c, roc_score),c=color_dict[c])

plt.legend(loc="lower right")
plt.savefig('Rudqvist_AUC.eps')