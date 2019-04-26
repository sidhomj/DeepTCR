from DeepTCR.DeepTCR import DeepTCR_WF
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

#Train Sequence Classifier
DTCR = DeepTCR_WF('Human_TIL',device='/gpu:2')
dir = 'Topalian/beta/pre_crpr_sdpd'
DTCR.Get_Data(directory='../../Data/Topalian',Load_Prev_Data=False,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=0.25,
              hla='../../Data/Topalian/HLA_Ref.csv')

folds = 100
LOO = 5
epochs_min = 50
weight_by_class = True
size_of_net = 'small'

y_pred_list = []
y_test_list = []
names = ['Seq','HLA','Seq+HLA']

#Just train w/ Sequence Information
DTCR.use_hla = False
DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,
                          use_only_hla=False,weight_by_class=weight_by_class,size_of_net=size_of_net)
y_pred_list.append(DTCR.y_pred)
y_test_list.append(DTCR.y_test)

#Just train w/HLA
DTCR.use_hla = True
DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,
                          use_only_hla=True,weight_by_class=weight_by_class,size_of_net=size_of_net)
y_pred_list.append(DTCR.y_pred)
y_test_list.append(DTCR.y_test)

#Train with both Seq + HLA
DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,
                          use_only_hla=False,weight_by_class=weight_by_class,size_of_net=size_of_net)
y_pred_list.append(DTCR.y_pred)
y_test_list.append(DTCR.y_test)

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
plt.savefig('Human_TIL_AUC.eps')
