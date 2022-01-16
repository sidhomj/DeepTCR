from DeepTCR.DeepTCR import DeepTCR_WF
import numpy as np

DTCR = DeepTCR_WF('murine')
DTCR.Get_Data(directory='../../Data/Rudqvist',Load_Prev_Data=True,
              aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21)
              # type_of_data_cut='Num_Seq',
              # data_cut=100)
folds=10
graph_seed=0
seeds= np.array(range(folds))
DTCR.Monte_Carlo_CrossVal(folds=folds,combine_train_valid=True,train_loss_min=0.5,
                          seeds=seeds,graph_seed=graph_seed)
DTCR.AUC_Curve()