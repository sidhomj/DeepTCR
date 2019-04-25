from DeepTCR.DeepTCR import DeepTCR_SS, DeepTCR_U
import numpy as np
import seaborn as sns

#Run Supervised Sequence Classifier
DTCRS = DeepTCR_SS('Sequence_C')
DTCRS.Get_Data(directory='../../Data/Murine_Antigens',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3)

DTCRS.Monte_Carlo_CrossVal(folds=25,stop_criterion=0.01)
# indices = DTCRS.beta_indices
# max_len = np.sum(DTCRS.X_Seq_beta>0,-1)
# ind_norm = np.ndarray.flatten(indices/max_len)
# sns.distplot(ind_norm)

#DTCRS.Representative_Sequences()
DTCRS.Representative_Sequences_Motif(10,unique=True)
antigens =['Db-F2', 'Db-M45', 'Db-NP', 'Db-PA', 'Db-PB1', 'Kb-M38', 'Kb-SIY',
       'Kb-TRP2', 'Kb-m139']

for a in antigens:
    DTCRS.Motif_Identification(group=a)