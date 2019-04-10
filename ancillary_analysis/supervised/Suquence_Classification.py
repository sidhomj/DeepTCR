from DeepTCR.DeepTCR import DeepTCR_SS
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

DTCRS = DeepTCR_SS('Sequence_C')

#Murine Antigens
DTCRS.Get_Data(directory='../Data/Murine_Antigens',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3)

#Human Antigens
DTCRS.Get_Data(directory='../Data/Human_Antigens',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3)

AUC = []
Method = []
Class = []
for i in range(10):
    DTCRS.Get_Train_Valid_Test()

    DTCRS.Train(use_only_seq=True)
    DTCRS.AUC_Curve(plot=False)
    AUC.extend(DTCRS.AUC_DF['AUC'].tolist())
    Method.extend(len(DTCRS.AUC_DF)*['Seq'])
    Class.extend(DTCRS.AUC_DF['Class'].tolist())

    DTCRS.Train(use_only_gene=True)
    DTCRS.AUC_Curve(plot=False)
    AUC.extend(DTCRS.AUC_DF['AUC'].tolist())
    Method.extend(len(DTCRS.AUC_DF)*['Gene'])
    Class.extend(DTCRS.AUC_DF['Class'].tolist())

    DTCRS.Train()
    DTCRS.AUC_Curve(plot=False)
    AUC.extend(DTCRS.AUC_DF['AUC'].tolist())
    Method.extend(len(DTCRS.AUC_DF)*['Seq-Gene'])
    Class.extend(DTCRS.AUC_DF['Class'].tolist())


df_comp = pd.DataFrame()
df_comp['AUC'] = AUC
df_comp['Method'] = Method
df_comp['Class'] = Class

order = ['Kb-SIY','Kb-TRP2','Kb-M38','Kb-m139','Db-F2','Db-M45','Db-NP','Db-PA','Db-PB1']
sns.violinplot(data=df_comp,x='Class',y='AUC',hue='Method',order=order)
sns.violinplot(data=df_comp,x='Class',y='AUC',hue='Method')


#Show where kernels are learning
DTCRS.Get_Train_Valid_Test()
DTCRS.Train(use_only_seq=True)

DTCRS.Train()

seq_len = np.sum(DTCRS.X_Seq_beta>0,-1)
loc = DTCRS.beta_indices/seq_len
sns.distplot(np.ndarray.flatten(loc))
plt.xlim([0,1])