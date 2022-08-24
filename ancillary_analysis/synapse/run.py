import pandas as pd
import numpy as np
from DeepTCR.DeepSynapse import DeepSynapse
from DeepTCR.functions_syn.data_processing import Process_Seq, supertype_conv_op

df = pd.read_csv('../../Data/synapse/training_data.csv')
df['bind'] = True
df['id'] = df['CDR3']+'_'+df['Antigen']+'_'+df['HLA']
# df = df.sample(n=5000,replace=False)
hla_list = np.array(df['HLA'].value_counts().index)

# bg = pd.read_csv('library/bg_tcr_library/TCR_10k_bg_seq.csv')

# # shuffle across all HLA
# dfs = []
# for _ in range(100):
#     df_shuffle = pd.DataFrame()
#     df_shuffle['CDR3'] = np.random.choice(df['CDR3'],size=len(df),replace=False)
#     # df_shuffle['CDR3'] = np.random.choice(bg['x'],size=len(df),replace=True)
#     df_shuffle['Antigen'] = df['Antigen']
#     df_shuffle['HLA'] = df['HLA']
#     df_shuffle['id'] = df_shuffle['CDR3']+'_'+df_shuffle['Antigen']+'_'+df_shuffle['HLA']
#     df_shuffle['bind'] = df_shuffle['id'].isin(df['id'])
#     df_shuffle = df_shuffle[df_shuffle['bind'] != True]
#     dfs.append(df_shuffle)
#     dfs_temp = pd.concat(dfs)
#     dfs_temp.drop_duplicates(inplace=True)
#     if len(dfs_temp) > 10 * len(df):
#         break
# dfs = pd.concat(dfs)
# dfs.drop_duplicates(inplace=True)

#shuffle within HLA types
dfs = []
for h in hla_list:
    df_sel = df[df['HLA']==h]
    df_sel.reset_index(drop=True,inplace=True)
    dfs_temp = []
    for _ in range(100):
        df_shuffle = pd.DataFrame()
        df_shuffle['CDR3'] = np.random.choice(df_sel['CDR3'], size=len(df_sel), replace=False)
        df_shuffle['Antigen'] = df_sel['Antigen']
        df_shuffle['HLA'] = df_sel['HLA']
        df_shuffle['id'] = df_shuffle['CDR3'] + '_' + df_shuffle['Antigen'] + '_' + df_shuffle['HLA']
        df_shuffle['bind'] = df_shuffle['id'].isin(df_sel['id'])
        df_shuffle = df_shuffle[df_shuffle['bind'] != True]
        dfs_temp.append(df_shuffle)
        dfs_temp2 = pd.concat(dfs_temp)
        dfs_temp2.drop_duplicates(inplace=True)
        if len(dfs_temp2) > 10 * len(df_sel):
            break
    dfs.append(dfs_temp2)
dfs = pd.concat(dfs)
dfs.drop_duplicates(inplace=True)


df_train  = pd.concat([df,dfs])
df_train['bind'] = df_train['bind'].astype(int)
df_train['bind_cat'] = None
df_train['bind_cat'][df_train['bind']==1] = 'bind'
df_train['bind_cat'][df_train['bind']!=1] = 'non-bind'

df_train = Process_Seq(df_train,'CDR3')
df_train = Process_Seq(df_train,'Antigen')
df_train['HLA'] = df_train['HLA'].str.replace('*',"")
df_train['HLA'] = df_train['HLA'].str.replace(':',"")
df_train['HLA'] = df_train['HLA'].str[0:5]
df_train = df_train[df_train['HLA'].str.len()==5]
df_train['HLA_sup'] = supertype_conv_op(df_train['HLA'],keep_non_supertype_alleles=True)
# df_train.drop_duplicates(inplace=True,subset=['HLA','HLA_sup'])
# df_train = df_train[df_train['HLA_sup'].isin(['A02','B07'])]
# df_train = df_train[df_train['HLA'].isin(['A0301','A0201'])]

DTCR = DeepSynapse('epitope_tcr')
DTCR.Load_Data(beta_sequences=np.array(df_train['CDR3']),
               epitope_sequences = np.array(df_train['Antigen']),
               hla=np.array(df_train['HLA']),
                class_labels= np.array(df_train['bind_cat']),
               use_hla_seq=True)
DTCR.Monte_Carlo_CrossVal(folds=1,batch_size=50000,epochs_min=50,
                          num_fc_layers=3,units_fc=256,
                          units_hla=[12,12,12],kernel_hla=[30,30,30],stride_hla=[5,5,5],padding_hla='same')

DTCR.Representative_Sequences(top_seq=50,make_seq_logos=False)
class_sel = 'bind'
DTCR.Residue_Sensitivity_Logo(beta_sequences=np.array(DTCR.Rep_Seq[class_sel]['beta'])[0:25],
                              alpha_sequences=np.array(DTCR.Rep_Seq[class_sel]['alpha'])[0:25],
                              hla=np.array(DTCR.Rep_Seq[class_sel]['HLA'])[0:25],
                              class_sel = class_sel)

df_preds = pd.DataFrame()
df_preds['CDR3'] = DTCR.beta_sequences
df_preds['Antigen'] = DTCR.alpha_sequences
df_preds['HLA'] = DTCR.hla_data_seq
df_preds['bind'] = DTCR.class_id
df_preds['pred'] = DTCR.predicted[:,0]