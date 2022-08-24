import pandas as pd
import numpy as np
from DeepTCR.DeepSynapse import DeepSynapse
from DeepTCR.functions_syn.data_processing import Process_Seq, supertype_conv_op

df = pd.read_csv('../../Data/synapse/testing_data.csv')
df['bind'] = True
df['id'] = df['CDR3']+'_'+df['Antigen']+'_'+df['HLA']
hla_list = np.array(df['HLA'].value_counts().index)

# shuffle across all HLA
dfs = []
n_count = 0
for _ in range(100):
    df_shuffle = pd.DataFrame()
    df_shuffle['CDR3'] = np.random.choice(df['CDR3'],size=len(df),replace=False)
    # df_shuffle['CDR3'] = np.random.choice(bg['x'],size=len(df),replace=True)
    df_shuffle['Antigen'] = df['Antigen']
    df_shuffle['HLA'] = df['HLA']
    df_shuffle['id'] = df_shuffle['CDR3']+'_'+df_shuffle['Antigen']+'_'+df_shuffle['HLA']
    df_shuffle['bind'] = df_shuffle['id'].isin(df['id'])
    df_shuffle = df_shuffle[df_shuffle['bind'] != True]
    dfs.append(df_shuffle)
    dfs_temp = pd.concat(dfs)
    dfs_temp.drop_duplicates(inplace=True)
    if len(dfs_temp) > 10 * len(df):
        break
dfs = pd.concat(dfs)
dfs.drop_duplicates(inplace=True)

# #shuffle within HLA types
# dfs = []
# for h in hla_list:
#     df_sel = df[df['HLA']==h]
#     df_sel.reset_index(drop=True,inplace=True)
#     n_count = 0
#     for _ in range(100):
#         df_shuffle = pd.DataFrame()
#         df_shuffle['CDR3'] = np.random.choice(df_sel['CDR3'], size=len(df_sel), replace=False)
#         df_shuffle['Antigen'] = df_sel['Antigen']
#         df_shuffle['HLA'] = df_sel['HLA']
#         df_shuffle['id'] = df_shuffle['CDR3'] + '_' + df_shuffle['Antigen'] + '_' + df_shuffle['HLA']
#         df_shuffle['bind'] = df_shuffle['id'].isin(df_sel['id'])
#         df_shuffle = df_shuffle[df_shuffle['bind'] != True]
#         dfs.append(df_shuffle)
#         n_count += len(df_shuffle)
#         if n_count > 10*len(df_sel):
#             break
# dfs = pd.concat(dfs)
# dfs.drop_duplicates(inplace=True)

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

check=1
