import pandas as pd
import numpy as np
from DeepTCR.DeepSynapse import DeepSynapse
from DeepTCR.functions_syn.data_processing import Process_Seq, supertype_conv_op
import utils

df = pd.read_csv('../../Data/synapse/training_data.csv')
df['bind'] = True
df['id'] = df['CDR3']+'_'+df['Antigen']+'_'+df['HLA']
# df = df.sample(n=1000,replace=False)
# bg = pd.read_csv('library/bg_tcr_library/TCR_10k_bg_seq.csv')
dfs = utils.create_negative_samples(df,within_hla=False,multiplier=1)

#add labels
df_input  = pd.concat([df,dfs])
df_input['bind'] = df_input['bind'].astype(int)
df_input['bind_cat'] = None
df_input['bind_cat'][df_input['bind']==1] = 'bind'
df_input['bind_cat'][df_input['bind']!=1] = 'non-bind'

#process seq & hla
df_input = Process_Seq(df_input,'CDR3')
df_input = Process_Seq(df_input,'Antigen')
df_input['HLA'] = df_input['HLA'].str.replace('*',"")
df_input['HLA'] = df_input['HLA'].str.replace(':',"")
df_input['HLA'] = df_input['HLA'].str[0:5]

#convert supertypes to alleles
hla_supertype = pd.read_csv('../../DeepTCR/library/Supertype_Data_Dict.csv')
hla_supertype = hla_supertype[['Supertype_2','Allele']]
hla_supertype.drop_duplicates(inplace=True,subset=['Supertype_2'])
hla_dict = dict(zip(hla_supertype['Supertype_2'],hla_supertype['Allele']))
df_input['HLA'] = df_input['HLA'].map(hla_dict).fillna(df_input['HLA'])
df_input = df_input[df_input['HLA'].str.len()==5]

df_input['HLA_sup'] = supertype_conv_op(df_input['HLA'],keep_non_supertype_alleles=True)

DTCR = DeepSynapse('epitope_tcr')
DTCR.Load_Data(beta_sequences=np.array(df_input['CDR3']),
               epitope_sequences = np.array(df_input['Antigen']),
               # hla=np.array(df_input['HLA_sup']),
                class_labels= np.array(df_input['bind_cat']),
               # use_hla_seq=False
               )

DTCR.Monte_Carlo_CrossVal(folds=1,batch_size=50000,epochs_min=50,
                          num_fc_layers=3,units_fc=256,
                          units_hla=[12,12,12],kernel_hla=[30,30,30],stride_hla=[5,5,5])