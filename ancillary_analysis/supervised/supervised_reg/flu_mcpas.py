import pandas as pd
from DeepTCR.DeepTCR import DeepTCR_SS
import numpy as np

epitope = 'GILGFVFTL'
cdr3_beta_col = 'CDR3.beta.aa'
cdr3_alpha_col = 'CDR3.alpha.aa'
epitope_col = 'Epitope.peptide'

df = pd.read_csv('../../../Data/10x_Data/Data_Regression.csv')
DTCRS = DeepTCR_SS('reg_flu',device=2)

#Check performance no sequences in MCPAS
df_train_pep = pd.DataFrame()
df_train_pep['alpha'] = np.asarray(df['alpha'].tolist())
df_train_pep['beta'] = np.asarray(df['beta'].tolist())
df_train_pep['seq_id'] = df_train_pep['alpha'] + '_' + df_train_pep['beta']

df_tcr = pd.read_csv('../../../Data/McPAS-TCR.csv')
df_tcr.dropna(subset=[cdr3_alpha_col,cdr3_beta_col],inplace=True)
df_tcr = df_tcr.groupby([cdr3_alpha_col,cdr3_beta_col]).agg({epitope_col:'first'}).reset_index()
df_tcr['seq_id'] = df_tcr[cdr3_alpha_col] + '_' + df_tcr[cdr3_beta_col]
df_tcr = df_tcr[~df_tcr['seq_id'].isin(df_train_pep['seq_id'])]
remove = ["""[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ 1234567890]"""]
df_tcr = df_tcr[~df_tcr[cdr3_alpha_col].str.contains('|'.join(remove),regex=True)]
df_tcr = df_tcr[~df_tcr[cdr3_beta_col].str.contains('|'.join(remove),regex=True)]
remove = ['0','\?','O','9','\*','B','X']
df_tcr = df_tcr[~df_tcr[cdr3_alpha_col].str.contains('|'.join(remove),regex=True)]
df_tcr = df_tcr[~df_tcr[cdr3_beta_col].str.contains('|'.join(remove),regex=True)]
df_tcr[cdr3_alpha_col] =  df_tcr[cdr3_alpha_col].str.replace('[^\x00-\x7F]','')
df_tcr[cdr3_beta_col] =  df_tcr[cdr3_beta_col].str.replace('[^\x00-\x7F]','')

temp = df_tcr[df_tcr[epitope_col]==epitope]
temp = temp.groupby([cdr3_alpha_col,cdr3_beta_col]).agg({epitope_col:'first'}).reset_index()
temp = temp[~temp['CDR3.alpha.aa'].str.contains('#')]
temp['seq_id'] = temp[cdr3_alpha_col] + '_' + temp[cdr3_beta_col]
temp = temp[~temp['seq_id'].isin(df_train_pep['seq_id'])]
out = DTCRS.Sequence_Inference(beta_sequences=np.array(temp[cdr3_beta_col]),alpha_sequences=np.array(temp[cdr3_alpha_col]))
df_true = pd.DataFrame()
df_true['pred'] = np.squeeze(out)
df_true['label'] = 1.0

temp = df_tcr[df_tcr[epitope_col]!=epitope]
out = DTCRS.Sequence_Inference(beta_sequences=np.array(temp[cdr3_beta_col]),alpha_sequences=np.array(temp[cdr3_alpha_col]))
df_false = pd.DataFrame()
df_false['pred'] = np.squeeze(out)
df_false['label'] = 0.0
df_preds = pd.concat([df_true,df_false])

df_preds.to_csv('flu_mcpas_val.csv',index=False)