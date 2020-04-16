import pandas as pd

df_vdj = pd.read_csv('VDJ.tsv',sep='\t')
df_mcpas = pd.read_csv('McPAS-TCR.csv')
check=1

df_vdj_beta = df_vdj[df_vdj['Gene'] == 'TRB']
df_vdj_sel = pd.DataFrame()
df_vdj_sel['epitope'] = df_vdj_beta['Epitope']
df_vdj_sel['cdr3'] = df_vdj_beta['CDR3']

df_mcpas_beta = df_mcpas.dropna(subset=['CDR3.beta.aa'])
df_mcpas_sel = pd.DataFrame()
df_mcpas_sel['epitope'] = df_mcpas_beta['Epitope.peptide']
df_mcpas_sel['cdr3'] = df_mcpas_beta['CDR3.beta.aa']

df_out = pd.concat([df_vdj_sel,df_mcpas_sel])
df_out.drop_duplicates(inplace=True)
df_out.to_csv('vdj_mcpas.csv',index=False)

sel = 'ISPRTLNAW'
sel = df_vdj[df_vdj['Epitope']==sel]