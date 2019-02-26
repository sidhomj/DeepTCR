import pandas as pd
import numpy as np
import re

def Parse_DF(file):
    df = pd.read_csv(file,sep='\t')
    df = df.ix[:, 1:30]
    int_df = df.select_dtypes(include=['int64'])
    templates_column = int_df.columns[0]
    amino_column = int(np.where([re.search('Acid', f, flags=re.IGNORECASE) != None for f in df.columns.tolist()])[0])
    df = df.dropna(subset=[df.columns[amino_column]])
    searchfor = ['\*', 'X', 'O']
    df = df[~df.iloc[:, amino_column].str.contains('|'.join(searchfor))]
    idx_templates = np.where(df.columns == templates_column)[0]
    idx_sel = idx_templates

    df = df.sort_values(df.columns[idx_templates[0]], ascending=False)
    df1 = pd.DataFrame()
    df1['aminoAcid'] = df[df.columns[0]]
    df1['counts'] = df[df.columns[idx_templates[0]]]
    #df1['J'] = df['jGeneName']
    df1['Freq'] = df1['counts']/np.sum(df1['counts'])
    df1.reset_index(inplace=True,drop=True)
    return df1