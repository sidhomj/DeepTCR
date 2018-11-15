from Bio.SubsMat import MatrixInfo
from Bio.Alphabet import IUPAC
from collections import OrderedDict
import numpy as np
import pandas as pd
import glob
import re


def Embed_Seq_Num(seq,aa_idx,maxlength):
    seq_embed = np.zeros((1, maxlength)).astype('int64')
    if seq != 'null':
        n = 0
        for c in seq:
            seq_embed[0,n] = aa_idx[c]
            n += 1
    return seq_embed

def make_aa_df():
    # load matrix as symmetric pandas dataframe
    aa_df = pd.Series(MatrixInfo.blosum100).unstack()
    aa_df.where(~aa_df.isnull(), other=aa_df.T, inplace=True)

    # only keep IUPAC protein letters
    aa_keep = list(IUPAC.IUPACProtein.letters)
    aa_df = aa_df.loc[aa_keep, aa_keep]

    # add NULL '-' with max loss (min value) & median for match
    aa_df.loc['-', :] = aa_df.values.min()
    aa_df.loc[:, '-'] = aa_df.values.min()
    aa_df.loc['-', '-'] = np.median(aa_df.values.diagonal())

    # checks that we have square matrix with same elements
    if (aa_df.columns == aa_df.index).all():
        return OrderedDict(tuple(zip(list(aa_df.index.values), list(range(1, len(aa_df.index.values)))))), aa_df.values
    else:
        return None, None

def Cut_DF(df,type_cut,cut):
    if type_cut == 'Fraction_Response':
        if cut == 1.0:
            return df
        else:
            return df.iloc[np.where(np.cumsum(df.ix[:,2])<=(cut))[0]]
    elif type_cut == 'Frequency_Cut':
        return df.iloc[np.where(df.ix[:,2]>=cut)[0]]

    elif type_cut == 'Num_Seq':
        return df.iloc[0:cut]

    elif type_cut == 'Read_Cut':
        return df.iloc[np.where(df.ix[:,1]>=cut)[0]]

    elif type_cut == 'Read_Sum':
        return df.iloc[np.where(np.cumsum(df.ix[:, 1]) <= (cut))[0]]

def Get_DF_Data(file,type_of_data_cut='Fraction_Response',data_cut = 1.0,aa_column=None,count_column=None,sep='\t',max_length=40,aggregate_by_aa=True):
    df = pd.read_csv(file, sep=sep)
    df = df.ix[:, 0:10]
    if aa_column is None:
        amino_column = int(np.where([re.search('Acid', f, flags=re.IGNORECASE) != None for f in df.columns.tolist()])[0])
    else:
        amino_column = aa_column

    if count_column is None:
        int_df = df.select_dtypes(include=['int64'])
        templates_column = int_df.columns[0]
    else:
        templates_column = df.columns[count_column]

    df = df.dropna(subset=[df.columns[amino_column]])
    df[df.columns[amino_column]] = df[df.columns[amino_column]].str.strip()
    searchfor = ['\*', 'X','O']
    df = df[~df.iloc[:, amino_column].str.contains('|'.join(searchfor))]

    idx_templates = np.where(df.columns == templates_column)[0]
    idx_sel = idx_templates

    if aggregate_by_aa is True:
        # Combine sequences that share nucleotide sequence
        df = df.groupby(df.columns[amino_column]).agg({df.columns[idx_sel[0]]: 'sum'})
        df = df.sort_values(df.columns[0], ascending=False)
        df1 = pd.DataFrame()
        df1['aminoAcid'] = df.index
        df = df.reset_index(drop=True)
        df1['counts'] = df.iloc[:, 0]
        df = df1
    else:
        df1 = pd.DataFrame()
        df1['aminoAcid'] = df['aminoAcid']
        df1['counts'] = df[df.columns[idx_sel[0]]]
        df = df1

    df = df[df['aminoAcid'].str.len() <= max_length]

    df['Frequency'] = df['counts'] / np.sum(df['counts'])
    df = Cut_DF(df=df, type_cut=type_of_data_cut, cut=data_cut)

    return df

