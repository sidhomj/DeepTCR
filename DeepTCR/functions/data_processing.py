from Bio.SubsMat import MatrixInfo
from Bio.Alphabet import IUPAC
from collections import OrderedDict
import numpy as np
import pandas as pd
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
            return df.iloc[np.where(np.cumsum(df['Frequency'])<=cut)]
    elif type_cut == 'Frequency_Cut':
        return df.iloc[np.where(df['Frequency']>=cut)[0]]

    elif type_cut == 'Num_Seq':
        return df.iloc[0:cut]

    elif type_cut == 'Read_Cut':
        return df.iloc[np.where(df['counts']>=cut)[0]]

    elif type_cut == 'Read_Sum':
        return df.iloc[np.where(np.cumsum(df['counts']) <= (cut))[0]]

def Process_Seq(df,col):
    #Drop null values
    df = df.dropna(subset=[col])

    #strip any white space and remove non-IUPAC characters
    df[col] = df[col].str.strip()
    searchfor = ['\*', 'X', 'O']
    df = df[~df[col].str.contains('|'.join(searchfor))]

    return df

def Get_DF_Data(file,type_of_data_cut='Fraction_Response',data_cut = 1.0,aa_column_alpha=None,aa_column_beta=None,
                count_column=None,sep='\t',max_length=40,aggregate_by_aa=True,v_beta_column=None,
                d_beta_column=None,j_beta_columns=None,
                v_alpha_column=None,j_alpha_column=None):

    df = pd.read_csv(file, sep=sep)

    #First collect columns in dataframe based on user preferences
    cols_to_keep = []
    column_names = []
    sequence_columns = []
    if aa_column_alpha is not None:
        cols_to_keep.append(aa_column_alpha)
        column_names.append('alpha')
        sequence_columns.append('alpha')
    if aa_column_beta is not None:
        cols_to_keep.append(aa_column_beta)
        column_names.append('beta')
        sequence_columns.append('beta')

    if count_column is not None:
        cols_to_keep.append(count_column)
        column_names.append('counts')
    else:
        df['counts'] = 1
        cols_to_keep.append(np.where(df.columns=='counts')[0][0])
        column_names.append('counts')

    if v_alpha_column is not None:
        cols_to_keep.append(v_alpha_column)
        column_names.append('v_alpha')

    if j_alpha_column is not None:
        cols_to_keep.append(j_alpha_column)
        column_names.append('j_alpha')

    if v_beta_column is not None:
        cols_to_keep.append(v_beta_column)
        column_names.append('v_beta')

    if d_beta_column is not None:
        cols_to_keep.append(d_beta_column)
        column_names.append('d_beta')

    if j_beta_columns is not None:
        cols_to_keep.append(j_beta_columns)
        column_names.append('j_beta')


    df = df.iloc[:,cols_to_keep]

    df.columns = column_names

    #Process Sequences
    if aa_column_alpha is not None:
        df = Process_Seq(df,'alpha')

    if aa_column_beta is not None:
        df = Process_Seq(df,'beta')

    #Aggregate or not by unique NT
    if aggregate_by_aa is True:
        agg_dict = {}
        agg_dict['counts']='sum'
        for col in df.columns:
            if (not col in sequence_columns) and (col != 'counts'):
                agg_dict[col] = 'first'

        df = df.groupby(sequence_columns).agg(agg_dict)
        df = df.sort_values(['counts'], ascending=False)
        df.reset_index(inplace=True)

    #Remove sequences with greater than max_length
    for seq in sequence_columns:
        df = df[df[seq].str.len() <= max_length]

    #Compute frequencies
    df['Frequency'] = df['counts'] / np.sum(df['counts'])

    #Threshold
    df = Cut_DF(df=df, type_cut=type_of_data_cut, cut=data_cut)

    return df
