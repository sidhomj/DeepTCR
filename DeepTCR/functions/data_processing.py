from Bio.SubsMat import MatrixInfo
from Bio.Alphabet import IUPAC
from collections import OrderedDict
import numpy as np
import pandas as pd
import re
import os
import pickle

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

    with pd.option_context('mode.chained_assignment', None):

        df = pd.read_csv(file, sep=sep,dtype=object)
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
        df.dropna(subset=['counts'],inplace=True)
        df['counts']=df['counts'].astype(int)
        df= df[df['counts'] >= 1]

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
        df_temp = Cut_DF(df=df, type_cut=type_of_data_cut, cut=data_cut)
        if len(df_temp)==0:
            df = df.iloc[0].to_frame().T
        else:
            df = df_temp

    return df

def supertype_conv_op(hla,keep_non_supertype_alleles=False):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df_supertypes = pd.read_csv(os.path.join(dir_path,'Supertype_Data_Dict.csv'))
    df_supertypes = df_supertypes[~df_supertypes['Supertype_2'].isin(['AU', 'BU'])]
    hla_dict = dict(zip(df_supertypes['Allele'], df_supertypes['Supertype_2']))
    hla_list_sup = []
    for h in hla:
        if not keep_non_supertype_alleles:
            h = [x for x in h if x in hla_dict.keys()]
        hla_list_sup.append(np.array([hla_dict[x] if x in hla_dict.keys() else x for x in h]))
    return hla_list_sup

def supertype_conv(df,keep_non_supertype_alleles=False):
    hla = np.array(df.iloc[:,1:])
    hla_list_sup = supertype_conv_op(hla,keep_non_supertype_alleles)
    hla_sup = pd.DataFrame()
    colname = df.columns[0]
    hla_sup[colname] = df[colname]
    hla_sup = pd.concat([hla_sup, pd.DataFrame(hla_list_sup)], axis=1)
    return hla_sup

def save_model_data(self,saver,sess,name,get,iteration=0):
    saver.save(sess, os.path.join(self.Name, 'models', 'model_' + str(iteration), 'model.ckpt'))
    with open(os.path.join(self.Name, 'models', 'model_type.pkl'), 'wb') as f:
        pickle.dump([name, get.name, self.use_alpha, self.use_beta,
                     self.use_v_beta, self.use_d_beta, self.use_j_beta,
                     self.use_v_alpha, self.use_j_alpha, self.use_hla, self.use_hla_sup, self.keep_non_supertype_alleles,
                     self.lb_v_beta, self.lb_d_beta, self.lb_j_beta,
                     self.lb_v_alpha, self.lb_j_alpha, self.lb_hla, self.lb,
                     self.ind,self.regression], f)

def load_model_data(self):
    with open(os.path.join(self.Name, 'models', 'model_type.pkl'), 'rb') as f:
        model_type, get, self.use_alpha, self.use_beta, \
        self.use_v_beta, self.use_d_beta, self.use_j_beta, \
        self.use_v_alpha, self.use_j_alpha, self.use_hla, self.use_hla_sup,self.keep_non_supertype_alleles, \
        self.lb_v_beta, self.lb_d_beta, self.lb_j_beta, \
        self.lb_v_alpha, self.lb_j_alpha, self.lb_hla, self.lb,\
            self.ind,self.regression = pickle.load(f)
    return model_type,get

def make_seq_list(seq,
                  ref=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                       'Y']):
    seq_run = list(seq)
    seq_run_list = []
    pos = []
    ref_list = []
    alt_list = []
    for ii, c in enumerate(seq_run, 0):
        seq_run_temp = seq_run.copy()
        for r in ref:
            seq_run_temp[ii] = r
            seq_run_list.append(''.join(seq_run_temp))
            pos.append(ii)
            ref_list.append(seq_run[ii])
            alt_list.append(r)
    return (seq_run_list, pos, ref_list, alt_list)

