import pandas as pd
import numpy as np
import os
from Bio.SubsMat import MatrixInfo
from Bio.Alphabet import IUPAC
from collections import OrderedDict
import numpy as np
import pandas as pd
import os
import pickle

def create_negative_samples(df,within_hla=False,max_iter=100,multiplier=10):
    """
    df: dataframe with CDR3, Antigen, HLA columns
    within_hla (bool): whether or not to shuffle within HLA subyptes or not.
    max_iter (int): maximum number of times to shuffle when creating negative samples
    multiplier (int): how many times one wants the negative outgroup over the positive samples
    """
    if within_hla is False:
        # shuffle across all HLA
        dfs = []
        for _ in range(max_iter):
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
            if len(dfs_temp) > multiplier * len(df):
                break
        dfs = pd.concat(dfs)
        dfs.drop_duplicates(inplace=True)
    else:
        # shuffle within HLA types
        hla_list = np.array(df['HLA'].value_counts().index)
        dfs = []
        for h in hla_list:
            df_sel = df[df['HLA'] == h]
            df_sel.reset_index(drop=True, inplace=True)
            dfs_temp = []
            for _ in range(max_iter):
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
                if len(dfs_temp2) > multiplier * len(df_sel):
                    break
            dfs.append(dfs_temp2)
        dfs = pd.concat(dfs)
        dfs.drop_duplicates(inplace=True)

    return dfs

def load_mhc_binding():
    filename = os.path.join('../../Data/synapse/iedb/mhc_ligand_full.csv')

    # load data
    df = load_data(filename)

    # Use only data from selected assay types
    list_assays = ['purified MHC/competitive/radioactivity',
                   'purified MHC/competitive/fluorescence',
                   'purified MHC/direct/fluorescence',
                   'purified MHC/direct/radioactivity']

    idx = df['MHC allele class'].isin(['I']) \
          & ~df['Allele Name'].str.contains('mutant') \
          & df['Method/Technique'].isin(list_assays) \
          & ~df['Quantitative measurement'].isnull()

    df = df[idx]

    idx = df['Allele Name'].str.contains('HLA') & ~df['Allele Name'].str.contains('*', regex=False)
    df = df[~idx]

    idx = df['Allele Name'].str.contains('SLA') & ~df['Allele Name'].str.contains('*', regex=False)
    df = df[~idx]

    list_species = ['HLA']
    idx = df['Allele Name'].str.contains('|'.join(list_species))
    df = df[idx]

    df_train = df.loc[idx, ['AA Sequence', 'Allele Name', 'Quantitative measurement']]
    df_train['Quantitative measurement'] = df_train['Quantitative measurement'].astype(float)
    Y = np.asarray(df_train['Quantitative measurement'].tolist())
    df_train['Quantitative measurement'][df_train['Quantitative measurement'] == 0] = np.min(Y[np.nonzero(Y)])
    df_train['Y'] = 1 - (np.log10(df_train['Quantitative measurement']) / np.log10(50000))

    # Get Kim Data
    df_kim1 = Get_Supplementary_Data()
    df_kim2 = Get_Kim_Data()
    df_kim = pd.concat((df_kim1, df_kim2))
    df_train = pd.concat([df_train, df_kim])

    # Aggregate Dataframe
    df_train = df_train.drop_duplicates()
    df_train = df_train.groupby(['Allele Name', 'AA Sequence']).agg({'Y': 'median'})
    df_train = df_train.reset_index()
    idx = df_train['Allele Name'].str.contains('HLA-(A|B|C)')
    df_train = df_train[idx]
    df_train['Allele Name'] = df_train['Allele Name'].str.replace('HLA-','')
    df_train['Allele Name'] = df_train['Allele Name'].str.replace('*','')
    df_train['Allele Name'] = df_train['Allele Name'].str.replace(':','')

    return df_train

    # Y = np.asarray(df_train['Y'].tolist())
    # # Apply Ramp
    # Y[Y > 1] = 1
    # Y[Y < 0] = 0
    # sequences = df_train['AA Sequence'].tolist()
    # hla = df_train['Allele Name'].tolist()
    #
    # counts = df_train['Allele Name'].value_counts().to_frame().reset_index()
    # counts = counts.rename(columns={'index': 'Allele Name', 'Allele Name': 'Counts'})
    # counts.sort_values(by='Allele Name', inplace=True)
    # counts.reset_index(inplace=True, drop=True)
    # df_allele_counts = counts
    # df_allele_counts.to_csv(os.path.join(self.directory_results, 'IC50_Allele_Counts.csv'), index=False)
    #
    # hla_num = self.lb_hla.transform(hla)
    # sequences_hla = []
    # for ii in hla:
    #     sequences_hla.append(self.DF_HLA.loc[ii].tolist()[0])
    #
    # args = list(zip(sequences, [self.aa_idx] * len(sequences), [self.max_length] * len(sequences)))
    # print('Embedding Sequences')
    # p = Pool(40)
    # result = p.starmap(Embed_Seq_Num, args)
    # p.close()
    # X_Seq = np.vstack(result).reshape(len(sequences), -1, self.max_length)
    # sequences = np.asarray(sequences)
    #
    # args = list(zip(sequences_hla, [self.aa_idx] * len(sequences_hla), [self.max_length_hla] * len(sequences_hla)))
    # print('Embedding Sequences')
    # p = Pool(40)
    # result = p.starmap(Embed_Seq_Num, args)
    # p.close()
    # X_HLA = np.vstack(result).reshape(len(sequences_hla), -1, self.max_length_hla)
    # sequences_hla = np.asarray(sequences_hla)
    #
    # with open(os.path.join(self.Name, 'IC50_Data.pkl'), 'wb') as f:
    #     pickle.dump([X_Seq, X_HLA, Y, hla_num, df_allele_counts, sequences, sequences_hla], f, protocol=4)


def load_data(filename):
    # set input file from IDB
    # set columns to load from input file
    load_cols = ['Object Type', 'Description', 'Antigen Name', 'Parent Protein', 'Parent Species',
                 'Method/Technique', 'Assay Group', 'Units', 'Quantitative measurement', 'Qualitative Measure',
                 'Allele Name', 'MHC allele class']
    # load into pd.DataFrame
    df = pd.read_csv(filename, sep=',', skiprows=1, usecols=load_cols, dtype=object)
    # parse out AA Sequence and potential modification from input 'Description' column
    df[['AA Sequence', 'AA Modifications']] = df['Description'].str.split('\s', n=1, expand=True)
    # validate AA Sequence against IUPAC
    idx = df['AA Sequence'].str.contains(r'^[%s]+$' % IUPAC.protein.letters)
    # filter out non-validated ones for now
    if (~idx).any():
        print('%d records will be filtered out due to non IUPAC protein letters.' % sum(~idx))

    return df.loc[idx, ]

def load_data_tcr(filename):
    # set input file from IDB
    # set columns to load from input file
    load_cols = ['Description', 'Qualitative Measure','Parent Species','Method/Technique','Cell Type','Host ID','Allele Name','Class']
    # load into pd.DataFrame
    df = pd.read_csv(filename, sep=',', usecols=load_cols, dtype=object)
    # parse out AA Sequence and potential modification from input 'Description' column
    #df[['AA Sequence', 'AA Modifications']] = df['Description'].str.split('\s', n=1, expand=True)
    df['AA Sequence'] = df['Description']
    # validate AA Sequence against IUPAC
    idx = df['AA Sequence'].str.contains(r'^[%s]+$' % IUPAC.protein.letters)
    # filter out non-validated ones for now
    if (~idx).any():
        print('%d records will be filtered out due to non IUPAC protein letters.' % sum(~idx))

    return df.loc[idx, ]

def Get_Kim_Data():
    # Get BD2013 data
    load_cols = ['mhc', 'sequence', 'meas']
    filenames = [os.path.join('../../Data/synapse/iedb/Blind_human.csv')]

    df_list = []
    for ii,file in enumerate(filenames,0):
        df = pd.read_csv(file, usecols=load_cols)
        #df['mhc'] = df['mhc'].str.replace('[*:-]', '')
        #df['mhc'] = df['mhc'].str[3:]
        df['Y'] = 1-(np.log10(df['meas'])/np.log10(50000))
        df = df.rename(columns={'mhc': 'Allele Name', 'sequence': 'AA Sequence', 'meas': 'Quantitative measurement'})
        df_list.append(df)
    df_kim = pd.concat(df_list, ignore_index=True)
    df_kim = df_kim.groupby(['Allele Name', 'AA Sequence']).agg({'Y': 'median'})
    df_kim = df_kim.reset_index()

    return df_kim

def Get_Supplementary_Data():
    filenames = [os.path.join('../../Data/synapse/iedb/BD2009_human.csv')]
    load_cols = ['mhc', 'sequence', 'meas']
    df_list = []
    for ii, file in enumerate(filenames, 0):
        df = pd.read_csv(file, usecols=load_cols)
        idx = df['mhc'].str.contains(r'.-.-')
        df = df[idx]
        df['mhc'] = df['mhc'].str.replace('-','*')
        df['mhc'] = df['mhc'].str.replace('*', '-',1)
        df['mhc'] = df['mhc'].str.slice(start=0, stop=8) + ':' + df['mhc'].str.slice(start=8)

        #df['mhc'] = df['mhc'].str.replace('[*:-]', '')
        #df['mhc'] = df['mhc'].str[3:]
        df['Y'] = 1 - (np.log10(df['meas']) / np.log10(50000))
        df = df.rename(columns={'mhc': 'Allele Name', 'sequence': 'AA Sequence', 'meas': 'Quantitative measurement'})
        df_list.append(df)
    df_kim = pd.concat(df_list, ignore_index=True)
    df_kim = df_kim.groupby(['Allele Name', 'AA Sequence']).agg({'Y': 'median'})
    df_kim = df_kim.reset_index()
    return df_kim

def load_foreign_v_self():
    filename = os.path.join('../../Data/synapse/iedb/mhc_ligand_full.csv')
    # load data
    df = load_data(filename)

    # Get Sequence & Parent Species
    df_train = df[['AA Sequence', 'Parent Species']]

    # Aggregate possible duplicates
    df_train = df_train.groupby(['AA Sequence']).agg({'Parent Species': 'first'})
    df_train = df_train.reset_index()

    # remove nan's from parent species
    idx = df_train['Parent Species'].isnull()
    df_train = df_train[~idx]

    # Assign self vs non-self
    df_train['Y'] = ''
    idx = df_train['Parent Species'].str.contains('Homo sapiens')
    df_train['Y'][idx] = 'self'
    df_train['Y'][~idx] = 'foreign'

    return df_train



