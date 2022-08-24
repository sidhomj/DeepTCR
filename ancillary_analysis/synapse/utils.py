import pandas as pd
import numpy as np

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