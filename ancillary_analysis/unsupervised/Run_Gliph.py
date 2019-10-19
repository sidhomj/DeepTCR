"""This script is used to benchmark GLIPH's performance across a variety of clustering thresholds by
varying the hamming distance parameter. This script required a local installation of GLIPH"""
import pandas as pd
import os
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

directory = '../../Data/Glanville/'
antigens = os.listdir(directory)

seq = []
label = []
for antigen in antigens:
    df = pd.read_csv(os.path.join(directory,antigen,antigen+'.tsv'),sep='\t')
    seq.extend(df['aminoAcid'].tolist())
    label.extend([antigen]*len(df))

df_out = pd.DataFrame()
df_out['Sequences'] = seq
df_out.to_csv('cdr3.txt',index=False)

df_ref = pd.DataFrame()
df_ref['Beta_Sequences'] = seq
df_ref['Labels'] = label
df_ref_dict = df_ref.set_index('Beta_Sequences').T.to_dict('list')

total_seq = len(seq)
num_clusters = []
variance = []
x=[]
y=[]
r = np.asarray(range(5))
for t in r:
    #Erase previous GLIPH outputs
    files = glob.glob('cdr3*')
    for file in files:
        if file != 'cdr3.txt':
            os.remove(file)

    #Run GLIPH
    os.system('gliph/bin/gliph-group-discovery.pl --tcr cdr3.txt --gccutoff='+str(t))
    df_in = pd.read_csv('cdr3-convergence-groups.txt',sep='\t',header=None)

    #Collect Clusters
    DFs = []
    for c in range(len(df_in)):
        df_temp = pd.DataFrame()
        seq = df_in[2][c]
        seq = seq.split()
        label = []
        for s in seq:
            label.append(df_ref_dict[s][0])
        df_temp['Beta_Sequences'] = seq
        df_temp['Labels'] = label
        DFs.append(df_temp)

    #Determine Specificity
    correct = 0
    clustered = 0
    df_clusters = []
    for df in DFs:
        if len(df) >= 3:
            common = df['Labels'].value_counts()
            if len(common) == 1:
                most_common = df['Labels'].value_counts().index[0]
                correct += np.sum(df['Labels'] == most_common)
                clustered += len(df)
                df_clusters.append(df)

            elif (common[0] > common[1]):
                most_common = df['Labels'].value_counts().index[0]
                correct += np.sum(df['Labels'] == most_common)
                clustered += len(df)
                df_clusters.append(df)


    x.append(clustered/total_seq)
    y.append(correct/clustered)


#Save Data
df_out = pd.DataFrame()
df_out['Percent Clustered'] = 100*np.asarray(x)
df_out['Percent Correctly Clustered'] = 100*np.asarray(y)
df_out.to_csv('GLIPH.csv',index=False)
plt.figure()
sns.regplot(data=df_out,x='Percent Clustered',y='Percent Correctly Clustered',fit_reg=False)








