import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from DeepTCR.DeepTCR import DeepTCR_U
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage,fcluster

#Instantiate training object
DTCRU = DeepTCR_U('Glanville_v_Hamming')
DTCRU.Get_Data(directory='../../Data/Glanville/',Load_Prev_Data=False,aa_column_beta=1,aggregate_by_aa=False)

method_dim = 'Hamming'
distances_hamming = pdist(np.squeeze(DTCRU.X_Seq_beta, 1), metric='hamming')

#Clustering Thresholds
r = np.logspace(np.log10(0.001), np.log10(10), 50)

#Collect data for plots
x = []
y = []
total_seq = len(DTCRU.X_Seq_beta)
for t in r:
    print(t)
    Z = linkage(distances_hamming, method='ward')
    IDX = fcluster(Z, t, criterion='distance')
    correct = 0
    clustered = 0

    for i in np.unique(IDX):
        sel = np.where(IDX==i)[0]
        df_temp = pd.DataFrame( DTCRU.class_id[sel])
        if len(sel) >= 3:
            common = df_temp[0].value_counts()
            if len(common) == 1:
                most_common = df_temp[0].value_counts().index[0]
                correct += np.sum(df_temp[0] == most_common)
                clustered += len(df_temp)
            elif common[0]>common[1]:
                most_common = df_temp[0].value_counts().index[0]
                correct += np.sum(df_temp[0] == most_common)
                clustered += len(df_temp)
    x.append(clustered / total_seq)
    y.append(correct / clustered)


#Save Data
df_out = pd.DataFrame()
df_out['Percent Clustered'] = 100*np.asarray(x)
df_out['Percent Correctly Clustered'] = 100*np.asarray(y)
df_out.to_csv(method_dim+'.csv',index=False)
df_gliph = pd.read_csv('GLIPH.csv')

with plt.style.context('ggplot'):
    fig, ax = plt.subplots()
    sns.lineplot(data=df_out,x='Percent Clustered',y='Percent Correctly Clustered',label='Hamming',ax=ax)
    sns.lineplot(data=df_gliph,x='Percent Clustered',y='Percent Correctly Clustered',label='GLIPH',ax=ax)
    plt.ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color([0.4, 0.4, 0.4])
    ax.spines['bottom'].set_color([0.4, 0.4, 0.4])
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    ax.set_facecolor('w')
    plt.legend()
    plt.savefig('GLIPH_v_Hamming.tif')



