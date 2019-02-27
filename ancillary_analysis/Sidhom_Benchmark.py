from DeepTCR.DeepTCR_U import DeepTCR_U
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Instantiate training object
DTCRU = DeepTCR_U('Dash')
# DTCRU.Get_Data(directory='../Data/Dash/Traditional/Human',Load_Prev_Data=False,aggregate_by_aa=False,
#                aa_column_alpha=0,aa_column_beta=1,count_column=2,v_alpha_column=3,j_alpha_column=4,v_beta_column=5,j_beta_column=6)

DTCRU.Get_Data(directory='../Data/Sidhom',Load_Prev_Data=False,aggregate_by_aa=False,aa_column_beta=None,count_column=None,
               v_beta_column=7,d_beta_column=14,j_beta_column=21)

#Choose Method to Analyze
method_dim = 'VAE' #Set to 'VAE' or 'GAN'

#Get Feature from VAE/GAN
if method_dim is 'GAN':
    DTCRU.Train_GAN(Load_Prev_Data=False,latent_dim=256,ortho_norm=False,use_distances=False)
else:
    DTCRU.Train_VAE(accuracy_min=0.9,Load_Prev_Data=True)

#Collect data for plots
x = []
y = []

num_steps = 10
max = 30
min=0.001

if method_dim is 'GAN':
    #GAN
    r = np.logspace(np.log10(min),np.log10(max),num_steps)
else:
    # VAE
    r = np.logspace(np.log10(min), np.log10(max), num_steps)

total_seq = len(DTCRU.X_Seq_alpha)
df_look = []
num_clusters = []
variance = []
for t in r:
    print(t)
    DTCRU.Cluster(t=t, criterion='distance',on=method_dim)
    correct = 0
    clustered = 0
    df_clusters = []
    for df in DTCRU.DFs:
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

    num_clusters.append(len(DTCRU.DFs))
    variance.append(np.var(DTCRU.var_beta))

    x.append(clustered/total_seq)
    y.append(correct/clustered)

#Save Data
df_out = pd.DataFrame()
df_out['Percent Clustered'] = 100*np.asarray(x)
df_out['Percent Correctly Clustered'] = 100*np.asarray(y)
df_out['Number of Clusters'] = num_clusters
df_out['Length Variance of Clusters'] = variance
df_out.to_csv(method_dim+'_Sidhom_Genes.csv',index=False)

#Plot Performance
sns.regplot(data=df_out,x='Percent Clustered',y='Percent Correctly Clustered',fit_reg=False)

#Compare VAE/GAN Performance and Clustering Characteristics

#Performance Metrics for Both Methods
df_gan = pd.read_csv('GAN_Sidhom.csv')
df_vae = pd.read_csv('VAE_Sidhom.csv')
plt.figure()
sns.set(font_scale=1.0)
sns.lineplot(data=df_gan,x='Percent Clustered',y='Percent Correctly Clustered',label='GAN')
sns.lineplot(data=df_vae,x='Percent Clustered',y='Percent Correctly Clustered',label = 'VAE')
plt.ylim(0,100)
plt.legend()
plt.savefig('Specificity_Clustering_Sidhom.tif')

#Cluster Characteristics for Both Methods
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(15,10))
sns.set(font_scale=1.5)
sns.lineplot(data=df_gan,x='Percent Clustered',y='Number of Clusters',label='GAN',ax=ax1)
sns.lineplot(data=df_vae,x='Percent Clustered',y='Number of Clusters',label='VAE',ax=ax1)
sns.lineplot(data=df_gan,x='Percent Clustered',y='Length Variance of Clusters',label='GAN',ax=ax2)
sns.lineplot(data=df_vae,x='Percent Clustered',y='Length Variance of Clusters',label='VAE',ax=ax2)
ax1.set_xlabel('')
plt.savefig('Cluster_Characteristics_Sidhom.tif')


