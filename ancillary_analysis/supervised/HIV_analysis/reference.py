import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact

with open('screen.pkl','rb') as f:
    [df_auc,sequences,predicted] = pickle.load(f)

pw = pd.read_csv('hiv_ref_pw.csv')
preds = []
for row in pw.iterrows():
    tcr = row[1]['TCR']
    pep = row[1]['Peptide']
    u = np.where(df_auc['epitope']==pep)[0][0]
    preds.append(np.mean(predicted[u][np.where(np.isin(sequences[u],tcr))[0]]))

pw['preds'] = preds
pw['preds'] = pw['preds'].round(3)
pw.to_csv('val_preds.csv',index=False)

all_pred = np.hstack(predicted)
df_plot = pd.DataFrame()
df_plot['preds'] = np.hstack([preds,all_pred])
df_plot['label'] = np.hstack([['validated+']*len(preds),['background']*len(all_pred)])
sns.violinplot(data=df_plot[df_plot['label']=='background'],x='label',y='preds',cut=0)
sns.swarmplot(data=df_plot[df_plot['label']=='validated+'],x='label',y='preds',color='red',size=8,alpha=0.75)
plt.xlabel('')
plt.ylabel('Prediction Value',fontsize=16)
plt.xticks([])
plt.yticks()
plt.show()
plt.savefig('val_dist.png')

thresh = 0.95
x = np.array([[np.sum(all_pred < thresh)-1,np.sum(all_pred >= thresh)-17],
     [1,17]])

_,p_val = fisher_exact(x)
enrichment = (x[1,1]/np.sum(x[:,1]))/(np.sum(x[1,:])/np.sum(x))



