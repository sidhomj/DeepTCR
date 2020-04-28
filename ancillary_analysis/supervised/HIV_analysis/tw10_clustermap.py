import numpy as np
import pickle
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file = 'aucs_gagtw10.pkl'
with open(file,'rb') as f:
    aucs,preds,group = pickle.load(f)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['white','red'])
preds = preds.T + preds
preds[preds<0.0] = 0.0
preds = 1 - preds
df = pd.DataFrame(preds)
df.index = group
df.columns = group
ax = sns.clustermap(data=df,cmap=cmap,row_cluster=True,col_cluster=True,method='average',vmin=0.75)
plt.subplots_adjust(bottom=0.2,right=0.8)
plt.savefig('tw10_clustermap.eps')