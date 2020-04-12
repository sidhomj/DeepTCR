"""Figure 2E"""

"""This script runs regression for the 10x Dataset where alpha/beta TCR's are
regressed against the quantitative evaluation of antigen-specificity via
dCODE Dextramer reagents"""

import pandas as pd
from DeepTCR.DeepTCR import DeepTCR_SS
import numpy as np
from multiprocessing import Pool
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

df = pd.read_csv('../../Data/10x_Data/Data_Regression.csv')
DTCRS = DeepTCR_SS('reg',device=2)
p = Pool(40)

#Get alpha/beta sequences
alpha = np.asarray(df['alpha'].tolist())
beta = np.asarray(df['beta'].tolist())

y_pred = []
y_test = []
antigen = []
folds = 5
seeds = np.array(range(folds))
graph_seed = 0
#Iterate through all antigens
for i in range(2,df.columns.shape[0]):
    print(df.iloc[:,i].name)
    sel = df.iloc[:,i]
    Y = np.log2(np.asarray(sel.tolist()) + 1)
    DTCRS.Load_Data(alpha_sequences=alpha, beta_sequences=beta, Y=Y,p=p)
    DTCRS.K_Fold_CrossVal(folds=folds,seeds=seeds,graph_seed=graph_seed)
    y_pred.append(DTCRS.y_pred)
    y_test.append(DTCRS.y_test)
    antigen.append([sel.name]*len(DTCRS.y_pred))

antigen = np.hstack(antigen)
y_pred = np.vstack(y_pred)
y_test = np.vstack(y_test)

#Save Data
df_out = pd.DataFrame()
df_out['Antigen'] = antigen
df_out['Y_Pred'] = y_pred
df_out['Y_Test'] = y_test
df_out.to_csv('Regression_Results.csv',index=False)

#Load Data
df_out = pd.read_csv('Regression_Results.csv')
ant = np.unique(df_out['Antigen'])

dir  = 'density_plots'
if not os.path.exists(dir):
    os.makedirs(dir)

plt.ioff()
#Create Plots
for a in ant:
    df_temp = df_out[df_out['Antigen'] == a]
    x = np.array(df_temp['Y_Pred'])
    y = np.array(df_temp['Y_Test'])
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    r = np.argsort(z)
    x ,y, z = x[r], y[r], z[r]
    plt.figure()
    plt.scatter(x,y,s=15,c=z,cmap=plt.cm.jet)
    plt.title(a,fontsize=12)
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.xlabel('Predicted',fontsize=12)
    plt.ylabel('Log2(counts+1)',fontsize=12)
    plt.savefig(os.path.join(dir,a+'.eps'))
    plt.close()
