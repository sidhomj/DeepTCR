"""This script runs regression for the 10x Dataset where alpha/beta TCR's are
regressed against the quantitative evaluation of antigen-specificity via
dCODE Dextramer reagents"""

import pandas as pd
import pickle
from DeepTCR.DeepTCR import DeepTCR_SS
import numpy as np

with open('../../Data/10x_Data/Data_Regression.pkl','rb') as f:
    df = pickle.load(f)

DTCRS = DeepTCR_SS('reg')

#Get alpha/beta sequences
alpha = np.asarray(df['alpha'].tolist())
beta = np.asarray(df['beta'].tolist())

y_pred = []
y_test = []
antigen = []
#Iterate through all antigens
for i in range(2,df.columns.shape[0]):
    sel = df.iloc[:,i]
    Y = np.log2(np.asarray(sel.tolist()) + 1)
    DTCRS.Load_Data(alpha_sequences=alpha, beta_sequences=beta, Y=Y)
    DTCRS.K_Fold_CrossVal(folds=5)
    y_pred.append(DTCRS.y_pred)
    y_test.append(DTCRS.y_test)
    antigen.append([sel.name]*len(DTCRS.y_pred))

antigen = np.hstack(antigen)
y_pred = np.hstack(y_pred)
y_test = np.hstack(y_test)

df_out = pd.DataFrame()
df_out['Antigen'] = antigen
df_out['Y_Pred'] = y_pred
df_out['Y_Test'] = y_test
df_out.to_csv('Regression_Results.csv',index=False)
df_out.sort_values(by='Y_Pred',ascending=False,inplace=True)


