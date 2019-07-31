import numpy as np
import pandas as pd
from DeepTCR.DeepTCR import DeepTCR_U
import seaborn as sns
import matplotlib.pyplot as plt

# Instantiate training object
DTCRU = DeepTCR_U('Repertoire_Classification')

DTCRU.Get_Data(directory='../../Data/Rudqvist',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21)

df = pd.DataFrame()
df['Sample'] = DTCRU.sample_id
df['Label'] = DTCRU.class_id
df['Length'] = np.sum(DTCRU.X_Seq_beta>0,-1)
df['Freq'] = DTCRU.freq
df['W_Length'] = df['Length']*df['Freq']

df_agg = df.groupby(['Sample']).agg({'Label':'first','W_Length':'sum'})
sns.swarmplot(data=df_agg,x='Label',y='W_Length',order=['Control','9H10','RT','Combo'])
plt.ylabel('Average Length')
