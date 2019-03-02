from DeepTCR.DeepTCR_U import DeepTCR_U
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from multiprocessing import Pool
from scipy.stats import entropy


#Instantiate training object
DTCRU = DeepTCR_U('Reperoire_Distances')

# #Assess ability for structural entropy to be of measure of number of antigens
# classes_all = np.array(['F2', 'M38', 'M45', 'NP', 'PA', 'PB1', 'm139'])
#
# p = Pool(40)
# num = [1,2,3,4,5,6,7]
# reps = 5
#
# num_list=[]
# entropy_list=[]
# distances_list = []
#
# for n in num:
#     temp = []
#     for r in range(reps):
#         classes = np.random.choice(classes_all,n)
#         DTCRU.Get_Data(directory='../Data/Dash/Traditional/Mouse',Load_Prev_Data=False,aggregate_by_aa=True,classes=classes,
#                             aa_column_alpha=0,aa_column_beta=1,count_column=2,v_alpha_column=3,j_alpha_column=4,v_beta_column=5,j_beta_column=6,p=p)
#         DTCRU.Train_VAE(seq_features_latent=True,accuracy_min=0.9)
#         DTCRU.Structural_Entropy(plot=False)
#         temp.append(DTCRU.distances)
#         num_list.append(n)
#         entropy_list.append(entropy(DTCRU.distances))
#
# df = pd.DataFrame()
# df['Number Of Antigens'] = num_list
# df['Structural Entropy'] = entropy_list
# sns.catplot(data=df,x='Number Of Antigens',y='Structural Entropy',kind='point')


DTCRU.Get_Data(directory='../Data/Rudqvist',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_alpha=None,aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21)
DTCRU.Train_VAE(seq_features_latent=True,accuracy_min=0.9)
DTCRU.Structural_Entropy(plot=True)
check=1


