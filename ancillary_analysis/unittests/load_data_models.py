from DeepTCR.DeepTCR import DeepTCR_SS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import seaborn as sns
import pickle
import os
import matplotlib
matplotlib.rc('font', family='Arial')

#Instantiate training object
DTCRU = DeepTCR_SS('Murine_Sup')
#Load Data
DTCRU.Get_Data(directory='../../Data/Murine_Antigens',Load_Prev_Data=False,
               aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3,
               classes=['Db-F2', 'Db-M45', 'Db-NP', 'Db-PA', 'Db-PB1'])