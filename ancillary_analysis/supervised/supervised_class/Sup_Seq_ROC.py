"""Figure 2B"""

"""This script is used to create the ROC curves for assessing the ability
of supervised sequence classifier to correctly predict the antigen-specificity of 
the 9 murine antigens in the manuscript.."""

from DeepTCR.DeepTCR import DeepTCR_SS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Arial')

#Run Supervised Sequence Classifier
DTCRS = DeepTCR_SS('Sequence_C',device=2)


DTCRS.Get_Data(directory='../../../Data/Murine_Antigens',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3)

folds = 10
seeds = np.array(range(folds))
graph_seed = 0
DTCRS.Monte_Carlo_CrossVal(folds=folds,seeds=seeds,graph_seed=graph_seed)
DTCRS.AUC_Curve(xlabel_size=24,ylabel_size=24,xtick_size=18,ytick_size=18,legend_font_size=14,frameon=False,
                diag_line=False)