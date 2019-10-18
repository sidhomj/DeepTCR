from DeepTCR.DeepTCR import DeepTCR_SS

#Run Supervised Sequence Classifier
DTCRS = DeepTCR_SS('Sequence_C')
DTCRS.Get_Data(directory='../../Data/Murine_Antigens',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3)
DTCRS.Monte_Carlo_CrossVal(folds=10)
DTCRS.AUC_Curve()
