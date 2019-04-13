from DeepTCR.DeepTCR import DeepTCR_SS, DeepTCR_WF

#Train Sequence Classifier
DTCR_SS = DeepTCR_SS('Rudqvist')
DTCR_SS.Get_Data(directory='../../Data/Rudqvist',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21)

DTCR_SS.Monte_Carlo_CrossVal(folds=100,test_size=0.25)
DTCR_SS.AUC_Curve()

#Train Repertoire Classifier without on-graph clustering
DTCR_WF = DeepTCR_WF('Rudqvist')
DTCR_WF.Get_Data(directory='../../Data/Rudqvist',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21)

DTCR_WF.Monte_Carlo_CrossVal(folds=100,LOO=4)
DTCR_WF.AUC_Curve()

#Train Repertoire Classifier with on-graph clustering
DTCR_WF.Monte_Carlo_CrossVal(folds=100,LOO=4,on_graph_clustering=True)
DTCR_WF.AUC_Curve()

