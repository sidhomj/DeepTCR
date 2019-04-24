from DeepTCR.DeepTCR import DeepTCR_SS, DeepTCR_WF

folds = 100
LOO = 4
epochs_min = 100

#Train Sequence Classifier
DTCR_SS = DeepTCR_SS('Rudqvist_SS')
DTCR_SS.Get_Data(directory='../../Data/Rudqvist',Load_Prev_Data=False,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21)

DTCR_SS.Monte_Carlo_CrossVal(folds=folds,test_size=0.25)
DTCR_SS.AUC_Curve(filename='AUC.eps')

#Train Repertoire Classifier without on-graph clustering
DTCR_WF = DeepTCR_WF('Rudqvist_WF')
DTCR_WF.Get_Data(directory='../../Data/Rudqvist',Load_Prev_Data=False,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21)

DTCR_WF.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min)
DTCR_WF.AUC_Curve(filename='Rep_AUC.eps')

#Train Repertoire Classifier with on-graph clustering
DTCR_WF.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,on_graph_clustering=True,epochs_min=epochs_min)
DTCR_WF.AUC_Curve(filename='Rep_AUC_clustering.eps')

#Visualize Latent Space
DTCR_WF.UMAP_Plot(by_class=True,freq_weight=True,show_legend=True,scale=5000,Load_Prev_Data=False,
                  alpha=0.5)

