from DeepTCR.DeepTCR import DeepTCR_WF

#Train Sequence Classifier
DTCR = DeepTCR_WF('TIL')
dir = 'Topalian/beta/pre_crpr_sdpd'
DTCR.Get_Data(directory='../../Data/Topalian',Load_Prev_Data=False,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=0.25,
              hla='../../Data/Topalian/HLA_Ref.csv')

folds = 100

#Just train w/ Sequence Information
DTCR.use_hla = False
DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=5,stop_criterion=0.25,epochs_min=50,
                          use_only_hla=False,size_of_net='small',weight_by_class=True)
DTCR.AUC_Curve()

#Just train w/HLA
DTCR.use_hla = True
DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=5,stop_criterion=0.25,epochs_min=50,
                          use_only_hla=True,size_of_net='small',weight_by_class=True)
DTCR.AUC_Curve()

#Train with both Seq + HLA
DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=5,stop_criterion=0.25,epochs_min=50,
                          use_only_hla=False,size_of_net='small',weight_by_class=True)
DTCR.AUC_Curve()


