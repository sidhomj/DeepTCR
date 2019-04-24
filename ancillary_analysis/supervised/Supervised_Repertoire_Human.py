from DeepTCR.DeepTCR import DeepTCR_WF

#Train Sequence Classifier
DTCR = DeepTCR_WF('Human_TIL',device='/gpu:2')
dir = 'Topalian/beta/pre_crpr_sdpd'
DTCR.Get_Data(directory='../../Data/Topalian',Load_Prev_Data=False,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=0.25,
              hla='../../Data/Topalian/HLA_Ref.csv')

folds = 100
LOO = 5
epochs_min = 50
weight_by_class = True
size_of_net = 'small'

#Just train w/ Sequence Information
DTCR.use_hla = False
DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,
                          use_only_hla=False,weight_by_class=weight_by_class,size_of_net=size_of_net)
DTCR.AUC_Curve(filename='AUC_Seq.eps')

#Just train w/HLA
DTCR.use_hla = True
DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,
                          use_only_hla=True,weight_by_class=weight_by_class,size_of_net=size_of_net)
DTCR.AUC_Curve(filename='AUC_HLA.eps')

#Train with both Seq + HLA
DTCR.Monte_Carlo_CrossVal(folds=folds,LOO=LOO,epochs_min=epochs_min,
                          use_only_hla=False,weight_by_class=weight_by_class,size_of_net=size_of_net)
DTCR.AUC_Curve(filename='AUC_Seq_HLA.eps')


