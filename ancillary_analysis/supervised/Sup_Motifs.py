from DeepTCR.DeepTCR import DeepTCR_SS, DeepTCR_U

#Run Supervised Sequence Classifier
DTCRS = DeepTCR_SS('Sequence_C')
DTCRS.Get_Data(directory='../../Data/Murine_Antigens',Load_Prev_Data=False,aggregate_by_aa=True,
               aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3)

DTCRS.Get_Train_Valid_Test()
DTCRS.Train()
#DTCRS.AUC_Curve()

DTCRS.Representative_Sequences()
antigens =['Db-F2', 'Db-M45', 'Db-NP', 'Db-PA', 'Db-PB1', 'Kb-M38', 'Kb-SIY',
       'Kb-TRP2', 'Kb-m139']
for a in antigens:
    DTCRS.Motif_Identification(group=a)