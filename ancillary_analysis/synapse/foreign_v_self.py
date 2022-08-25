import numpy as np
import utils
from DeepTCR.DeepSynapse import DeepSynapse

df = utils.load_foreign_v_self()

DTCR = DeepSynapse('foreign_v_self')
DTCR.Load_Data(epitope_sequences = np.array(df['AA Sequence']),
                class_labels=np.array(df['Y']),
               )

DTCR.Monte_Carlo_CrossVal(folds=1,batch_size=50000,epochs_min=10)
DTCR.AUC_Curve(title='Alien Score Predictive Power')
