"""Figure 2E"""

"""This script runs regression for the 10x Dataset where alpha/beta TCR's are
regressed against the quantitative evaluation of antigen-specificity via
dCODE Dextramer reagents"""

import pandas as pd
from DeepTCR.DeepTCR import DeepTCR_SS
import numpy as np
from multiprocessing import Pool
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import shutil
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC

df = pd.read_csv('../../Data/10x_Data/Data_Regression.csv')
DTCRS = DeepTCR_SS('reg',device=2)
p = Pool(40)

#Get alpha/beta sequences
alpha = np.asarray(df['alpha'].tolist())
beta = np.asarray(df['beta'].tolist())

antigen = 'A0201_GILGFVFTL_Flu-MP_Influenza'
i = np.where(df.columns==antigen)[0][0]
sel = df.iloc[:, i]
Y = np.log2(np.asarray(sel.tolist()) + 1)
DTCRS.Load_Data(alpha_sequences=alpha, beta_sequences=beta, Y=Y, p=p)
DTCRS.K_Fold_CrossVal(split_by_sample=False, folds=5)
DTCRS.Representative_Sequences(top_seq=100,motif_seq=10,color_scheme='hydrophobicity')

dir = 'Reg_Rep_Sequences'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

item = 'High'
t = DTCRS.Rep_Seq[item]
t = t.sort_values(by='Predicted', ascending=False)
t.reset_index(inplace=True)
seq = t['beta'].tolist()
seq = seq[:10]
out = []
for ii,s in enumerate(seq,0):
    out.append(SeqRecord(Seq(s, IUPAC.protein), str(ii)))
SeqIO.write(out, os.path.join(dir, 'beta.fasta'), 'fasta')

seq = t['alpha'].tolist()
seq = seq[:10]
out = []
for ii,s in enumerate(seq,0):
    out.append(SeqRecord(Seq(s, IUPAC.protein), str(ii)))
SeqIO.write(out, os.path.join(dir, 'alpha.fasta'), 'fasta')


