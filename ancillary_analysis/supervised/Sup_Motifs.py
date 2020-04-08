"""
Fig 2C
"""

"""This script was used to train the supervised TCR sequence classifier
and generate the top representative sequences for each class and derive the 
motifs that were learned by the network."""

from DeepTCR.DeepTCR import DeepTCR_SS
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
import numpy as np

#Run Supervised Sequence Classifier
DTCRS = DeepTCR_SS('Sequence_C',device=6)
DTCRS.Get_Data(directory='../../Data/Murine_Antigens',Load_Prev_Data=True,aggregate_by_aa=True,
               aa_column_beta=0,count_column=1,v_beta_column=2,j_beta_column=3)

folds = 100
seeds = np.array(range(folds))
graph_seed = 0
DTCRS.Monte_Carlo_CrossVal(folds=folds,graph_seed=graph_seed,seeds=seeds)
DTCRS.Representative_Sequences(top_seq=50,motif_seq=10,color_scheme='hydrophobicity')

for item in DTCRS.Rep_Seq:
    t = DTCRS.Rep_Seq[item]
    t = t.groupby(['beta']).agg({item:'first'})
    t = t.sort_values(by=item,ascending=False)
    t.reset_index(inplace=True)
    seq = t['beta'].tolist()
    seq = seq[:10]
    out = []
    for s in seq:
        out.append(SeqRecord(Seq(s, IUPAC.protein), s))
    SeqIO.write(out,item+'.fasta','fasta')