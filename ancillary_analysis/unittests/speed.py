import numpy as np
import pandas as pd
import glob
import os
import time
from DeepTCR.functions.data_processing import Get_DF_Data
import pickle
import tracemalloc

files = glob.glob('../../Data/natgen/data/cohort1/*.tsv')
np.random.seed(0)
files = np.random.choice(files,100,replace=False)

tracemalloc.start()
start = time.time()
for file in files:
    out = Get_DF_Data(file,type_of_data_cut='Fraction_Response', data_cut=1.0, aa_column_alpha=None, aa_column_beta=1,
                    count_column=5, sep='\t', max_length=40, aggregate_by_aa=True, v_beta_column=10,
                    d_beta_column=13, j_beta_columns=16,
                    v_alpha_column=None, j_alpha_column=None)
end = time.time()
first_size, first_peak = tracemalloc.get_traced_memory()
with open('old.pkl','wb') as f:
    pickle.dump([end-start,first_peak/1e6],f,protocol=4)
with open('old.pkl','rb') as f:
    total_time, mem = pickle.load(f)

with open('old.pkl', 'rb') as f:
    total_time_old, mem_old = pickle.load(f)

with open('new.pkl','rb') as f:
    total_time_new, mem_new = pickle.load(f)

