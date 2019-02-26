import os
import glob
from multiprocessing import Pool
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns

#Get Data from All Samples Used in Manuscript
main_dir = '/home/sidhom/DeepTCR_2/Data'
sub_dirs = os.listdir(main_dir)
sub_dirs = sub_dirs[0:2]

l = []
freq = []
files = []
clonality = []
p = Pool(40)

for dir in sub_dirs:
    classes = os.listdir(os.path.join(main_dir,dir))
    for type in classes:
        files_read = glob.glob(os.path.join(main_dir,dir,type)+'/*tsv')
        DFs = p.map(Parse_DF, files_read)
        for df,file in zip(DFs,files_read):
            l+=df['aminoAcid'].str.len().tolist()
            f = np.asarray(df['Freq'].tolist())
            clonality.append((-np.sum((f * np.log10(f))))/np.log10(len(f)))
            freq.append(df['Freq'].tolist())
            files.append(file)

p.close()


#Plot Length Distribution
plt.figure()
sns.distplot(l,kde=False)
plt.xlabel('Length of CDR3',fontsize=12)
plt.ylabel('Number of Sequences',fontsize=12)
plt.title('Length Distribution of CDR3',fontsize=16)
plt.gcf().subplots_adjust(left=0.2)
plt.savefig('Length_Distribution.tif')

#Plot Frequency Distribution
dir = 'Sub_Plots'
if not os.path.exists(dir):
    os.makedirs(dir)

count = 0
out = 1
num_rows = 4
num_cols = 3
num_page = num_rows*num_cols
for f,file in zip(freq,files):
    if count % num_page == 0:
        fig,ax = plt.subplots(num_rows,num_cols,figsize=(15,15))
        ax = np.ndarray.flatten(ax)
        n=0
    sns.distplot(np.log10(f), kde=False,ax=ax[n])
    ax[n].set_title(file.split('/')[-1].split('.')[0])
    n+=1
    count+=1
    if (n == num_page) or (count == len(files)):
        plt.savefig(os.path.join(dir,str(out)+'.tif'))
        plt.close()
        out+=1

#Plot Clonality Distribution
plt.figure()
sns.distplot(clonality,kde=False,norm_hist=False)
plt.xlabel('Clonality',fontsize=12)
plt.ylabel('Number of Samples',fontsize=12)
plt.title('Clonality Distribution',fontsize=16)
plt.savefig('Clonality_Distribution.tif')

