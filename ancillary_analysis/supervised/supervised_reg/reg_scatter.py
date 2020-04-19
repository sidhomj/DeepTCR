from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Arial')
import pickle
import numpy as np

with open('flu_preds.pkl','rb') as f:
    antigen,predicted,Y = pickle.load(f)

# with open('ebv_preds.pkl','rb') as f:
#     antigen,predicted,Y = pickle.load(f)
#
# with open('mart1_preds.pkl','rb') as f:
#     antigen,predicted,Y = pickle.load(f)

x = predicted
y = Y
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)
r = np.argsort(z)
x, y, z = x[r], y[r], z[r]
plt.figure(figsize=(6,5))
plt.scatter(x, y, s=15, c=z, cmap=plt.cm.jet)
plt.title(antigen, fontsize=18)
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.xlabel('Predicted', fontsize=24)
plt.ylabel('Log2(counts+1)', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.subplots_adjust(bottom=0.15)
plt.savefig(antigen+'.png',dpi=1200)