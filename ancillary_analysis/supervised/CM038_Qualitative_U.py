"""
Figure 3f-i, Supplementary Figure 23
"""

"""This script is used to provide a descriptive analysis of the distribution of TCR sequences
within the CheckMate-038 clinical trial.
"""

import pickle
import numpy as np
import pandas as pd
import umap
from DeepTCR.DeepTCR import DeepTCR_WF,DeepTCR_U
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact, ranksums, spearmanr
from sklearn.model_selection import StratifiedShuffleSplit
from umap import UMAP
from scipy import ndimage as ndi
from matplotlib.patches import Circle
import pickle

os.environ["CUDA DEVICE ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

DTCR = DeepTCR_WF('Human_TIL',device='/device:GPU:0')
DTCR.Get_Data(directory='../../Data/CheckMate_038',Load_Prev_Data=False,
               aa_column_beta=1,count_column=2,v_beta_column=7,d_beta_column=14,j_beta_column=21,data_cut=1.0,
              hla='../../Data/CheckMate_038/HLA_Ref_sup_AB.csv')

with open('cm038_ft_pred_perc.pkl','rb') as f:
    features,predicted,perc = pickle.load(f)

win = 10
cut_bottom = np.percentile(predicted[:,0],win)
cut_top = np.percentile(predicted[:,0],100-win)

df_plot = pd.DataFrame()
df_plot['beta'] = DTCR.beta_sequences
df_plot['sample'] = DTCR.sample_id
df_plot['pred'] = predicted[:,0]
df_plot['gt'] = DTCR.class_id
df_plot['freq'] = DTCR.freq

plt.figure()
ax = sns.distplot(df_plot['pred'],1000,color='k',kde=False)
N,bins= np.histogram(df_plot['pred'],1000)
for p,b in zip(ax.patches,bins):
    if b < cut_bottom:
        p.set_facecolor('r')
    elif b > cut_top:
        p.set_facecolor('b')
y_min,y_max = plt.ylim()
plt.xlim([0,1])
plt.xticks(np.arange(0.0,1.1,0.1))
plt.yticks([])
plt.xlabel('')
plt.ylabel('')
plt.show()

beta_sequences = DTCR.beta_sequences
v_beta = DTCR.v_beta
j_beta = DTCR.j_beta
d_beta = DTCR.d_beta
hla = DTCR.hla_data_seq
sample_id = DTCR.sample_id

file = 'cm038_x2_u.pkl'
featurize = False
if featurize:
    DTCR_U = DeepTCR_U('test_hum', device='/device:GPU:6')
    DTCR_U.Load_Data(beta_sequences=beta_sequences, v_beta=v_beta, d_beta=d_beta, j_beta=j_beta, hla=hla)
    DTCR_U.Train_VAE(Load_Prev_Data=False, latent_dim=64,stop_criterion=0.01)
    X_2 = umap.UMAP().fit_transform(DTCR_U.features)
    with open(file, 'wb') as f:
        pickle.dump(X_2, f, protocol=4)
else:
    with open(file,'rb') as f:
        X_2 = pickle.load(f)


df_plot['x'] = X_2[:,0]
df_plot['y'] = X_2[:,1]


def histogram_2d_cohort(d, w, grid_size):
    # center of data
    d_center = np.mean(np.concatenate(d, axis=0), axis=0)
    # largest radius
    d_radius = np.max(np.sum((d_center[np.newaxis, :] - np.concatenate(d, axis=0)) ** 2, axis=1) ** (1 / 2))
    # padding factors
    d_pad = 1.2
    c_pad = 0.9

    # set step and edges of bins for 2d hist
    x_edges = np.linspace(d_center[0] - (d_radius * d_pad), d_center[0] + (d_radius + d_pad), grid_size + 1)
    y_edges = np.linspace(d_center[1] - (d_radius * d_pad), d_center[1] + (d_radius + d_pad), grid_size + 1)
    X, Y = np.meshgrid(x_edges[:-1] + (np.diff(x_edges) / 2), y_edges[:-1] + (np.diff(y_edges) / 2))

    # construct 2d smoothed histograms for each element in lists d and w
    h = np.stack([np.histogramdd(_d, bins=[x_edges, y_edges], weights=_w)[0] for _d, _w in zip(d, w)], axis=2)

    return dict(h=h, X=X, Y=Y, c=dict(center=d_center, radius=d_radius * d_pad * c_pad))


def hist2d_denisty_plot(h, X, Y, ax, log_transform=False, gaussian_sigma=-1, normalize=True, cmap=None, vmax=None, vsym=False):
    D = h
    if log_transform:
        D = np.log(h + 1)
    if gaussian_sigma > 0:
        D = ndi.gaussian_filter(D, gaussian_sigma)
    if normalize:
        D /= np.sum(D)

    if cmap is None:
        cmap = plt.get_cmap('viridis')

    ax.cla()
    ax.pcolormesh(X, Y, D, shading='gouraud', cmap=cmap, vmin=-vmax if (vsym == True) & (vmax is not None) else None, vmax=vmax)
    ax.set(xticks=[], yticks=[], frame_on=False)


grid_size = 250
gaussian_sigma = 1.25
density_vmax = 0.0003
diff_vmax = 0.0003

d = df_plot
d['file'] = d['sample']
d['sample'] = d['sample'].str.replace('_TCRB.tsv', '')
d['counts'] = d.groupby('sample')['freq'].transform(lambda x: x / x.min())

s = pd.read_csv('CM038_BM.csv')
s.rename(columns={'DeepTCR': 'preds'}, inplace=True)
s = s.sort_values('preds')
c_dict = dict(crpr='blue', sdpd='red')
color_labels = [c_dict[_] for _ in s['Response_cat'].values]

cmap_blue = plt.get_cmap('Blues')
cmap_blue(0)
cmap_blue._lut[0] = np.ones(4)
cmap_blue._lut[256] = np.ones(4)
cmap_red = plt.get_cmap('Reds')
cmap_red(0)
cmap_red._lut[0] = np.ones(4)
cmap_red._lut[256] = np.ones(4)
map_dict = dict(crpr=cmap_blue, sdpd=cmap_red)
map_labels = [map_dict[_] for _ in s['Response_cat'].values]

# cmap = plt.get_cmap('viridis')
# cmap = plt.get_cmap('YlGnBu')
# cmap(0)
# cmap._lut = cmap._lut[np.concatenate([np.flip(np.arange(256)), [257, 256, 258]])]
# cmap._lut[[0, 256]] = np.ones(4)

fig_sample_density, ax = plt.subplots(nrows=4, ncols=11)
ax_supp_density = ax.flatten()
H = histogram_2d_cohort([d.loc[d['sample'] == i, ['y', 'x']].values for i in s['sample'].values], [d.loc[d['sample'] == i, 'counts'].values for i in s['sample'].values], grid_size)
for i in range(H['h'].shape[2]):
    hist2d_denisty_plot(H['h'][:, :, i], H['X'], H['Y'], ax_supp_density[i], log_transform=True, gaussian_sigma=gaussian_sigma, cmap=map_labels[i], vmax=density_vmax)
    ax_supp_density[i].add_artist(Circle(H['c']['center'], H['c']['radius'], color=color_labels[i], lw=3, fill=False))
    ax_supp_density[i].set_title('%.3f' % s['preds'].iloc[i], fontsize=18)
[ax_supp_density[i].set(xticks=[], yticks=[], frame_on=False) for i in range(H['h'].shape[-1], len(ax_supp_density))]
plt.gcf().set_size_inches(13, 5.5)
plt.tight_layout()
fig_sample_density.savefig('qual/sample_density.tif',format='tif',dpi=1200)

fig_crpr, ax_crpr = plt.subplots()
ax_crpr.cla()
D = H['h'][:, :, s['Response_cat'] == 'crpr']
D = np.log(D + 1)
D = np.stack([ndi.gaussian_filter(D[:, :, i], sigma=gaussian_sigma) for i in range(D.shape[2])], axis=2)
D /= D.sum(axis=0).sum(axis=0)[np.newaxis, np.newaxis, :]
D = np.mean(D, axis=2)
ax_crpr.pcolormesh(H['X'], H['Y'], D, cmap=cmap_blue, shading='gouraud', vmin=0, vmax=density_vmax)
ax_crpr.set(xticks=[], yticks=[], frame_on=False)
ax_crpr.add_artist(Circle(H['c']['center'], H['c']['radius'], color='blue', lw=5, fill=False))
plt.gcf().set_size_inches(5, 5)
plt.tight_layout()
fig_crpr.savefig('qual/crpr.tif',format='tif',dpi=1200)


fig_sdpd, ax_crpr = plt.subplots()
ax_crpr.cla()
D = H['h'][:, :, s['Response_cat'] == 'sdpd']
D = np.log(D + 1)
D = np.stack([ndi.gaussian_filter(D[:, :, i], sigma=gaussian_sigma) for i in range(D.shape[2])], axis=2)
D /= D.sum(axis=0).sum(axis=0)[np.newaxis, np.newaxis, :]
D = np.mean(D, axis=2)
ax_crpr.pcolormesh(H['X'], H['Y'], D, cmap=cmap_red, shading='gouraud', vmin=0, vmax=density_vmax)
ax_crpr.set(xticks=[], yticks=[], frame_on=False)
ax_crpr.add_artist(Circle(H['c']['center'], H['c']['radius'], color='red', lw=5, fill=False))
plt.gcf().set_size_inches(5, 5)
plt.tight_layout()
fig_sdpd.savefig('qual/sdpd.tif',format='tif',dpi=1200)


fig_sample_diff, ax = plt.subplots(nrows=4, ncols=11)
ax_diff_sample = ax.flatten()

qs = np.quantile(d['pred'].values, [0.1, 0.9])
Ha = histogram_2d_cohort([d.loc[d['sample'] == i, ['y', 'x']].values for i in s['sample'].values],
                         [d.loc[d['sample'] == i, 'counts'].values * (d.loc[d['sample'] == i, 'pred'].values > qs[1]) for i in s['sample'].values],
                         grid_size=grid_size)
Hb = histogram_2d_cohort([d.loc[d['sample'] == i, ['y', 'x']].values for i in s['sample'].values],
                         [d.loc[d['sample'] == i, 'counts'].values * (d.loc[d['sample'] == i, 'pred'].values < qs[0]) for i in s['sample'].values],
                         grid_size=grid_size)

D = np.stack([Ha['h'], Hb['h']], axis=3)
D = np.log(D + 1)
D = np.stack([np.stack([ndi.gaussian_filter(D[:, :, i, j], sigma=gaussian_sigma) for i in range(D.shape[2])], axis=2) for j in range(D.shape[3])], axis=3)
D = (D[:, :, :, 1] - D[:, :, :, 0]) / D.sum(axis=0).sum(axis=0).sum(axis=1)[np.newaxis, np.newaxis, :]

for i in range(D.shape[2]):
    hist2d_denisty_plot(D[:, :, i], Ha['X'], Ha['Y'], ax_diff_sample[i], cmap='bwr', vmax=diff_vmax, vsym=True, normalize=False)
    ax_diff_sample[i].add_artist(Circle(H['c']['center'], H['c']['radius'], color=color_labels[i], lw=3, fill=False))
    ax_diff_sample[i].set_title('%.3f' % s['preds'].iloc[i], fontsize=18)
[ax_diff_sample[i].set(xticks=[], yticks=[], frame_on=False) for i in range(D.shape[2], len(ax_diff_sample))]
plt.gcf().set_size_inches(13, 5.5)
plt.tight_layout()
fig_sample_diff.savefig('qual/sample_diff.tif',format='tif',dpi=1200)

fig_diff_overall, ax_diff_overall = plt.subplots()
hist2d_denisty_plot(np.mean(D, axis=2), Ha['X'], Ha['Y'], ax_diff_overall, cmap='bwr', vmax=diff_vmax, vsym=True, normalize=False)
ax_diff_overall.add_artist(Circle(H['c']['center'], H['c']['radius'], color='grey', lw=5, fill=False))
plt.gcf().set_size_inches(5, 5)
plt.tight_layout()
fig_diff_overall.savefig('qual/cohort_diff.tif',format='tif',dpi=1200)
