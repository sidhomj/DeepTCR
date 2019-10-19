import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics as skmetrics
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage,fcluster
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, optimal_leaf_ordering, leaves_list
from scipy.stats import entropy
from scipy import ndimage as ndi
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, KFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score
from matplotlib.patches import Ellipse
import os


def get_batches(Vars, batch_size=10,random=False):
    """ Return a generator that yields batches from vars. """
    #batch_size = len(x) // n_batches
    x = Vars[0]
    if len(x) % batch_size == 0:
        n_batches = (len(x) // batch_size)
    else:
        n_batches = (len(x) // batch_size) + 1

    sel = np.asarray(list(range(x.shape[0])))
    if random is True:
        np.random.shuffle(sel)

    for ii in range(0, n_batches * batch_size, batch_size):
        # If we're not on the last batch, grab data with size batch_size
        if ii != (n_batches - 1) * batch_size:
            sel_ind=sel[ii: ii + batch_size]
        else:
            sel_ind = sel[ii:]

        Vars_Out = [var[sel_ind] for var in Vars]

        yield Vars_Out

def hierarchical_optimization(distances,features,method,criterion):
    Z = linkage(squareform(distances), method=method)
    t_list = np.arange(0, 100, 1)
    sil = []
    for t in t_list:
        IDX = fcluster(Z, t, criterion=criterion)
        if len(np.unique(IDX[IDX >= 0])) == 1:
            sil.append(0.0)
            continue
        sel = IDX >= 0
        sil.append(skmetrics.silhouette_score(features[sel, :], IDX[sel]))

    IDX = fcluster(Z, t_list[np.argmax(sil)], criterion=criterion)
    return IDX

def dbscan_optimization(distances, features):
    eps_list = np.arange(0.0, 20, 0.1)[1:]
    sil = []
    for ii,eps in enumerate(eps_list,0):
        IDX = DBSCAN(eps=eps, metric='precomputed').fit_predict(distances)
        IDX[IDX == -1] = np.max(IDX + 1)
        if len(np.unique(IDX[IDX >= 0])) == 1:
                sil.append(0.0)
                continue
        sel = IDX >= 0
        sil.append(skmetrics.silhouette_score(features[sel, :], IDX[sel]))

    IDX = DBSCAN(eps=eps_list[np.argmax(sil)], metric='precomputed').fit_predict(distances)

    return IDX

def sym_KL(u,v):
    return entropy(u,v) + entropy(v,u)

def pol2cart(phi, rho=1.):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y]).T

def smoothsegment(seg, Nsmooth=100):
    return np.concatenate([[seg[0]], np.linspace(seg[1], seg[2], Nsmooth), [seg[3]]])

def polar_dendrogram(dg, fig, ax_radius=0.2, log_scale=False):
    icoord = np.asarray(dg['icoord'], dtype=float)
    dcoord = np.asarray(dg['dcoord'], dtype=float)

    # adjust dcoord for radial
    if log_scale:
        dcoord = -np.log(dcoord + 1)
    else:
        dcoord = dcoord.max() - (dcoord + 1)

    # adjust icoord for radial
    imax = icoord.max()
    imin = icoord.min()
    icoord = 2 * np.pi * (icoord.shape[0] / (icoord.shape[0] + 1)) * ((icoord - imin) / (imax - imin))

    # plot
    with plt.style.context("seaborn-white"):
        ax = fig.add_axes([0.5 - ax_radius, 0.5 - ax_radius, 2 * ax_radius, 2 * ax_radius], polar=True)
        for xs, ys in zip(icoord, dcoord):
            xs = smoothsegment(xs)
            ys = smoothsegment(ys)
            ax.plot(xs, ys, color="black")

        ax.spines['polar'].set_visible(False)
        # ax.set(xticks=np.linspace(0, 2 * np.pi, icoord.shape[0] + 2), xticklabels=dg['ivl'], yticks=[])
        ax.set(xticks=[], yticks=[])

def rad_plot(X_2, sample_id, samples, labels, color_dict, self=None, pairwise_distances=None, gridsize=50, n_pad=5, lw=None, dg_radius=0.2, axes_radius=0.4, figsize=8, log_scale=False, linkage_method='complete', filename=None, sample_labels=False, gaussian_sigma=0.5, vmax=0.01):
    # set line width
    if lw is None:
        lw = n_pad / 2

    # number of samplea
    n_s = len(np.unique(samples))

    # min max of input 2D data
    d_max = np.max(X_2, axis=0)
    d_min = np.min(X_2, axis=0)

    # set step and edges of bins for 2d hist
    x_step = (d_max[0] - d_min[0]) / gridsize
    x_edges = np.linspace(d_min[0] - (n_pad * x_step), d_max[0] + (n_pad * x_step), gridsize + (2 * n_pad) + 1)
    y_step = (d_max[1] - d_min[1]) / gridsize
    y_edges = np.linspace(d_min[1] - (n_pad * y_step), d_max[1] + (n_pad * y_step), gridsize + (2 * n_pad) + 1)
    Y, X = np.meshgrid(x_edges[:-1] + (np.diff(x_edges) / 2), y_edges[:-1] + (np.diff(y_edges) / 2))

    # construct 2d smoothed histograms for each sample
    H = list()
    for i in range(n_s):
        # get sample instance data
        smp_d = X_2[sample_id == samples[i]]
        # get counts
        h, _ = np.histogramdd(smp_d, bins=[x_edges, y_edges])
        if log_scale:
            h = np.log(h + 1)
        # normalize and smooth
        H.append(ndi.gaussian_filter(h / np.sum(h), sigma=gaussian_sigma))
    H = np.stack(H, axis=2)

    # center and radius of circle
    e_c = np.array([np.mean(X[:, 0]), np.mean(Y[0, :])])
    e_r = np.abs(np.array([Y[-n_pad + 2, 0] - e_c[1], X[0, -n_pad + 2] - e_c[0]]))
    xlim = [X[0, 0] - (y_step * 2), X[-1, 0] + (y_step * 2)]
    ylim = [Y[0, 0] - (x_step * 2), Y[0, -1] + (x_step * 2)]

    if pairwise_distances is None:
        pairwise_distances = pdist(H.reshape([-1, H.shape[2]]).T, metric='jensenshannon')

    Z = optimal_leaf_ordering(linkage(pairwise_distances, method=linkage_method), pairwise_distances)
    dg_order = leaves_list(Z)

    fig = plt.figure(figsize=[figsize, figsize])
    axes_pos = pol2cart(np.linspace(0, 2 * np.pi, n_s + 1), rho=axes_radius) + 0.5
    axes_size = axes_radius * np.sin(0.5 * (2 * np.pi / n_s))
    ax = [None] * n_s

    cmap_viridis = plt.get_cmap('viridis')
    cmap_viridis.set_under(color='white', alpha=0)
    c_mask = np.meshgrid(np.arange(2 * n_pad + gridsize), np.arange(2 * n_pad + gridsize))
    c_mask = np.sqrt(((c_mask[0] - ((2 * n_pad + gridsize) / 2)) ** 2) + ((c_mask[1] - ((2 * n_pad + gridsize) / 2)) ** 2)) >= (0.95 * ((2 * n_pad + gridsize) / 2))

    for i in range(n_s):
        ax[i] = fig.add_axes([axes_pos[i, 0] - axes_size, axes_pos[i, 1] - axes_size, 2 * axes_size, 2 * axes_size])

        if sample_labels:
            ax[i].text(.5, 0.2, samples[dg_order[i]], horizontalalignment='center', transform=ax[i].transAxes)

        ax[i].pcolormesh(X, Y, np.ma.masked_array(H[:, :, dg_order[i]], c_mask), cmap=cmap_viridis, shading='gouraud', vmin=0, vmax=vmax)
        ax[i].add_artist(Ellipse(e_c, width=2 * e_r[1], height=2 * e_r[0], color=color_dict[labels[dg_order[i]]], fill=False, lw=lw))
        ax[i].set(xticks=[], yticks=[], xlim=xlim, ylim=ylim, frame_on=False)

    dg = dendrogram(Z, no_plot=True)
    polar_dendrogram(dg, fig, ax_radius=dg_radius, log_scale=log_scale)
    if filename is not None:
        plt.savefig(os.path.join(self.directory_results, filename))

    return H

def KNN(distances,labels,k=1,folds=5,metrics=['Recall','Precision','F1_Score','AUC'],n_jobs=1):
    lb = LabelEncoder()
    labels = lb.fit_transform(labels)

    if folds > np.min(np.bincount(labels)):
        skf = KFold(n_splits=folds, random_state=None, shuffle=True)
    else:
        skf = StratifiedKFold(n_splits=folds, random_state=None, shuffle=True)

    neigh = KNeighborsClassifier(n_neighbors=k, metric='precomputed', weights='distance',n_jobs=n_jobs)

    pred_list = []
    pred_prob_list = []
    labels_list = []
    for train_idx, test_idx in skf.split(distances,labels):
        distances_train = distances[train_idx, :]
        distances_train = distances_train[:, train_idx]

        distances_test = distances[test_idx, :]
        distances_test = distances_test[:, train_idx]

        labels_train = labels[train_idx]
        labels_test = labels[test_idx]

        neigh.fit(distances_train, labels_train)
        pred = neigh.predict(distances_test)
        pred_prob = neigh.predict_proba(distances_test)

        labels_list.extend(labels_test)
        pred_list.extend(pred)
        pred_prob_list.extend(pred_prob)

    pred = np.asarray(pred_list)
    pred_prob = np.asarray(pred_prob_list)
    labels = np.asarray(labels_list)

    OH = OneHotEncoder(sparse=False,categories='auto')
    labels = OH.fit_transform(labels.reshape(-1,1))
    pred = OH.transform(pred.reshape(-1,1))

    metric = []
    value = []
    classes=[]
    k_list = []
    for ii,c in enumerate(lb.classes_):
        if 'Recall' in metrics:
            value.append(recall_score(y_true=labels[:,ii],y_pred=pred[:,ii]))
            metric.append('Recall')
            classes.append(c)
            k_list.append(k)
        if 'Precision' in metrics:
            value.append(precision_score(y_true=labels[:,ii],y_pred=pred[:,ii]))
            metric.append('Precision')
            classes.append(c)
            k_list.append(k)
        if 'F1_Score' in metrics:
            value.append(f1_score(y_true=labels[:, ii], y_pred=pred[:,ii]))
            metric.append('F1_Score')
            classes.append(c)
            k_list.append(k)
        if 'AUC' in metrics:
            value.append(roc_auc_score(labels[:, ii],pred_prob[:,ii]))
            metric.append('AUC')
            classes.append(c)
            k_list.append(k)


    return classes,metric,value,k_list

def KNN_samples(distances,labels,k,metrics,folds,n_jobs):
    lb = LabelEncoder()
    labels = lb.fit_transform(labels)

    if folds > np.min(np.bincount(labels)):
        skf = KFold(n_splits=folds, random_state=None, shuffle=True)
    else:
        skf = StratifiedKFold(n_splits=folds, random_state=None, shuffle=True)

    neigh = KNeighborsClassifier(n_neighbors=k, metric='precomputed', weights='distance',n_jobs=n_jobs)

    pred_list = []
    pred_prob_list = []
    labels_list = []
    for train_idx, test_idx in skf.split(distances,labels):
        distances_train = distances[train_idx, :]
        distances_train = distances_train[:, train_idx]

        distances_test = distances[test_idx, :]
        distances_test = distances_test[:, train_idx]

        labels_train = labels[train_idx]
        labels_test = labels[test_idx]

        neigh.fit(distances_train, labels_train)
        pred = neigh.predict(distances_test)
        pred_prob = neigh.predict_proba(distances_test)

        labels_list.extend(labels_test)
        pred_list.extend(pred)
        pred_prob_list.extend(pred_prob)

    pred = np.asarray(pred_list)
    pred_prob = np.asarray(pred_prob_list)
    labels = np.asarray(labels_list)

    OH = OneHotEncoder(sparse=False,categories='auto')
    labels = OH.fit_transform(labels.reshape(-1,1))
    pred = OH.transform(pred.reshape(-1,1))

    metric = []
    value = []
    classes=[]
    k_list = []
    for ii,c in enumerate(lb.classes_):
        if 'Recall' in metrics:
            value.append(recall_score(y_true=labels[:,ii],y_pred=pred[:,ii]))
            metric.append('Recall')
            classes.append(c)
            k_list.append(k)
        if 'Precision' in metrics:
            value.append(precision_score(y_true=labels[:,ii],y_pred=pred[:,ii]))
            metric.append('Precision')
            classes.append(c)
            k_list.append(k)
        if 'F1_Score' in metrics:
            value.append(f1_score(y_true=labels[:, ii], y_pred=pred[:,ii]))
            metric.append('F1_Score')
            classes.append(c)
            k_list.append(k)
        if 'AUC' in metrics:
            value.append(roc_auc_score(labels[:, ii],pred_prob[:,ii]))
            metric.append('AUC')
            classes.append(c)
            k_list.append(k)

    return classes,metric,value,k_list


