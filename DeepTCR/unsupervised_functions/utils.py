import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics as skmetrics
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage,fcluster
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, optimal_leaf_ordering, leaves_list
from scipy.stats import entropy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score,accuracy_score


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

def rad_plot(X_2,pairwise_distances,samples,labels,file_id,color_dict,gridsize=50,
             dg_radius=0.2,axes_radius=0.4,figsize=8,log_scale=False,linkage_method='complete',plot_type='hexbin'):

    n_s = len(np.unique(samples))
    clim = np.array([0, .1])
    d_max = np.max(X_2, axis=0)
    d_min = np.min(X_2, axis=0)
    c_center = (d_max + d_min) / 2
    c_radius = np.max(np.sqrt(np.sum(np.power(X_2 - c_center[np.newaxis, :], 2), axis=1))) * 1.1
    c_pos = pol2cart(np.linspace(0, 2 * np.pi, 200), c_radius) + c_center[np.newaxis, :]

    x_edges = np.linspace(d_min[0], d_max[0], gridsize)
    y_edges = np.linspace(d_min[1], d_max[1], gridsize)
    Y, X = np.meshgrid(x_edges[:-1] + (np.diff(x_edges) / 2), y_edges[:-1] + (np.diff(y_edges) / 2))

    Z = optimal_leaf_ordering(linkage(pairwise_distances, method=linkage_method), pairwise_distances)
    dg_order = leaves_list(Z)

    fig = plt.figure(figsize=[figsize, figsize])
    axes_pos = pol2cart(np.linspace(0, 2 * np.pi, n_s + 1), rho=axes_radius) + 0.5
    axes_size = axes_radius * np.sin(0.5 * (2 * np.pi / n_s))
    ax = [None] * n_s

    for i in range(n_s):
        ax[i] = fig.add_axes([axes_pos[i, 0] - axes_size, axes_pos[i, 1] - axes_size, 2 * axes_size, 2 * axes_size])
        ax[i].plot(c_pos[:, 0], c_pos[:, 1], '-', linewidth=5., color=color_dict[labels[dg_order[i]]])
        smp_d = X_2[file_id == samples[dg_order[i]], :]
        if plot_type is 'hexbin':
            ax[i].hexbin(smp_d[:, 0], smp_d[:, 1], gridsize=gridsize, mincnt=1)
        elif plot_type is '2dhist':
            h, _ = np.histogramdd(smp_d, [x_edges, y_edges])
            ax[i].pcolormesh(X, Y, h / np.sum(h), shading='gouraud', vmin=clim[0], vmax=clim[1], cmap='GnBu')
        else:
            ax[i].plot(smp_d, '.', markersize=1, alpha=0.5)
        ax[i].set(xticks=[], yticks=[],frame_on=False)

    dg = dendrogram(Z, no_plot=True)
    polar_dendrogram(dg, fig, ax_radius=dg_radius, log_scale=log_scale)

def KNN(distances,labels,k=1,metrics=['Recall','Precision','F1_Score','AUC']):
    lb = LabelEncoder()
    labels = lb.fit_transform(labels)

    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
    neigh = KNeighborsClassifier(n_neighbors=k, metric='precomputed', weights='distance')

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

def KNN_samples(distances,labels,k,metrics):
    lb = LabelEncoder()
    labels = lb.fit_transform(labels)

    skf = LeaveOneOut()
    neigh = KNeighborsClassifier(n_neighbors=k, metric='precomputed', weights='distance')

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


