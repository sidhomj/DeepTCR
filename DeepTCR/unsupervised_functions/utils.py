import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics as skmetrics
import sklearn
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage,fcluster


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

