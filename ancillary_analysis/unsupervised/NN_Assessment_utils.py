from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score,accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, KFold
import os
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
import itertools
from Bio.Align import PairwiseAligner
import numpy as np
from multiprocessing import Pool
import colorsys
import matplotlib.patches as mpatches
from scipy.stats import wasserstein_distance, entropy
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics as skmetrics
import phenograph_local
from scipy.spatial import distance
import itertools


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

    OH = OneHotEncoder(sparse_output=False)
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

def Assess_Performance_KNN(distances,names,labels,dir_results,k_values=list(range(1, 500, 25)),rep=5,
                           metrics=['Recall','Precision','F1_Score','AUC']):
    temp = []
    for v in k_values:
        temp.extend(rep*[v])
    k_values = temp
    class_list = []
    algorithm = []
    k_list = []
    metric_list = []
    val_list = []

    #convert distances to squareform if need be
    temp = []
    for d in distances:
        if len(d.shape)==1:
            d = squareform(d)
        temp.append(d)
    distances = temp

    for k in k_values:
        for n,d in zip(names,distances):
            classes, metric,value,k_l = KNN(d, labels, k=k,metrics=metrics)
            metric_list.extend(metric)
            val_list.extend(value)
            class_list.extend(classes)
            k_list.extend(k_l)
            algorithm.extend(len(classes) * [n])


    df_out = pd.DataFrame()
    df_out['Classes'] = class_list
    df_out['Metric'] = metric_list
    df_out['Value'] = val_list
    df_out['Algorithm'] = algorithm
    df_out['k'] = k_list
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    df_out.to_csv(os.path.join(dir_results,'df.csv'),index=False)

    return df_out

def Plot_Performance(df,dir_results,metrics=None):
    subdir = 'Performance'
    if not os.path.exists(os.path.join(dir_results,subdir)):
        os.makedirs(os.path.join(dir_results,subdir))

    if metrics is None:
        metrics = np.unique(df['Metric'].tolist())

    types = np.unique(df['Classes'].tolist())
    for m in metrics:
        fig,ax = plt.subplots(3,3,figsize=(10,10))
        ax = np.ndarray.flatten(ax)
        for ii,t in enumerate(types,0):
            df_temp = df[(df['Classes']==t) & (df['Metric']==m)]
            sns.pointplot(data=df_temp,x='k',y='Value',hue='Algorithm',capsize=0.2,ax=ax[ii])
            ax[ii].set_title(t,fontsize=24)
            ax[ii].set_ylabel(m)
            ax[ii].set_xticklabels(ax[ii].get_xticklabels(),rotation=90,fontsize=12)
            # ax[ii].set_yticklabels(ax[ii].get_yticklabels(),fontsize=12)
            ax[ii].set_xlabel('k',fontsize=18)
            ax[ii].set_ylabel(m,fontsize=18)
            ax[ii].get_legend().remove()
        plt.tight_layout()
        plt.savefig(os.path.join(dir_results,subdir,m+'.png'),dpi=1200)
        plt.close()

def Plot_Performance_Samples(df,dir_results,metrics=None,distance_methods=None):
    subdir = 'Performance'
    if not os.path.exists(os.path.join(dir_results,subdir)):
        os.makedirs(os.path.join(dir_results,subdir))

    if metrics is None:
        metrics = np.unique(df['Metric'].tolist())

    if distance_methods is None:
        distance_methods = np.unique(df['Distance Metric'].tolist())

    types = np.unique(df['Classes'].tolist())
    # metrics = ['AUC']
    # types = ['Combo']

    for m in metrics:
        fig,axes = plt.subplots(len(distance_methods),len(types),figsize=(15,20))
        for ii,t in enumerate(types,0):
            for jj,d in enumerate(distance_methods,0):
                df_temp = df[(df['Classes']==t) & (df['Metric']==m) & (df['Distance Metric']==d)]
                ax = axes[jj,ii]
                sns.pointplot(data=df_temp,x='k',y='Value',hue='Algorithm',capsize=0.2,ax=ax)
                ax.get_legend().remove()
                ax.set_title(t+'_'+d)
                ax.tick_params(axis='x',rotation=90)
                ax.get_xaxis().set_visible(False)
                ax.set_ylabel('')
                ax.set_ylim([0,1.0])

        plt.savefig(os.path.join(dir_results,subdir,m+'.eps'))

def Plot_Latent(labels,methods,dir_results):
    subdir = 'Latent'
    if not os.path.exists(os.path.join(dir_results,subdir)):
        os.makedirs(os.path.join(dir_results,subdir))

    N = len(np.unique(labels))
    HSV_tuples = [(x * 1.0 / N, 1.0, 0.5) for x in range(N)]
    np.random.shuffle(HSV_tuples)
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    color_dict = dict(zip(np.unique(labels), RGB_tuples))

    patches = []
    for item in color_dict.items():
        patches.append(mpatches.Patch(color=item[1], label=item[0]))

    c = [color_dict[x] for x in labels]
    names = ['VAE-Seq', 'VAE-Seq-Gene', 'Hamming', 'K-mer','Global Seq-Align']
    wasserstein_distances_out = []
    X_2_list = []
    for m, n in zip(methods, names):
        X_2 = umap.UMAP(metric='precomputed').fit_transform(m)
        X_2_list.append(X_2)
        H,edges = np.histogramdd(X_2,normed=True)
        w_distances = []
        for type in np.unique(labels):
            idx = labels==type
            d_1,_ = np.histogramdd(X_2[idx],bins=edges,normed=True)
            d_2,_ = np.histogramdd(X_2[~idx],bins=edges,normed=True)
            w_distances.append(wasserstein_distance(np.ndarray.flatten(d_1),np.ndarray.flatten(d_2)))

        wasserstein_distances_out.append(w_distances)

        plt.figure()
        plt.scatter(X_2[:, 0], X_2[:, 1], c=c,s=100,alpha=0.75,label=labels)
        plt.title(n)
        plt.legend(handles=patches)
        plt.savefig(os.path.join(dir_results,subdir,n+'_latent.tif'))

        import pickle
        with open('X_2.pkl','wb') as f:
            pickle.dump([X_2_list,labels],f)

def kmer_search(sequences):
    all = []
    for seq in sequences:
        all.extend(seq)
    all = list(set(all))

    mers = list(itertools.product(all, repeat=3))
    matrix = np.zeros(shape=[len(sequences), len(mers)])

    for ii, seq in enumerate(sequences, 0):
        for jj, m in enumerate(mers, 0):
            matrix[ii, jj] = seq.count(''.join(m))

    matrix = matrix[:,np.sum(matrix,0)>0]

    return matrix

def get_score(a,b):
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    s_12 = aligner.score(a, b)
    s_11 = aligner.score(a, a)
    s_22 = aligner.score(b, b)
    return (1-s_12/s_11)*(1-s_12/s_22)


def pairwise_alignment(sequences):

    seq_pw = list(itertools.product(sequences,repeat=2))
    p = Pool(40)
    result = p.starmap(get_score, seq_pw)
    p.close()
    p.join()
    matrix = np.array(result).reshape([len(sequences),len(sequences)])

    return matrix

def Class_Distances(distances,labels):
    intra_distances = []
    inter_distances = []
    for c in np.unique(labels):
        idx = labels == c
        d = distances[idx, :]
        d = d[:, idx]
        intra_distances.append(np.ndarray.flatten(d))
        d = distances[idx, :]
        d = d[:, ~idx]
        inter_distances.append(np.ndarray.flatten(d))

    intra_distances = np.hstack(intra_distances)
    inter_distances = np.hstack(inter_distances)

    plt.figure()
    plt.hist(inter_distances, 100, alpha=0.5,label='Within-Class Distances')
    plt.hist(intra_distances, 100, alpha=0.5,label='Out-of-Class Distances')
    plt.legend()

def variance_ratio_criteria(d, l):
    idx = l[:, np.newaxis] == l[np.newaxis, :]
    n_clusters = len(np.unique(l))
    n_data = d.shape[0]
    return (np.sum(d[~idx]) / np.sum(d[idx])) * ((n_data - n_clusters) / (n_clusters - 1))

def Clustering_Quality(distances,m,l):
    temp = []
    for d in distances:
        if len(d.shape) == 1:
            d = squareform(d)
        temp.append(d)
    distances = temp

    #n_clusters = np.concatenate([np.arange(5, 21, 1), np.arange(20, 101, 10)])
    n_clusters = np.arange(5,101,1)
    cluster_metrics = list()
    for n in n_clusters:
        print(n)
        for i in range(len(distances)):
            sc = AgglomerativeClustering(n_clusters=n, affinity='precomputed', linkage='complete').fit(distances[i])

            cluster_metrics.append([m[i], n,
                                    skmetrics.homogeneity_score(l, sc.labels_),
                                    skmetrics.completeness_score(l, sc.labels_),
                                    skmetrics.v_measure_score(l, sc.labels_),
                                    skmetrics.adjusted_rand_score(l, sc.labels_),
                                    skmetrics.adjusted_mutual_info_score(l, sc.labels_, average_method='arithmetic'),
                                    skmetrics.silhouette_score(distances[i], sc.labels_, metric='precomputed'),
                                    variance_ratio_criteria(distances[i], sc.labels_)])

    df = pd.DataFrame(cluster_metrics, columns=['Algorithm', 'n_clusters', 'Homogeneity', 'Completeness', 'V-measure',
                                               'Adjusted Rand Index', 'Adjusted Mutual Information',
                                               'Silhouette Coefficient', 'Variance Ratio Criteria'])

    return df


def phenograph_clustering(d):
    nbrs = NearestNeighbors(n_neighbors=30, metric='precomputed').fit(d)
    d, idx = nbrs.kneighbors(d)
    IDX, _, _ = phenograph_local.cluster(d=d, idx=idx, n_jobs=1)
    c_freq = []
    for i in np.unique(IDX):
        if i != -1:
            sel = IDX == i
            c_freq.append(np.sum(sel))

    return c_freq

def phenograph_clustering_freq(d,DTCRU,n_jobs=1):
    nbrs = NearestNeighbors(n_neighbors=30, metric='precomputed',n_jobs=n_jobs).fit(d)
    d_knn, idx = nbrs.kneighbors(d)
    IDX, _, _ = phenograph_local.cluster(d=d_knn, idx=idx, n_jobs=1)

    label_id = DTCRU.class_id
    file_id = DTCRU.sample_id
    freq = DTCRU.freq
    file_list = DTCRU.sample_list

    DFs = []
    DF_Sum = pd.DataFrame()
    DF_Sum['File'] = file_list
    DF_Sum.set_index('File', inplace=True)
    for i in np.unique(IDX):
        if i != -1:
            sel = IDX == i
            label_sel = label_id[sel]
            file_sel = file_id[sel]
            freq_sel = freq[sel]

            df = pd.DataFrame()
            df['Labels'] = label_sel
            df['File'] = file_sel
            df['Frequency'] = freq_sel

            df_sum = df.groupby(by='File', sort=False).agg({'Frequency': 'sum'})

            DF_Sum['Cluster_' + str(i)] = df_sum

            DFs.append(df)

    DF_Sum.fillna(0.0, inplace=True)
    return DF_Sum, IDX

def KNN_samples(distances,labels,k,metrics,folds):
    lb = LabelEncoder()
    labels = lb.fit_transform(labels)

    if folds > np.min(np.bincount(labels)):
        skf = KFold(n_splits=folds, random_state=None, shuffle=True)
    else:
        skf = StratifiedKFold(n_splits=folds, random_state=None, shuffle=True)
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

    OH = OneHotEncoder(sparse_output=False)
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

def sym_KL(u,v):
    return entropy(u,v) + entropy(v,u)


def Get_Prop_Distances(prop_list,names,eps = 1e-9):
    # distance_func = [wasserstein_distance, distance.euclidean, sym_entropy, distance.correlation, distance.braycurtis,
    #                  distance.canberra, distance.chebyshev,
    #                  distance.cityblock, distance.cosine, distance.jensenshannon, distance.minkowski]
    # distance_names = ['Wasserstein', 'Euclidean', 'KL', 'Correlation', 'Bray-Curtis', 'Canberra', 'Chebyshev',
    #                   'Manhattan', 'Cosine', 'Jensen-Shannon', 'Minkowski']

    distance_func = [distance.euclidean,distance.correlation,sym_KL,distance.jensenshannon,wasserstein_distance]
    distance_names = ['Euclidean','Correlation','KL-Divergence','JS-Divergence','Wasserstein']

    distances_list = []
    distances_names_list = []
    method_names = []
    for func, n in zip(distance_func, distance_names):
        for prop,n_method in zip(prop_list,names):
            pairwise_distances = np.zeros(shape=[len(prop), len(prop)])
            prop += eps
            for ii, i in enumerate(prop.index, 0):
                for jj, j in enumerate(prop.index, 0):
                    pairwise_distances[ii, jj] = func(prop.loc[i], prop.loc[j])

            distances_list.append(pairwise_distances)
            distances_names_list.append(n)
            method_names.append(n_method)

    return distances_list, distances_names_list, method_names

def Assess_Performance_KNN_Samples(distances_list,distances_names,method_names,dir_results,labels,folds=5):
    k_values = list(range(1,len(distances_list[0])))
    metrics = ['Recall', 'Precision', 'F1_Score', 'AUC']


    class_list = []
    algorithm = []
    distance_metric = []
    k_list = []
    metric_list = []
    val_list = []
    for k in k_values:
        for d,n_d,n_m in zip(distances_list,distances_names,method_names):
            try:
                classes,metric,value,k_l = KNN_samples(d,labels,k=k,metrics=metrics,folds=folds)
                metric_list.extend(metric)
                val_list.extend(value)
                class_list.extend(classes)
                k_list.extend(k_l)
                algorithm.extend(len(classes) * [n_m])
                distance_metric.extend(len(classes) * [n_d])
            except:
                continue


    df_out = pd.DataFrame()
    df_out['Classes'] = class_list
    df_out['Metric'] = metric_list
    df_out['Value'] = val_list
    df_out['Algorithm'] = algorithm
    df_out['k'] = k_list
    df_out['Distance Metric'] = distance_metric

    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    df_out.to_csv(os.path.join(dir_results,'df.csv'),index=False)

    return df_out
