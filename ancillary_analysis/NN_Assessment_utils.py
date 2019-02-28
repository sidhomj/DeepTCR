from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score,accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import os
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
import itertools
from Bio import pairwise2
import numpy as np
from Bio.pairwise2 import format_alignment
from multiprocessing import Pool


def KNN(distances,labels,k=1):
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

    OH = OneHotEncoder(sparse=False)
    labels = OH.fit_transform(labels.reshape(-1,1))
    pred = OH.transform(pred.reshape(-1,1))

    recall = []
    precision = []
    f_score = []
    auc_score = []
    acc_score = []
    for ii,c in enumerate(lb.classes_):
        recall.append(recall_score(y_true=labels[:,ii],y_pred=pred[:,ii]))
        precision.append(precision_score(y_true=labels[:,ii],y_pred=pred[:,ii]))
        f_score.append(f1_score(y_true=labels[:, ii], y_pred=pred[:,ii]))
        auc_score.append(roc_auc_score(labels[:, ii],pred_prob[:,ii]))
        acc_score.append(accuracy_score(y_true=labels[:,ii],y_pred=pred[:,ii]))

    return lb.classes_,recall, precision,f_score,auc_score,acc_score

def VAE_GAN_Distances(DTCRU,Load_Prev_Data=False):
    # Train VAE/GAN
    DTCRU.Train_VAE(accuracy_min=0.9, Load_Prev_Data=Load_Prev_Data,ortho_norm=True,seq_features_latent=True)
    distances_vae = squareform(pdist(DTCRU.features, metric='euclidean'))
    DTCRU.Train_GAN(Load_Prev_Data=Load_Prev_Data,ortho_norm=True,use_distances=False)
    distances_gan = squareform(pdist(DTCRU.features, metric='euclidean'))
    return distances_vae, distances_gan

def Assess_Performance(DTCRU, distances_vae, distances_gan, distances_hamming, distances_kmer,dir_results,use_genes_label='use_genes'):
    labels = DTCRU.label_id
    k_values = list(range(1, 500, 10))
    #k_values = 100*[300]
    class_list = []
    recall_list = []
    precision_list = []
    f1_score_list = []
    accuracy_list = []
    auc_list = []
    algorithm = []
    k_list = []
    use_genes_list=[]

    for k in k_values:
        # Collect performance metrics for various methods
        # VAE
        classes, recall, precision, f1_score, auc,acc = KNN(distances_vae, labels, k=k)
        class_list.extend(classes)
        recall_list.extend(recall)
        precision_list.extend(precision)
        f1_score_list.extend(f1_score)
        accuracy_list.extend(acc)
        auc_list.extend(auc)
        algorithm.extend(len(classes) * ['VAE'])
        k_list.extend(len(classes) * [k])
        use_genes_list.extend(len(classes)*[use_genes_label])

        # Train GAN
        classes, recall, precision, f1_score, auc,acc = KNN(distances_gan, labels, k=k)
        class_list.extend(classes)
        recall_list.extend(recall)
        precision_list.extend(precision)
        f1_score_list.extend(f1_score)
        accuracy_list.extend(acc)
        auc_list.extend(auc)
        algorithm.extend(len(classes) * ['GAN'])
        k_list.extend(len(classes) * [k])
        use_genes_list.extend(len(classes)*[use_genes_label])


        # Hamming Distance
        classes, recall, precision, f1_score, auc,acc = KNN(distances_hamming, labels, k=k)
        class_list.extend(classes)
        recall_list.extend(recall)
        precision_list.extend(precision)
        f1_score_list.extend(f1_score)
        auc_list.extend(auc)
        accuracy_list.extend(acc)
        algorithm.extend(len(classes) * ['Hamming'])
        k_list.extend(len(classes) * [k])
        use_genes_list.extend(len(classes)*[use_genes_label])


        # Kmer search
        classes, recall, precision, f1_score, auc,acc = KNN(distances_kmer, labels, k=k)
        class_list.extend(classes)
        recall_list.extend(recall)
        precision_list.extend(precision)
        f1_score_list.extend(f1_score)
        auc_list.extend(auc)
        accuracy_list.extend(acc)
        algorithm.extend(len(classes) * ['K-Mer'])
        k_list.extend(len(classes) * [k])
        use_genes_list.extend(len(classes)*[use_genes_label])


        # #Sequence Alignment
        # classes,recall, precision, f1_score = KNN(seq_align_distances,labels[idx],k=k)
        #
        # class_list.extend(classes)
        # recall_list.extend(recall)
        # precision_list.extend(precision)
        # f1_score_list.extend(f1_score)
        # algorithm.extend(len(classes)*['Seq-Align'])
        # k_list.extend(len(classes)*[k])

    df_out = pd.DataFrame()
    df_out['Classes'] = class_list
    df_out['Recall'] = recall_list
    df_out['Precision'] = precision_list
    df_out['F1_Score'] = f1_score_list
    df_out['Accuracy'] = accuracy_list
    df_out['AUC'] = auc_list
    df_out['Algorithm'] = algorithm
    df_out['k'] = k_list
    df_out['Gene_Usage'] = use_genes_list
    df_out.to_csv(os.path.join(dir_results,'df.csv'),index=False)

    classes = DTCRU.lb.classes_
    measurements = ['Recall', 'Precision', 'F1_Score', 'Accuracy','AUC']

    if not os.path.exists(dir_results):
        os.makedirs(dir_results)

    # i = 0
    # for c in classes:
    #     for m in measurements:
    #         idx = df_out['Classes'] == c
    #         plt.figure()
    #         sns.set(font_scale=1.2)
    #         sns.set_style('white')
    #         sns.lineplot(data=df_out[idx], x='k', y=m, hue='Algorithm')
    #         plt.title(c)
    #         plt.xticks(k_values)
    #         plt.savefig(os.path.join(dir_results, str(i) + '.tif'))
    #         plt.close()
    #         i += 1

    return df_out

def Plot_Performance(df):
    fig, ax = plt.subplots(2, 2)
    measurements = ['Recall', 'Precision', 'F1_Score', 'Accuracy']
    ax = np.ndarray.flatten(ax)
    for a, m in zip(ax, measurements):
        sns.boxplot(x='Algorithm', y=m, data=df, ax=a, hue='Classes')

    fig, ax = plt.subplots(2, 2)
    ax = np.ndarray.flatten(ax)
    for a, m in zip(ax, measurements):
        sns.boxplot(x='Algorithm', y=m, data=df, ax=a)

    fig, ax = plt.subplots(2, 2)
    ax = np.ndarray.flatten(ax)
    for a, m in zip(ax, measurements):
        sns.lineplot(x='k', y=m, data=df, hue='Algorithm', ci=None, ax=a)

def Plot_Latent(labels,methods):
    names = ['GAN', 'VAE', 'Hamming', 'K-mer']
    fig, ax = plt.subplots(2, 2)
    ax = np.ndarray.flatten(ax)
    for a, m, n in zip(ax, methods, names):
        X_2 = umap.UMAP(metric='precomputed').fit_transform(m)
        a.scatter(X_2[:, 0], X_2[:, 1], c=labels)
        a.set_title(n)

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
    s_12 =  pairwise2.align.globalxx(a, b)[-1][-1]
    s_11 =  pairwise2.align.globalxx(a, a)[-1][-1]
    s_22 =  pairwise2.align.globalxx(b, b)[-1][-1]
    return (1-s_12/s_11)*(1-s_12/s_22)

def pairwise_alignment(sequences):
    matrix = np.zeros(shape=[len(sequences), len(sequences)])
    seq_1 = []
    seq_2 = []
    for ii,a in enumerate(sequences,0):
        for jj,b in enumerate(sequences,0):
            seq_1.append(a)
            seq_2.append(b)

    args = list(zip(seq_1,seq_2))
    p = Pool(40)
    result = p.starmap(get_score, args)

    p.close()
    p.join()

    i=0
    for ii,a in enumerate(sequences,0):
        for jj,b in enumerate(sequences,0):
            matrix[ii,jj] = result[i]
            i+=1

    return matrix




