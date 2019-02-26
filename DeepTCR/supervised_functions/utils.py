import numpy as np
import colorsys
import matplotlib.pyplot as plt
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from scipy.stats import mannwhitneyu
import os
from Bio.Alphabet import IUPAC
import seaborn as sns


def Get_Train_Valid_Test(Vars,Y=None,test_size=0.25,regression=False,LOO = None):

    if regression is False:
        var_train = []
        var_valid = []
        var_test = []
        if Y is not None:
            y_label = np.argmax(Y,1)
            classes = list(set(y_label))

            if LOO is None:
                for ii, type in enumerate(classes, 0):
                    idx = np.where(y_label == type)[0]
                    if idx.shape[0] == 0:
                        continue

                    np.random.shuffle(idx)
                    train_idx = np.random.choice(idx, int((1 - test_size) * idx.shape[0]), replace=False)
                    idx = np.setdiff1d(idx, train_idx)
                    np.random.shuffle(idx)
                    half_val_len = int(idx.shape[0] * 0.5)
                    valid_idx, test_idx = idx[:half_val_len], idx[half_val_len:]

                    if ii == 0:
                        for var in Vars:
                            var_train.append(var[train_idx])
                            var_valid.append(var[valid_idx])
                            var_test.append(var[test_idx])

                        var_train.append(Y[train_idx])
                        var_valid.append(Y[valid_idx])
                        var_test.append(Y[test_idx])
                    else:
                        for jj, var in enumerate(Vars, 0):
                            var_train[jj] = np.concatenate((var_train[jj], var[train_idx]), 0)
                            var_valid[jj] = np.concatenate((var_valid[jj], var[valid_idx]), 0)
                            var_test[jj] = np.concatenate((var_test[jj], var[test_idx]), 0)

                        var_train[-1] = np.concatenate((var_train[-1], Y[train_idx]), 0)
                        var_valid[-1] = np.concatenate((var_valid[-1], Y[valid_idx]), 0)
                        var_test[-1] = np.concatenate((var_test[-1], Y[test_idx]), 0)

            else:
                idx = list(range(len(Y)))
                if LOO ==1:
                    test_idx = np.random.choice(idx, LOO, replace=False)[0]
                else:
                    test_idx = np.random.choice(idx, LOO, replace=False)

                train_idx = np.setdiff1d(idx,test_idx)
                valid_idx = test_idx

                for var in Vars:
                    var_train.append(var[train_idx])
                    if LOO ==1:
                        var_valid.append(np.expand_dims(var[valid_idx], 0))
                        var_test.append(np.expand_dims(var[test_idx], 0))
                    else:
                        var_valid.append(var[valid_idx])
                        var_test.append(var[test_idx])


                var_train.append(Y[train_idx])

                if LOO == 1:
                    var_valid.append(np.expand_dims(Y[valid_idx], 0))
                    var_test.append(np.expand_dims(Y[test_idx], 0))
                else:
                    var_valid.append(Y[valid_idx])
                    var_test.append(Y[test_idx])


    else:
        idx = np.asarray(list(range(len(Y))))
        np.random.shuffle(idx)
        train_idx = np.random.choice(idx, int((1 - test_size) * idx.shape[0]), replace=False)
        idx = np.setdiff1d(idx, train_idx)
        np.random.shuffle(idx)
        half_val_len = int(idx.shape[0] * 0.5)
        valid_idx, test_idx = idx[:half_val_len], idx[half_val_len:]

        var_train = []
        var_valid = []
        var_test = []
        for var in Vars:
            var_train.append(var[train_idx])
            var_valid.append(var[valid_idx])
            var_test.append(var[test_idx])

        var_train.append(Y[train_idx])
        var_valid.append(Y[valid_idx])
        var_test.append(Y[test_idx])



    return var_train,var_valid,var_test

def Get_Train_Test(Vars,test_idx,train_idx,Y=None):
    var_train = []
    var_test = []
    for var in Vars:
        var_train.append(var[train_idx])
        var_test.append(var[test_idx])

    if Y is not None:
        var_train.append(Y[train_idx])
        var_test.append(Y[test_idx])

    return var_train, var_test

def get_batches(x,y, batch_size=10,random=False):
    """ Return a generator that yields batches from arrays x and y. """
    #batch_size = len(x) // n_batches
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
            X,Y = x[sel_ind], y[sel_ind]
            # On the last batch, grab the rest of the data
        else:
            sel_ind = sel[ii:]
            X, Y  = x[sel_ind], y[sel_ind]
        yield X,Y

def get_batches_model(x,x2,y, batch_size=10,random=False):
    """ Return a generator that yields batches from arrays x and y. """
    #batch_size = len(x) // n_batches
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
            X,X2,Y = x[sel_ind],x2[sel_ind],y[sel_ind]
            # On the last batch, grab the rest of the data
        else:
            sel_ind = sel[ii:]
            X,X2,Y  = x[sel_ind],x2[sel_ind], y[sel_ind]
        yield X,X2,Y

def get_batches_model_2(x,x2,y,y2, batch_size=10,random=False):
    """ Return a generator that yields batches from arrays x and y. """
    #batch_size = len(x) // n_batches
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
            # On the last batch, grab the rest of the data
        else:
            sel_ind = sel[ii:]

        X, X2, Y, Y2 = x[sel_ind], x2[sel_ind], y[sel_ind], y2[sel_ind]
        yield X,X2,Y,Y2

def get_batches_gen(Vars, batch_size=10,random=False):
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

def get_motif_batches(x, batch_size=10,random=False):
    """ Return a generator that yields batches from arrays x and y. """
    #batch_size = len(x) // n_batches
    num_samples = x.shape[-1]
    if num_samples % batch_size == 0:
        n_batches = (num_samples // batch_size)
    else:
        n_batches = (num_samples // batch_size) + 1

    sel = np.asarray(list(range(num_samples)))
    if random is True:
        np.random.shuffle(sel)

    for ii in range(0, n_batches * batch_size, batch_size):
        # If we're not on the last batch, grab data with size batch_size
        if ii != (n_batches - 1) * batch_size:
            sel_ind=sel[ii: ii + batch_size]
            # On the last batch, grab the rest of the data
        else:
            sel_ind = sel[ii:]

        X = x[:,:,:,sel_ind]
        yield X

def get_batches_seq(x, batch_size=10,random=False):
    """ Return a generator that yields batches from arrays x and y. """
    #batch_size = len(x) // n_batches
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
            X = x[sel_ind]
            # On the last batch, grab the rest of the data
        else:
            sel_ind = sel[ii:]
            X   = x[sel_ind]
        yield X

def get_mers(X,mer=15,stride=7):

    Seq_ID = []
    Seq_Mers=[]
    for jj,seq in enumerate(X,0):
        for ii in range(0,len(seq),stride):
            if ii + mer <= len(seq):
                Seq_Mers.append(seq[ii:ii+mer])
                Seq_ID.append(jj)
            else:
                Seq_Mers.append(seq[len(seq)-mer:len(seq)])
                Seq_ID.append(jj)
                break

    return Seq_Mers, np.asarray(Seq_ID)

def Plot_Feature_Space(seq_f,freq,idx,ax,color_dict=None,classes=None,clustering=False,means=None,indicator_seq=None):
    if color_dict is None:
        N = len(idx)
        HSV_tuples = [(x * 1.0 / N, 1.0, 0.5) for x in range(N)]
        np.random.shuffle(HSV_tuples)
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        color_dict = dict(zip(range(len(idx)), RGB_tuples))


    if indicator_seq is not None:
        idx_sel = idx[0]
        idx_sel = np.invert(idx_sel)
        #ax.scatter(seq_f[idx_sel,0],seq_f[idx_sel,1],c='b',s=10000 * freq[idx_sel])
        ax.scatter(seq_f[idx_sel,0],seq_f[idx_sel,1],c='b',s=10)

        idx_sel = np.invert(idx_sel)
        #ax.scatter(seq_f[idx_sel,0],seq_f[idx_sel,1],c='y',s=10000 * freq[idx_sel])
        ax.scatter(seq_f[idx_sel,0],seq_f[idx_sel,1],c='y',s=10)


    else:
        for ii, (idx_sel, class_sel) in enumerate(zip(idx, classes), 0):
            if np.sum(idx_sel) != 0:
                #ax.scatter(seq_f[idx_sel, 0], seq_f[idx_sel, 1], c=color_dict[class_sel], s=10000 * freq[idx_sel])
                ax.scatter(seq_f[idx_sel, 0], seq_f[idx_sel, 1], c=color_dict[class_sel], s=10)



    if clustering is True:
        ax.scatter(means[-1][:, 0], means[-1][:, 1],c='r',s=100)

    return ax.get_xlim(),ax.get_ylim()

def t_stat(data, idx_logic):
    # means
    m = [np.mean(data[idx_logic]), np.mean(data[~idx_logic])]
    # variances
    v = [np.mean(np.square(data[idx_logic] - m[0])), np.mean(np.square(data[~idx_logic] - m[1]))]

    # difference in means
    out = list([m[1] - m[0]])
    # t_stat
    if out[0] == 0:
        out.append(0)
    else:
        out.append(out[0] / np.sqrt((v[0] / np.sum(idx_logic)) + (v[1] / np.sum(~idx_logic))))

    return out

def randperm_test(data, labels, func, n_perms=100):
    # get metrics with observed labels
    obs = np.apply_along_axis(t_stat, axis=0, arr=data, idx_logic=(labels == 0))

    # list to collect random permutation
    rnd = list()
    for i in range(n_perms):
        # get metrix with permuted labels
        rnd.append(np.apply_along_axis(t_stat, axis=0, arr=data, idx_logic=np.random.permutation(labels == 0)))
    # stack make to numpy
    rnd = np.stack(rnd, axis=2)

    # two sides empirical p-values, ie absolute change + or -
    p = np.mean(np.abs(rnd) >= np.abs(obs)[:, :, np.newaxis], axis=2)

    return obs, p

def Diff_Features(features,indices,sequences,type,p_val_threshold,idx_pos,idx_neg,directory_results,group,kernel):
    pos_mean = []
    neg_mean = []
    p_val = []
    feature_num = list(range(len(features.T)))
    for i in feature_num:
        pos = features[idx_pos, i]
        neg = features[idx_neg, i]
        pos_mean.append(np.mean(pos))
        neg_mean.append(np.mean(neg))
        try:
            stat, p = mannwhitneyu(pos, neg)
            p_val.append(p)
        except:
            p_val.append(1.0)

    df_features = pd.DataFrame()
    df_features['Feature'] = feature_num
    df_features['P_Val'] = p_val
    df_features['Pos'] = pos_mean
    df_features['Neg'] = neg_mean
    df_features['Mag'] = df_features['Pos'] - df_features['Neg']

    df_features = df_features[df_features['P_Val'] < p_val_threshold]

    df_features.sort_values(by='Mag', inplace=True, ascending=False)

    # Get motifs for positive features
    dir = os.path.join(directory_results, group + '_'+type+'_SS_Motifs')
    if not os.path.exists(dir):
        os.makedirs(dir)

    file_list = [f for f in os.listdir(dir)]
    [os.remove(os.path.join(dir, f)) for f in file_list]

    top_seq = 10
    seq_cluster = []
    feature_keep = []
    for feature in df_features['Feature'].tolist():
        if df_features['Mag'][feature] > 0:
            feature_keep.append(feature)
            sel = np.flip(features[:, feature].argsort(), -1)
            sel = sel[0:top_seq]
            seq_sel = sequences[sel]
            ind_sel = indices[sel, feature]
            seq_cluster.append(seq_sel)

            motifs = []
            for ii, i in enumerate(ind_sel, 0):
                motif = seq_sel[ii][int(i):int(i) + kernel]
                if len(motif) < kernel:
                    motif = motif + 'X' * (kernel - len(motif))
                motif = motif.lower()
                motif = SeqRecord(Seq(motif, IUPAC.protein), str(ii))
                motifs.append(motif)

            SeqIO.write(motifs, os.path.join(dir, 'feature_') + str(feature) + '.fasta', 'fasta')

    seq_features_df_pos = pd.DataFrame()
    for ii, f in enumerate(feature_keep, 0):
        seq_features_df_pos[f] = seq_cluster[ii]

    return seq_features_df_pos

def Diff_Features_WF(features_wf,features_seq,indices,sequences,X_Freq,labels,kernel,idx_pos,idx_neg,directory_results,
                     p_val_threshold,cut,type,group,save_images,lb):

    pos_mean = []
    neg_mean = []
    p_val = []
    feature_num = list(range(len(features_wf.T)))
    for i in feature_num:
        pos = features_wf[idx_pos, i]
        neg = features_wf[idx_neg, i]
        pos_mean.append(np.mean(pos))
        neg_mean.append(np.mean(neg))
        try:
            stat, p = mannwhitneyu(pos, neg)
            p_val.append(p)
        except:
            p_val.append(1.0)

    df_features = pd.DataFrame()
    df_features['Feature'] = feature_num
    df_features['P_Val'] = p_val
    df_features['Pos'] = pos_mean
    df_features['Neg'] = neg_mean
    df_features['Mag'] = df_features['Pos'] - df_features['Neg']
    df_features = df_features[df_features['P_Val'] < p_val_threshold]

    df_features.sort_values(by='Mag', inplace=True, ascending=False)

    features_seq = np.reshape(features_seq, [-1, features_seq.shape[-1]])
    sequences = np.asarray(np.hstack(sequences).tolist())
    indices = np.asarray(np.reshape(indices, [-1, indices.shape[-1]]))

    # Get motifs for positive features
    dir = os.path.join(directory_results, group + '_' + type + '_WF_Motifs')
    if not os.path.exists(dir):
        os.makedirs(dir)

    file_list = [f for f in os.listdir(dir)]
    [os.remove(os.path.join(dir, f)) for f in file_list]

    top_seq = 10
    seq_cluster = []
    feature_keep = []
    seq_thresh_pos = []
    for feature in df_features['Feature'].tolist():
        if df_features['Mag'][feature] > 0:
            feature_keep.append(feature)
            sel = np.flip(features_seq[:, feature].argsort(), -1)
            feature_sort = features_seq[sel, feature]
            seq_thresh_pos.append(np.percentile(feature_sort, cut))
            sel = sel[0:top_seq]
            seq_sel = sequences[sel]
            ind_sel = indices[sel, feature]
            seq_cluster.append(seq_sel)

            motifs = []
            for ii, i in enumerate(ind_sel, 0):
                motif = seq_sel[ii][int(i):int(i) + kernel]
                if len(motif) < kernel:
                    motif = motif + 'X' * (kernel - len(motif))
                motif = motif.lower()
                motif = SeqRecord(Seq(motif, IUPAC.protein), str(ii))
                motifs.append(motif)

            SeqIO.write(motifs, os.path.join(dir, 'feature_') + str(feature) + '.fasta', 'fasta')

    seq_features_df_pos = pd.DataFrame()
    for ii, f in enumerate(feature_keep, 0):
        seq_features_df_pos[f] = seq_cluster[ii]

    seq_features_df_pos.to_csv(os.path.join(dir, 'feature_sequences.csv'), index=False)

    if save_images is True:
        for feature, thresh in zip(feature_keep, seq_thresh_pos):
            labels = []
            values = []
            for g in lb.classes_:
                sel = labels == g
                features_seq_sel = (features_seq[sel, :, feature] > thresh) * X_Freq[sel]
                features_contrib = list(np.sum(features_seq_sel, -1))
                labels += [g] * np.sum(sel)
                values += features_contrib

            df = pd.DataFrame()
            df['Cohort'] = labels
            df['Fraction of Response'] = values
            plt.figure()
            sns.set(font_scale=1)
            sns.violinplot(data=df, x='Cohort', y='Fraction of Response')
            plt.title('Feature ' + str(feature))
            plt.savefig(os.path.join(dir, 'feature') + str(feature) + '.tif')
            plt.close()

    return seq_features_df_pos

def pad_sequences(sequences,num_seq_per_instance):
    for ii, sample in enumerate(sequences):
        if len(sample) > num_seq_per_instance:
            sequences[ii] = sample[0:num_seq_per_instance]
        elif len(sample) < num_seq_per_instance:
            sequences[ii] = sample + ['null'] * (num_seq_per_instance - len(sample))

    return sequences

def pad_freq(freq,num_seq_per_instance):
    for ii, sample in enumerate(freq):
        if len(sample) > num_seq_per_instance:
            freq[ii] = np.asarray(freq[ii][0:num_seq_per_instance])
        elif len(sample) < num_seq_per_instance:
            freq[ii] = np.pad(freq[ii], (0, num_seq_per_instance - len(sample)), mode='constant')

    return freq
