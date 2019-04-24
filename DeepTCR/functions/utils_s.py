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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf
from multiprocessing import Pool
from DeepTCR.functions.data_processing import *
from sklearn.model_selection import train_test_split

def custom_train_test_split(X,Y,test_size,stratify):
    idx = np.array(range(len(X)))

    num_per_class = test_size//len(np.unique(stratify))
    idx_test = []
    for i in np.unique(stratify):
        idx_test.append(np.random.choice(np.where(stratify == i)[0],num_per_class,replace=False))

    idx_left = np.setdiff1d(idx,np.hstack(idx_test))
    if len(np.hstack(idx_test)) < test_size:
        diff = test_size - len(np.hstack(idx_test))
        idx_test.append(np.random.choice(idx_left,diff,replace=False))

    idx_test = np.hstack(idx_test)
    idx_train = np.setdiff1d(idx,idx_test)

    X_train = X[idx_train]
    Y_train = Y[idx_train]

    X_test = X[idx_test]
    Y_test = Y[idx_test]

    return X_train,X_test,Y_train,Y_test

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
                idx = np.asarray(list(range(len(Y))))
                if LOO == 1:
                    test_idx = np.random.choice(idx, LOO, replace=False)[0]
                    train_idx = np.setdiff1d(idx, test_idx)
                    l_t = np.argmax(Y[test_idx],0)
                elif LOO < Y.shape[1]:
                    test_idx = np.random.choice(idx, LOO, replace=False)
                    train_idx = np.setdiff1d(idx, test_idx)
                else:
                    train_idx,test_idx,Y_train,_ = custom_train_test_split(idx,Y,test_size=LOO,stratify=np.argmax(Y,1))

                if LOO == 1:
                    valid_idx = np.random.choice(train_idx[np.argmax(Y[train_idx],1)==l_t],LOO, replace=False)[0]
                    train_idx = np.setdiff1d(train_idx, valid_idx)
                elif LOO < Y.shape[1]:
                    valid_idx = np.random.choice(train_idx, LOO, replace=False)
                    train_idx = np.setdiff1d(train_idx, valid_idx)
                else:
                    train_idx,valid_idx,_,_ = custom_train_test_split(train_idx,Y_train,test_size=LOO,stratify=np.argmax(Y_train,1))

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

def Get_Train_Valid_Test_KFold(Vars,test_idx,valid_idx,train_idx,Y=None):
    var_train = []
    var_valid = []
    var_test = []
    for var in Vars:
        var_train.append(var[train_idx])
        var_valid.append(var[valid_idx])
        var_test.append(var[test_idx])

    if Y is not None:
        var_train.append(Y[train_idx])
        var_valid.append(Y[valid_idx])
        var_test.append(Y[test_idx])

    return var_train, var_valid, var_test

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

def Diff_Features(features,indices,sequences,type,sample_id,p_val_threshold,
                  idx_pos,idx_neg,directory_results,group,kernel,sample_avg,top_seq):
    pos_mean = []
    neg_mean = []
    p_val = []
    feature_num = list(range(len(features.T)))
    for i in feature_num:
        if sample_avg is False:
            pos = features[idx_pos, i]
            neg = features[idx_neg, i]
        else:
            df_temp = pd.DataFrame()
            df_temp['pos'] = features[idx_pos, i]
            df_temp['sample_id'] = sample_id[idx_pos]
            df_temp = df_temp.groupby(['sample_id']).agg({'pos':'mean'})
            pos = np.asarray(df_temp['pos'].tolist())

            df_temp = pd.DataFrame()
            df_temp['neg'] = features[idx_neg, i]
            df_temp['sample_id'] = sample_id[idx_neg]
            df_temp = df_temp.groupby(['sample_id']).agg({'neg':'mean'})
            neg = np.asarray(df_temp['neg'].tolist())

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
    dir = os.path.join(directory_results, group + '_'+type+'_Motifs')
    if not os.path.exists(dir):
        os.makedirs(dir)

    file_list = [f for f in os.listdir(dir)]
    [os.remove(os.path.join(dir, f)) for f in file_list]

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

            mag_write = str(np.around(df_features['Mag'][feature],3))
            SeqIO.write(motifs, os.path.join(dir, str(mag_write)+'_feature_') + str(feature) +'.fasta', 'fasta')

    seq_features_df_pos = pd.DataFrame()

    for ii, f in enumerate(feature_keep, 0):
        seq_features_df_pos[f] = seq_cluster[ii]

    return seq_features_df_pos

def Run_Graph_SS(set,sess,self,GO,batch_size,random=True,train=True,drop_out_rate=None):
    loss = []
    accuracy = []
    predicted_list = []
    var_names = ['X_Seq_alpha','X_Seq_beta','v_beta_num','d_beta_num',
                 'j_beta_num','v_alpha_num','j_alpha_num','hla_data_seq_num']
    Vars = []
    for v in var_names:
        Vars.append(set[self.var_dict[v]])
    Vars.append(set[-1])

    for vars in get_batches(Vars, batch_size=batch_size, random=random):
        feed_dict = {GO.Y: vars[-1]}

        if drop_out_rate is not None:
            feed_dict[GO.prob] = drop_out_rate

        if self.use_alpha is True:
            feed_dict[GO.X_Seq_alpha] = vars[0]
        if self.use_beta is True:
            feed_dict[GO.X_Seq_beta] = vars[1]

        if self.use_v_beta is True:
            feed_dict[GO.X_v_beta] = vars[2]

        if self.use_d_beta is True:
            feed_dict[GO.X_d_beta] = vars[3]

        if self.use_j_beta is True:
            feed_dict[GO.X_j_beta] = vars[4]

        if self.use_v_alpha is True:
            feed_dict[GO.X_v_alpha] = vars[5]

        if self.use_j_alpha is True:
            feed_dict[GO.X_j_alpha] = vars[6]

        if self.use_hla:
            feed_dict[GO.X_hla] = vars[7]

        if train is True:
            loss_i, accuracy_i, _, predicted_i = sess.run([GO.loss, GO.accuracy, GO.opt, GO.predicted], feed_dict=feed_dict)
        else:
            loss_i, accuracy_i, predicted_i = sess.run([GO.loss, GO.accuracy, GO.predicted], feed_dict=feed_dict)

        loss.append(loss_i)
        accuracy.append(accuracy_i)
        predicted_list.append(predicted_i)

    loss = np.mean(loss)
    accuracy = np.mean(accuracy)
    predicted_out = np.vstack(predicted_list)
    try:
        auc = roc_auc_score(set[-1], predicted_out)
    except:
        auc = 0.0
    return loss,accuracy,predicted_out,auc

def Run_Graph_WF(set,sess,self,GO,batch_size,random=True,train=True,drop_out_rate=None):
    loss = []
    accuracy = []
    predicted_list = []
    for vars in get_batches(set, batch_size=batch_size, random=random):
        var_idx = np.where(np.isin(self.sample_id, vars[0]))[0]
        lb = LabelEncoder()
        lb.fit(vars[0])
        _,_,sample_idx = np.intersect1d(lb.classes_,vars[0],return_indices=True)
        vars = [v[sample_idx] for v in vars]
        i = lb.transform(self.sample_id[var_idx])

        OH = OneHotEncoder(categories='auto')
        sp = OH.fit_transform(i.reshape(-1, 1)).T
        sp = sp.tocoo()
        indices = np.mat([sp.row, sp.col]).T
        sp = tf.SparseTensorValue(indices, sp.data, sp.shape)

        feed_dict = {GO.Y: vars[-1],
                     GO.X_Freq: self.freq[var_idx],
                     GO.sp: sp}

        if drop_out_rate is not None:
            feed_dict[GO.prob] = drop_out_rate

        if self.use_alpha is True:
            feed_dict[GO.X_Seq_alpha] = self.X_Seq_alpha[var_idx]
        if self.use_beta is True:
            feed_dict[GO.X_Seq_beta] = self.X_Seq_beta[var_idx]

        if self.use_v_beta is True:
            feed_dict[GO.X_v_beta] = self.v_beta_num[var_idx]

        if self.use_d_beta is True:
            feed_dict[GO.X_d_beta] = self.d_beta_num[var_idx]

        if self.use_j_beta is True:
            feed_dict[GO.X_j_beta] = self.j_beta_num[var_idx]

        if self.use_v_alpha is True:
            feed_dict[GO.X_v_alpha] = self.v_alpha_num[var_idx]

        if self.use_j_alpha is True:
            feed_dict[GO.X_j_alpha] = self.j_alpha_num[var_idx]

        if self.use_hla:
            feed_dict[GO.X_hla] = self.hla_data_seq_num[var_idx]

        if train is True:
            loss_i, accuracy_i, _, predicted_i = sess.run([GO.loss, GO.accuracy, GO.opt, GO.predicted],
                                                          feed_dict=feed_dict)
        else:
            loss_i, accuracy_i, predicted_i = sess.run([GO.loss, GO.accuracy, GO.predicted],
                                                       feed_dict=feed_dict)

        loss.append(loss_i)
        accuracy.append(accuracy_i)
        pred_temp = np.zeros_like(predicted_i)
        pred_temp[sample_idx] = predicted_i
        predicted_i = pred_temp
        predicted_list.append(predicted_i)

    loss = np.mean(loss)
    accuracy = np.mean(accuracy)
    predicted_out = np.vstack(predicted_list)
    try:
        auc = roc_auc_score(set[-1], predicted_out)
    except:
        auc = 0.0
    return loss,accuracy,predicted_out,auc

def Get_Seq_Features_Indices(self,batch_size,GO,sess):
    alpha_features_list = []
    beta_features_list = []
    alpha_indices_list = []
    beta_indices_list = []
    Vars = [self.X_Seq_alpha, self.X_Seq_beta]
    for vars in get_batches(Vars, batch_size=batch_size, random=False):
        feed_dict = {}
        if self.use_alpha is True:
            feed_dict[GO.X_Seq_alpha] = vars[0]
        if self.use_beta is True:
            feed_dict[GO.X_Seq_beta] = vars[1]

        if self.use_alpha is True:
            features_i_alpha, indices_i_alpha = sess.run([GO.alpha_out, GO.indices_alpha], feed_dict=feed_dict)
            alpha_features_list.append(features_i_alpha)
            alpha_indices_list.append(indices_i_alpha)

        if self.use_beta is True:
            features_i_beta, indices_i_beta = sess.run([GO.beta_out, GO.indices_beta], feed_dict=feed_dict)
            beta_features_list.append(features_i_beta)
            beta_indices_list.append(indices_i_beta)

    if self.use_alpha is True:
        self.alpha_features = np.vstack(alpha_features_list)
        self.alpha_indices = np.vstack(alpha_indices_list)

    if self.use_beta is True:
        self.beta_features = np.vstack(beta_features_list)
        self.beta_indices = np.vstack(beta_indices_list)

def Get_Sequence_Pred(self,batch_size,GO,sess):
    predicted_list = []
    i = np.asarray(range(len(self.Y)))
    freq = np.ones_like(self.freq)
    idx = []
    for vars in get_batches(self.test, batch_size=batch_size, random=False):
        var_idx = np.where(np.isin(self.sample_id, vars[0]))[0]
        OH = OneHotEncoder(categories='auto')
        sp = OH.fit_transform(i[var_idx].reshape(-1, 1)).T
        sp = sp.tocoo()
        indices = np.mat([sp.row, sp.col]).T
        sp = tf.SparseTensorValue(indices, sp.data, sp.shape)

        feed_dict = {GO.X_Freq: freq[var_idx],
                     GO.sp: sp}

        if self.use_alpha is True:
            feed_dict[GO.X_Seq_alpha] = self.X_Seq_alpha[var_idx]
        if self.use_beta is True:
            feed_dict[GO.X_Seq_beta] = self.X_Seq_beta[var_idx]

        if self.use_v_beta is True:
            feed_dict[GO.X_v_beta] = self.v_beta_num[var_idx]

        if self.use_d_beta is True:
            feed_dict[GO.X_d_beta] = self.d_beta_num[var_idx]

        if self.use_j_beta is True:
            feed_dict[GO.X_j_beta] = self.j_beta_num[var_idx]

        if self.use_v_alpha is True:
            feed_dict[GO.X_v_alpha] = self.v_alpha_num[var_idx]

        if self.use_j_alpha is True:
            feed_dict[GO.X_j_alpha] = self.j_alpha_num[var_idx]

        if self.use_hla:
            feed_dict[GO.X_hla] = self.hla_data_seq_num[var_idx]

        predicted_list.append(sess.run(GO.predicted,feed_dict=feed_dict))
        idx.append(var_idx)

    return np.vstack(predicted_list),np.squeeze(np.hstack(idx))

def Get_Latent_Features(self,batch_size,GO,sess):
    Vars = [self.X_Seq_alpha, self.X_Seq_beta,self.v_beta_num,self.d_beta_num,self.j_beta_num,
            self.v_alpha_num,self.v_alpha_num,self.hla_data_seq_num]
    Features = []
    Features_c = []
    for vars in get_batches(Vars, batch_size=batch_size, random=False):
        feed_dict = {}
        if self.use_alpha is True:
            feed_dict[GO.X_Seq_alpha] = vars[0]
        if self.use_beta is True:
            feed_dict[GO.X_Seq_beta] = vars[1]

        if self.use_v_beta is True:
            feed_dict[GO.X_v_beta] = vars[2]

        if self.use_d_beta is True:
            feed_dict[GO.X_d_beta] = vars[3]

        if self.use_j_beta is True:
            feed_dict[GO.X_j_beta] = vars[4]

        if self.use_v_alpha is True:
            feed_dict[GO.X_v_alpha] = vars[5]

        if self.use_j_alpha is True:
            feed_dict[GO.X_j_alpha] = vars[6]

        if self.use_hla:
            feed_dict[GO.X_hla] = vars[7]

        Features.append(sess.run(GO.Features,feed_dict=feed_dict))
        if GO.on_graph_clustering is True:
            Features_c.append(sess.run(GO.Features_c,feed_dict=feed_dict))

    Features = np.vstack(Features)
    if GO.on_graph_clustering is True:
        Features_c = np.vstack(Features_c)

    return Features, Features_c

def inference_method_ss(get,alpha_sequences,beta_sequences,v_beta,d_beta,j_beta,v_alpha,j_alpha,hla,
                     p,batch_size,self):

    inputs = [alpha_sequences, beta_sequences, v_beta, d_beta, j_beta, v_alpha, j_alpha,hla]
    for i in inputs:
        if i is not None:
            len_input = len(i)
            break

    if p is None:
        p = Pool(40)

    if alpha_sequences is not None:
        args = list(
            zip(alpha_sequences, [self.aa_idx] * len(alpha_sequences), [self.max_length] * len(alpha_sequences)))
        result = p.starmap(Embed_Seq_Num, args)
        sequences_num = np.vstack(result)
        X_Seq_alpha = np.expand_dims(sequences_num, 1)
    else:
        X_Seq_alpha = np.zeros(shape=[len_input])
        alpha_sequences = np.asarray([None] * len_input)

    if beta_sequences is not None:
        args = list(
            zip(beta_sequences, [self.aa_idx] * len(beta_sequences), [self.max_length] * len(beta_sequences)))
        result = p.starmap(Embed_Seq_Num, args)
        sequences_num = np.vstack(result)
        X_Seq_beta = np.expand_dims(sequences_num, 1)
    else:
        X_Seq_beta = np.zeros(shape=[len_input])
        beta_sequences = np.asarray([None] * len_input)

    if v_beta is not None:
        v_beta = v_beta.astype(self.lb_v_beta.classes_.dtype)
        i_r = np.where(np.invert(np.isin(v_beta, self.lb_v_beta.classes_)))[0]
        v_beta[i_r] = np.random.choice(self.lb_v_beta.classes_, len(i_r))
        v_beta_num = self.lb_v_beta.transform(v_beta)
    else:
        v_beta_num = np.zeros(shape=[len_input])
        v_beta = np.asarray([None] * len_input)

    if d_beta is not None:
        d_beta = d_beta.astype(self.lb_d_beta.classes_.dtype)
        i_r = np.where(np.invert(np.isin(d_beta, self.lb_d_beta.classes_)))[0]
        d_beta[i_r] = np.random.choice(self.lb_d_beta.classes_, len(i_r))
        d_beta_num = self.lb_d_beta.transform(d_beta)
    else:
        d_beta_num = np.zeros(shape=[len_input])
        d_beta = np.asarray([None] * len_input)

    if j_beta is not None:
        j_beta = j_beta.astype(self.lb_j_beta.classes_.dtype)
        i_r = np.where(np.invert(np.isin(j_beta, self.lb_j_beta.classes_)))[0]
        j_beta[i_r] = np.random.choice(self.lb_j_beta.classes_, len(i_r))
        j_beta_num = self.lb_j_beta.transform(j_beta)
    else:
        j_beta_num = np.zeros(shape=[len_input])
        j_beta = np.asarray([None] * len_input)

    if v_alpha is not None:
        v_alpha = v_alpha.astype(self.lb_v_alpha.classes_.dtype)
        i_r = np.where(np.invert(np.isin(v_alpha, self.lb_v_alpha.classes_)))[0]
        v_alpha[i_r] = np.random.choice(self.lb_v_alpha.classes_, len(i_r))
        v_alpha_num = self.lb_v_alpha.transform(v_alpha)
    else:
        v_alpha_num = np.zeros(shape=[len_input])
        v_alpha = np.asarray([None] * len_input)

    if j_alpha is not None:
        j_alpha = j_alpha.astype(self.lb_j_alpha.classes_.dtype)
        i_r = np.where(np.invert(np.isin(j_alpha, self.lb_j_alpha.classes_)))[0]
        j_alpha[i_r] = np.random.choice(self.lb_j_alpha.classes_, len(i_r))
        j_alpha_num = self.lb_j_alpha.transform(j_alpha)
    else:
        j_alpha_num = np.zeros(shape=[len_input])
        j_alpha = np.asarray([None] * len_input)

    if hla is not None:
        hla_data_seq_num = self.lb_hla.transform(hla)
    else:
        hla_data_seq_num = np.zeros(shape=[len_input])

    if p is None:
        p.close()
        p.join()

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(os.path.join(self.Name, 'model', 'model.ckpt.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(self.Name, 'model')))
        graph = tf.get_default_graph()

        if self.use_alpha is True:
            X_Seq_alpha_v = graph.get_tensor_by_name('Input_Alpha:0')

        if self.use_beta is True:
            X_Seq_beta_v = graph.get_tensor_by_name('Input_Beta:0')

        if self.use_v_beta is True:
            X_v_beta = graph.get_tensor_by_name('Input_V_Beta:0')

        if self.use_d_beta is True:
            X_d_beta = graph.get_tensor_by_name('Input_D_Beta:0')

        if self.use_j_beta is True:
            X_j_beta = graph.get_tensor_by_name('Input_J_Beta:0')

        if self.use_v_alpha is True:
            X_v_alpha = graph.get_tensor_by_name('Input_V_Alpha:0')

        if self.use_j_alpha is True:
            X_j_alpha = graph.get_tensor_by_name('Input_J_Alpha:0')

        if self.use_hla:
            X_hla = graph.get_tensor_by_name('HLA:0')

        get_obj = graph.get_tensor_by_name(get)

        out_list = []
        Vars = [X_Seq_alpha, X_Seq_beta, v_beta_num, d_beta_num, j_beta_num,
                v_alpha_num, j_alpha_num,hla_data_seq_num]

        for vars in get_batches(Vars, batch_size=batch_size):
            feed_dict = {}
            if self.use_alpha is True:
                feed_dict[X_Seq_alpha_v] = vars[0]
            if self.use_beta is True:
                feed_dict[X_Seq_beta_v] = vars[1]

            if self.use_v_beta is True:
                feed_dict[X_v_beta] = vars[2]

            if self.use_d_beta is True:
                feed_dict[X_d_beta] = vars[3]

            if self.use_j_beta is True:
                feed_dict[X_j_beta] = vars[4]

            if self.use_v_alpha is True:
                feed_dict[X_v_alpha] = vars[5]

            if self.use_j_alpha is True:
                feed_dict[X_j_alpha] = vars[6]

            if self.use_hla:
                feed_dict[X_hla] = vars[7]

            get_ind = sess.run(get_obj, feed_dict=feed_dict)
            out_list.append(get_ind)

        return np.vstack(out_list)


def stop_check(loss,stop_criterion,stop_criterion_window):
    w = loss[-stop_criterion_window:]
    return (w[0]-w[-1])/w[0] < stop_criterion





