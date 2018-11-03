import numpy as np
import colorsys
import matplotlib.pyplot as plt

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