import numpy as np

def Get_Train_Valid_Test(Vars,Y=None,test_size=0.25,regression=False):

    if regression is False:
        var_train = []
        var_valid = []
        var_test = []
        if Y is not None:
            y_label = np.argmax(Y,1)
            classes = list(set(y_label))

            for ii,type in enumerate(classes,0):
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
                    for jj,var in enumerate(Vars,0):
                        var_train[jj] = np.concatenate((var_train[jj],var[train_idx]),0)
                        var_valid[jj] = np.concatenate((var_valid[jj],var[valid_idx]),0)
                        var_test[jj] = np.concatenate((var_test[jj],var[test_idx]),0)

                    var_train[-1] = np.concatenate((var_train[-1],Y[train_idx]),0)
                    var_valid[-1] = np.concatenate((var_valid[-1],Y[valid_idx]),0)
                    var_test[-1] = np.concatenate((var_test[-1],Y[test_idx]),0)
        else:
            idx = np.asarray(list(range(len(Vars[0]))))
            np.random.shuffle(idx)
            train_idx = np.random.choice(idx, int((1 - test_size) * idx.shape[0]), replace=False)
            idx = np.setdiff1d(idx, train_idx)
            np.random.shuffle(idx)
            half_val_len = int(idx.shape[0] * 0.5)
            valid_idx, test_idx = idx[:half_val_len], idx[half_val_len:]

            for var in Vars:
                var_train.append(var[train_idx])
                var_valid.append(var[valid_idx])
                var_test.append(var[test_idx])

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
            X,X2,Y = x[sel_ind],x2[sel_ind], y[sel_ind]
            # On the last batch, grab the rest of the data
        else:
            sel_ind = sel[ii:]
            X,X2, Y  = x[sel_ind],x2[sel_ind], y[sel_ind]
        yield X,X2,Y

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


