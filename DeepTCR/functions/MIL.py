import tensorflow as tf
import numpy as np

def isru(x, l=-1, h=1, a=None, b=None, name='isru', axis=-1):
    if a is None:
        _a = h - l
    else:
        _a = tf.Variable(name=name + '_a', initial_value=np.zeros(np.array([_.value for _ in x.shape])[axis]) + a, trainable=True, dtype=tf.float32)
        _a = 2 ** isru(_a, l=-4, h=4)

    if b is None:
        _b = 1
    else:
        _b = tf.Variable(name=name + '_b', initial_value=np.zeros(np.array([_.value for _ in x.shape])[axis]) + b, trainable=True, dtype=tf.float32)
        _b = 2 ** isru(_b, l=-4, h=4)

    return l + (((h - l) / 2) * (1 + (x * ((_a + ((x ** 2) ** _b)) ** -(1 / (2 * _b))))))

def MIL_Layer(features,num_classes,sp):

    features_list = []
    w_list = []
    for i in range(num_classes):
        #weights for each instance are learned
        w_temp = tf.layers.dense(features, 1, lambda x: isru(x, l=0, h=1, a=0, b=0))
        w_list.append(w_temp)

        #weights are used against sparse matrix
        sp_temp = sp * tf.squeeze(w_temp, -1)
        sum = tf.compat.v1.sparse.reduce_sum(sp, 1)
        sum.set_shape([sp.shape[1], ])
        sum = tf.expand_dims(sum, -1)
        features_list.append(tf.sparse.matmul(sp_temp, features) / sum)

    features_out = tf.stack(features_list)
    features_out = tf.transpose(features_out, perm=[1, 0, 2])
    logits = tf.squeeze(tf.layers.dense(features_out, 1), -1)
    w = tf.squeeze(tf.transpose(tf.stack(w_list), perm=[1, 0, 2]), -1)
    return logits,w