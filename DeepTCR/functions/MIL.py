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

def MultiLevel_Dropout(X,num_masks=2,activation=tf.nn.relu,use_bias=True,
                       rate=0.2,units=12,name='ml_weights'):
    out = []
    for i in range(num_masks):
        fc = tf.layers.dropout(X,rate=rate)
        if i==0:
            reuse=False
        else:
            reuse=True

        with tf.variable_scope(name,reuse=reuse):
            out.append(tf.layers.dense(fc,units=units,activation=activation,use_bias=use_bias))

    return tf.reduce_mean(tf.stack(out),0)

def MIL_Layer(features,num_classes,sp,freq=None,prob=0.0,num_layers=1):

    features_list = []
    w_list = []
    quant = []
    for i in range(num_classes):
        #weights for each instance are learned
        w_temp = features
        for n in range(num_layers-1):
            w_temp = tf.layers.dropout(w_temp, prob)
            w_temp = tf.layers.dense(w_temp, 1, tf.nn.relu)

        w_temp = tf.layers.dropout(w_temp,prob)
        w_temp = tf.layers.dense(w_temp, 1, lambda x: isru(x, l=0, h=1, a=0, b=0))
        w_list.append(w_temp)

        #weights are used against sparse matrix
        sp_temp = sp * tf.squeeze(w_temp, -1)
        if freq is not None:
            sp_temp = sp_temp*freq
        sum = tf.compat.v1.sparse.reduce_sum(sp_temp, 1)
        sum.set_shape([sp_temp.shape[1], ])
        sum = tf.expand_dims(sum, -1)
        features_list.append(tf.sparse.matmul(sp_temp, features) / sum)
        quant.append(sum)

    features_out = tf.stack(features_list)
    features_out = tf.transpose(features_out, perm=[1, 0, 2])

    quant = tf.concat(quant,1)[:,:,tf.newaxis]
    quant = tf.layers.dense(quant,features_out.shape[2],tf.nn.relu)
    quant = tf.layers.dense(quant,features_out.shape[2],tf.nn.relu)
    quant = tf.layers.dense(quant,features_out.shape[2],tf.nn.relu)


    features_out = tf.concat((quant,features_out),2)

    features_out = tf.layers.dropout(features_out,prob)
    logits = tf.squeeze(tf.layers.dense(features_out, 1), -1)

    #logits = tf.squeeze(MultiLevel_Dropout(features_out,num_masks=12,units=1,activation=None, rate=0.5,name='ml_2'),-1)

    w = tf.squeeze(tf.transpose(tf.stack(w_list), perm=[1, 0, 2]), -1)
    return logits,w

def MIL_Layer_2(features,num_classes,sp,freq=None):

    if freq is not None:
        features = tf.concat((features,freq[:,tf.newaxis]),1)

    features_list = []
    w_list = []
    for i in range(num_classes):
        #weights for each instance are learned
        w_temp = tf.layers.dense(features, 1, lambda x: isru(x, l=0, h=1, a=0, b=0))
        w_list.append(w_temp)

        #weights are used against sparse matrix
        sp_temp = sp * tf.squeeze(w_temp, -1)
        sum = tf.compat.v1.sparse.reduce_sum(sp_temp, 1)
        sum.set_shape([sp_temp.shape[1], ])
        sum = tf.expand_dims(sum, -1)
        features_list.append(tf.sparse.matmul(sp_temp, features) / sum)

    features_out = tf.stack(features_list)
    features_out = tf.transpose(features_out, perm=[1, 0, 2])
    logits = tf.squeeze(tf.layers.dense(features_out, 1), -1)
    w = tf.squeeze(tf.transpose(tf.stack(w_list), perm=[1, 0, 2]), -1)
    return logits,w