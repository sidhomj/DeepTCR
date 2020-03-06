import tensorflow as tf
import numpy as np

def isru(x, l=-1, h=1, a=None, b=None, name='isru', axis=-1):
    lim = 4
    if a is None:
        _a = h - l
    else:
        _a = tf.Variable(name=name + '_a', initial_value=np.ones(np.array([_.value for _ in x.shape])[axis]) + a, trainable=True, dtype=tf.float32)
        _a = 2 ** isru(_a, l=-lim, h=lim)

    if b is None:
        _b = 1
    else:
        _b = tf.Variable(name=name + '_b', initial_value=np.zeros(np.array([_.value for _ in x.shape])[axis]) + b, trainable=True, dtype=tf.float32)
        _b = (2 ** isru(_b, l=-lim, h=lim))+1

    return l + (((h - l) / 2) * (1 + (x * ((_a + ((x ** 2) ** _b)) ** -(1 / (2 * _b))))))

# def isru(x,l=-1,h=1,a=0,b=0,name='isru',axis=-1):
#     _a = tf.Variable(name=name + '_a', initial_value=np.zeros(np.array([_.value for _ in x.shape])[axis]) + a, trainable=True, dtype=tf.float32)
#     _a = tf.exp(_a)
#     _b = tf.Variable(name=name + '_b', initial_value=np.zeros(np.array([_.value for _ in x.shape])[axis]) + b, trainable=True, dtype=tf.float32)
#     _b = tf.exp(_b)
#
#     x_2 = x ** 2
#     lower_sqrt = (_a + x_2) ** (1 / 2)
#     upper_sqrt = (_b + x_2) ** (1 / 2)
#     return l + ((h - l) * ((x + lower_sqrt) / (lower_sqrt + upper_sqrt)))

def MultiLevel_Dropout(X,num_masks=2,activation=tf.nn.relu,use_bias=True,
                       rate=0.2,units=12,name='ml_weights',reg=0.0):
    out = []
    for i in range(num_masks):
        fc = tf.layers.dropout(X,rate=rate)
        if i==0:
            reuse=False
        else:
            reuse=True

        with tf.variable_scope(name,reuse=reuse):
            out.append(tf.layers.dense(fc,units=units,activation=activation,use_bias=use_bias,
                                       kernel_regularizer=tf.contrib.layers.l1_regularizer(reg)))

    return tf.reduce_mean(tf.stack(out),0)

def MIL_Layer(features,num_classes,num_concepts,sp,freq=None,prob=0.0,num_layers=1,units_fc=12):

    alpha=0.00
    w_list = []
    quant = []
    for i in range(num_concepts):
        #weights for each instance are learned
        w_temp = features
        for n in range(num_layers-1):
            w_temp = tf.layers.dropout(w_temp, prob)
            w_temp = tf.layers.dense(w_temp, units_fc)

        w_temp = tf.layers.dropout(w_temp,prob)
        w_temp = tf.layers.dense(w_temp, 1, lambda x: isru(x, l=0, h=1, a=0, b=0),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(alpha))
        w_list.append(w_temp)

        #weights are used against sparse matrix
        sp_temp = sp * tf.squeeze(w_temp, -1)
        if freq is not None:
            sp_temp = sp_temp*freq
        sum = tf.compat.v1.sparse.reduce_sum(sp_temp, 1)
        sum.set_shape([sp_temp.shape[1], ])
        sum = tf.expand_dims(sum, -1)
        quant.append(sum)

    quant = tf.concat(quant,1)
    num_log_layers = 1
    num_log_units = 12
    logits = quant
    logits = tf.layers.dropout(logits,prob)
    alpha=0.0
    for i in range(num_log_layers):
        if (i == 0) and (num_log_layers != 1):
            logits = tf.layers.dense(logits, num_log_units,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(alpha),
                                     activation=tf.nn.relu)
        elif (i == 0) and (num_log_layers == 1):
            logits = tf.layers.dense(logits, num_classes,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(alpha))
        elif i == num_log_layers-1:
            logits = tf.layers.dense(logits,num_classes)
        else:
            logits = tf.layers.dense(logits,num_log_units,tf.nn.relu)

    w = tf.squeeze(tf.transpose(tf.stack(w_list), perm=[1, 0, 2]), -1)
    return logits,w
