import tensorflow as tf
import numpy as np

def isru(x, l=-1, h=1, a=None, b=None, name='isru', axis=-1):
    lim = 4
    if a is None:
        _a = h - l
    else:
        _a = tf.Variable(name=name + '_a', initial_value=np.ones(np.array([_ for _ in x.shape])[axis]) + a, trainable=True, dtype=tf.float32)
        _a = 2 ** isru(_a, l=-lim, h=lim)

    if b is None:
        _b = 1
    else:
        _b = tf.Variable(name=name + '_b', initial_value=np.zeros(np.array([_ for _ in x.shape])[axis]) + b, trainable=True, dtype=tf.float32)
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