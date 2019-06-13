import tensorflow as tf
import numpy as np

def gbell(x, a_init=1.0, b_init=0.0, c_init=0.0, name='gbell'):
    a = tf.Variable(name=name + 'a', initial_value= a_init, trainable=True,dtype=tf.float32)
    b = tf.Variable(name=name + 'b', initial_value= b_init, trainable=True,dtype=tf.float32)
    c = tf.Variable(name=name + 'c', initial_value=c_init, trainable=False,dtype=tf.float32)
    return 1 / (1 + (((x - c)/ tf.exp(a)) ** (2 * (tf.exp(b) + 1)))),a,b,c

def anlu(x, init_s=0., init_b=None, name='anlu', axis=-1):
    b = 0
    if axis is None:
        s = tf.Variable(name=name + '_s', initial_value=init_s, trainable=True, dtype=tf.float32)
        if init_b is not None:
            b = tf.Variable(name=name + '_b', initial_value=init_b, trainable=True, dtype=tf.float32)
    else:
        s = tf.Variable(name=name + '_s', initial_value=np.random.normal(0, 0.01, np.array([_.value for _ in x.shape])[axis]) + init_s, dtype=tf.float32, trainable=True)
        if init_b is not None:
            b = tf.Variable(name=name + '_b', initial_value=np.random.normal(0, 0.01, np.array([_.value for _ in x.shape])[axis]) + init_b, dtype=tf.float32, trainable=True)

    _s = 2 ** isru(s, l=-4., h=4.)
    _b = (b ** 3) + b
    return ((x + _b) + ((_s + ((x + _b) ** 2)) ** (1 / 2))) / 2

def isru(x, l=-1., h=1., a=None):
    if a is None:
        a = h - l
    return l + (((h - l) / 2) * (1 + (x / (((a ** 2) + (x ** 2)) ** (1 / 2)))))

