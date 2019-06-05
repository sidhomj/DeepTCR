import tensorflow as tf
import numpy as np


def anlu(x, init_s=0, name='anlu', axis=-1):
    if axis is None:
        s = tf.Variable(name=name + '_s', initial_value=init_s, trainable=True, dtype=tf.float32)
    else:
        s = tf.Variable(name=name + '_s',
                        initial_value=np.random.normal(0, 0.01, np.array([_.value for _ in x.shape])[axis]) + init_s,
                        trainable=True, dtype=tf.float32)

    return (x + tf.pow(tf.pow(2., s) + tf.pow(x, 2.), 1. / 2.)) / 2.

def gbell(x,  a_init=1.0, b_init=0.0, c_init=0.0, name='gbell',axis=-1):
    if axis is None:
        a = tf.Variable(name=name + 'a', initial_value=a_init, trainable=True, dtype=tf.float32)
        b = tf.Variable(name=name + 'b', initial_value=b_init, trainable=True, dtype=tf.float32)
        c = tf.Variable(name=name + 'c', initial_value=c_init, trainable=False, dtype=tf.float32)
    else:
        a = tf.Variable(name=name + '_a',
                        initial_value=np.random.normal(0, 0.01, np.array([_.value for _ in x.shape])[axis]) + a_init,
                        trainable=True, dtype=tf.float32)
        b = tf.Variable(name=name + '_b',
                        initial_value=np.random.normal(0, 0.01, np.array([_.value for _ in x.shape])[axis]) + b_init,
                        trainable=True, dtype=tf.float32)
        c = tf.Variable(name=name + 'c', initial_value=c_init, trainable=False, dtype=tf.float32)

    return 1 / (1 + ((((x - c) / tf.exp(a)) ** 2) ** (tf.exp(b) + 1)))


def airsig(x, init_s=0., init_a=0., low=0., high=1., name='airsig'):
   s = tf.Variable(name=name + '_s', initial_value=tf.zeros(1) + init_s, trainable=True)
   a = tf.Variable(name=name + '_a', initial_value=tf.zeros(1) + init_a, trainable=False)
   return low + ((high - low) / 2.) * (1. + ((x + a) / (tf.pow(tf.pow(2., s) + tf.pow(x, 2.), 1. / 2.)
                                                        + tf.pow(tf.pow(a, 2.), 1. / 2.)))), s,a

def Parametric_Relu(x,alpha_pos_init=1.0,alpha_neg_init=1.0,pos_train=True,neg_train=True):
    alpha_pos = tf.Variable(name='alpha_pos', initial_value=alpha_pos_init, trainable=pos_train)
    alpha_neg = tf.Variable(name='alpha_neg', initial_value=alpha_neg_init, trainable=neg_train)
    pos = tf.cast(tf.greater(x, 0), tf.float32) * x * alpha_pos
    neg = tf.cast(tf.less(x,0),tf.float32) * x * alpha_neg
    return pos + neg, alpha_pos,alpha_neg

def Parametric_Step(x,alpha_pos_init=1.0,alpha_neg_init=1.0,pos_train=True,neg_train=True):
    alpha_pos = tf.Variable(name='alpha_pos', initial_value=alpha_pos_init, trainable=pos_train)
    alpha_neg = tf.Variable(name='alpha_neg', initial_value=alpha_neg_init, trainable=neg_train)
    pos = tf.cast(tf.greater(x, 0), tf.float32) * alpha_pos
    neg = tf.cast(tf.less(x,0),tf.float32) * alpha_neg
    return pos + neg, alpha_pos, alpha_neg

def ada_exp(x,init_a=1.0,trainable=True):
    a = tf.Variable(name='a', initial_value=init_a, trainable=trainable)
    return tf.exp(-a*x),a

def ada_exp_np(x,a=1.0):
    return np.exp(-a*x)

# def gbell(x, a_init=1.0, b_init=0.0, c_init=0.0, name='gbell'):
#     a = tf.Variable(name=name + 'a', initial_value= a_init, trainable=False,dtype=tf.float32)
#     b = tf.Variable(name=name + 'b', initial_value= b_init, trainable=False,dtype=tf.float32)
#     c = tf.Variable(name=name + 'c', initial_value=c_init, trainable=False,dtype=tf.float32)
#     return 1 / (1 + (((x - c)/ tf.exp(a)) ** (2 * (tf.exp(b) + 1)))),a,b,c

def gbell2(x,  a_init=1.0, b_init=0.0, c_init=0.0, name='gbell',axis=-1):
    if axis is None:
        a = tf.Variable(name=name + 'a', initial_value=a_init, trainable=True, dtype=tf.float32)
        b = tf.Variable(name=name + 'b', initial_value=b_init, trainable=True, dtype=tf.float32)
        c = tf.Variable(name=name + 'c', initial_value=c_init, trainable=False, dtype=tf.float32)
    else:
        a = tf.Variable(name=name + '_a',
                        initial_value=np.random.normal(0, 0.01, np.array([_.value for _ in x.shape])[axis]) + a_init,
                        trainable=True, dtype=tf.float32)
        b = tf.Variable(name=name + '_b',
                        initial_value=np.random.normal(0, 0.01, np.array([_.value for _ in x.shape])[axis]) + b_init,
                        trainable=True, dtype=tf.float32)
        c = tf.Variable(name=name + 'c', initial_value=c_init, trainable=False, dtype=tf.float32)

    return 1 / (1 + (((x - c)/ tf.exp(a)) ** (2 * (tf.exp(b) + 1))))

def gbell_np(x, a=1.0, b=0.0, c=0):
    return 1 / (1 + (((x - c)/ np.exp(a)) ** (2 * (np.exp(b) + 1))))
