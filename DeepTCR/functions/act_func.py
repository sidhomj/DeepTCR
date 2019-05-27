import tensorflow as tf
import numpy as np

def anlu(x, s_init=0.,a_init=1.0):
    s = tf.Variable(name='anlu_s', initial_value= s_init, trainable=True)
    a = tf.Variable(name='anlu_a', initial_value= a_init, trainable=True,
                    constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
    return a*(x + tf.sqrt(tf.pow(2., s) + tf.pow(x, 2.))) / 2.,s, a

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

def gbell(x, a_init=1.0, b_init=0.0, c_init=0.0, name='gbell'):
    a = tf.Variable(name=name + 'a', initial_value= a_init, trainable=True,dtype=tf.float32)
    b = tf.Variable(name=name + 'b', initial_value= b_init, trainable=True,dtype=tf.float32)
    c = tf.Variable(name=name + 'c', initial_value=c_init, trainable=False,dtype=tf.float32)
    return 1 / (1 + (((x - c)/ tf.exp(a)) ** (2 * (tf.exp(b) + 1)))),a,b,c