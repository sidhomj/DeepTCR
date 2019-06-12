import tensorflow as tf

def gbell(x, a_init=1.0, b_init=0.0, c_init=0.0, name='gbell'):
    a = tf.Variable(name=name + 'a', initial_value= a_init, trainable=True,dtype=tf.float32)
    b = tf.Variable(name=name + 'b', initial_value= b_init, trainable=True,dtype=tf.float32)
    c = tf.Variable(name=name + 'c', initial_value=c_init, trainable=False,dtype=tf.float32)
    return 1 / (1 + (((x - c)/ tf.exp(a)) ** (2 * (tf.exp(b) + 1)))),a,b,c