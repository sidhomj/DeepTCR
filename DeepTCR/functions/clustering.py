import tensorflow as tf
import numpy as np

def isru(x, l=-1., h=1., a=None):
    if a is None:
        a = h - l
    return l + (((h - l) / 2) * (1 + (x / (((a ** 2) + (x ** 2)) ** (1 / 2)))))


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


def kde_bell(x, init_a=0., init_b=0., name='kde_bell', axis=-1):
    if axis is None:
        a = tf.Variable(name=name + '_a', initial_value=init_a, trainable=True, dtype=tf.float32)
        b = tf.Variable(name=name + '_b', initial_value=init_b, trainable=True, dtype=tf.float32)
    else:
        a = tf.Variable(name=name + '_a', initial_value=np.random.normal(0, 0.01, np.array([_.value for _ in x.shape])[axis]) + init_a, trainable=True, dtype=tf.float32)
        b = tf.Variable(name=name + '_b', initial_value=np.random.normal(0, 0.01, np.array([_.value for _ in x.shape])[axis]) + init_b, trainable=True, dtype=tf.float32)

    a_ = 2 ** isru(a, l=-4., h=4.)
    b_ = (2 ** isru(b, l=-4., h=1.)) + 1

    return 0.5 * a_ * (a_ + ((x ** 2) ** b_)) ** -((1 / (2 * b_)) + 1)


def one_bell(x, a_init=0., b_init=0., name='one_bell', axis=-1):
    if axis is None:
        a = tf.Variable(name=name + '_a', initial_value=a_init, trainable=True, dtype=tf.float32)
        b = tf.Variable(name=name + '_b', initial_value=b_init, trainable=True, dtype=tf.float32)
    else:
        a = tf.Variable(name=name + '_a', initial_value=np.random.normal(0, 0.01, np.array([_.value for _ in x.shape])[axis]) + a_init, trainable=True, dtype=tf.float32)
        b = tf.Variable(name=name + '_b', initial_value=np.random.normal(0, 0.01, np.array([_.value for _ in x.shape])[axis]) + b_init, trainable=True, dtype=tf.float32)

    a_ = 2 ** isru(a, l=-4., h=4.)
    b_ = (2 ** isru(b, l=-4., h=1.)) + 1

    return (1 + (((x ** 2) / (a_ ** 2)) ** b_)) ** -1


def gvq(x, n_vectors, n_mixtures, vector_activation, vectors_init=None, data_activation=None):
    # generate vectors
    if vectors_init is None:
        vectors_init = np.random.uniform(-1, 1, [n_vectors, x.shape[-1]])
    vectors = tf.Variable(name='centroids', initial_value=vectors_init, dtype=tf.float32, trainable=True)
    # pass through activation function - should be same as data
    if vector_activation is not None:
        vectors = vector_activation(vectors)

    # data vectors have generally come off an activation function, but option is here if needed
    if data_activation is not None:
        data = data_activation(x)
    else:
        data = x

    # activation matrix events X centroids, in bounded space - isru (-1, 1)
    actmat = data[:, tf.newaxis, :] - vectors[tf.newaxis, :, :]
    actmat = kde_bell(tf.sqrt(tf.reduce_mean(actmat ** 2, axis=-1)), init_a=3., init_b=0., axis=-1)
    # actmat = tf.reduce_mean(kde_bell(actmat, init_a=3., init_b=0., axis=[2, 3]), axis=-1)

    #  mixture model approach
    mixture_weights = tf.Variable(name='mixture_weights', initial_value=np.random.normal(0, 0.01, [vectors.shape[0], n_mixtures]), dtype=tf.float32, trainable=True)
    mixture_weights = tf.nn.softmax(mixture_weights, axis=0)
    mixture_outputs = tf.tensordot(actmat, mixture_weights, [-1, 0])

    return mixture_outputs
