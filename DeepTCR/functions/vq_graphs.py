import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


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


def anlu(x, init_s=0., init_b=None, name='anlu', axis=-1):
    b = 0
    if axis is None:
        s = tf.Variable(name=name + '_s', initial_value=init_s, trainable=True, dtype=tf.float32)
        if init_b is not None:
            b = tf.Variable(name=name + '_b', initial_value=init_b, trainable=True, dtype=tf.float32)
    else:
        s = tf.Variable(name=name + '_s', initial_value=np.zeros(np.array([_.value for _ in x.shape])[axis]) + init_s, dtype=tf.float32, trainable=True)
        if init_b is not None:
            b = tf.Variable(name=name + '_b', initial_value=np.random.normal(0, 0.01, np.array([_.value for _ in x.shape])[axis]) + init_b, dtype=tf.float32, trainable=True)

    _s = 2 ** isru(s, l=-4., h=4.)
    # _b = (b ** 3) + b
    _b = b
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


def gvq(x, n_vectors, vector_activation, vectors_init=None, data_activation=None):
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
    actmat = kde_bell(tf.sqrt(tf.reduce_mean(actmat ** 2, axis=-1)), init_a=10., init_b=0., axis=-1)
    # actmat = tf.reduce_mean(kde_bell(actmat, init_a=3., init_b=0., axis=[2, 3]), axis=-1)
    return actmat, vectors


def cross_entropy(labels, predictions, labels_type='index'):
    # labels are assumed to be probabilty vectors, unless specified as 'index'
    if labels_type is 'index':
        return -tf.reduce_sum(tf.one_hot(labels, predictions.shape[-1], on_value=1., off_value=0.) * tf.log(predictions), axis=1)
    else:
        return -tf.reduce_sum(labels * tf.log(predictions), axis=1)


def seq_embed(input, dim=4, trainable=False, variable_scope='seq_embed'):
    with tf.variable_scope(variable_scope):
        embedding_matrix = tf.Variable(name='embedding_matrix', initial_value=tf.diag(tf.ones(dim)), trainable=trainable)
        embedding_matrix = tf.concat([tf.zeros([1, dim]), embedding_matrix], axis=0, name='embedding_matrix_padded')

        return tf.gather(embedding_matrix, input, axis=0)


def seq_conv2d(input, filters=12, kernel_size=5, strides=(1, 1, 1, 1), padding='VALID', activation=None, variable_scope='seq_conv2d', reverse_complement=False, complement_indices=None, kernels_init=None):
    with tf.variable_scope(variable_scope):
        # feature extract to nD
        if kernels_init is None:
            kernels = tf.Variable(name='kernels', initial_value=np.random.normal(0, 0.001, [1, kernel_size, input.shape[2].value, filters]), dtype=tf.float32, trainable=True)
        else:
            kernels = tf.Variable(name='kernels', initial_value=kernels_init, dtype=tf.float32, trainable=False)
        bias = tf.Variable(name='bias', initial_value=-tf.ones(filters), dtype=tf.float32, trainable=True)

        if reverse_complement:
            linear = tf.squeeze(tf.nn.conv2d(tf.expand_dims(input, axis=1), filter=kernels, strides=strides, padding=padding), axis=1)
        else:
            linear = tf.squeeze(tf.nn.conv2d(tf.expand_dims(input, axis=1), filter=kernels, strides=strides, padding=padding), axis=1)  # / tf.reduce_sum(kernels ** 2, axis=[0, 1, 2])

        if activation is not None:
            return activation(linear + bias)
        else:
            return linear + bias


def mil_pool(instance_features, instance_counts, instance_sample, mil_output=1, activation=anlu, mil_dim=2, weighted_average=True):
    # instance level weights
    # re-tooling raw tanh * sigmoid as an "expressive sigmoid" per paper to a fully adaptive isru
    mil_weights = tf.layers.dense(instance_features, units=mil_dim, use_bias=False, activation=lambda x: isru(x, l=-1, h=1, a=0, b=0))
    #mil_weights = tf.layers.dense(instance_features, units=mil_dim, use_bias=False, activation=tf.nn.tanh) * tf.layers.dense(instance_features, units=mil_dim, use_bias=False, activation=tf.nn.sigmoid)
    # mil_weights = tf.layers.dropout(mil_weights, 0.5)
    mil_weights = tf.layers.dense(mil_weights, units=mil_output, use_bias=False, activation=activation)

    # scale weights by instance counts
    w = mil_weights * instance_counts[:, tf.newaxis]

    # mil pooling, return but sample level pooling and instance level weights
    if weighted_average:  # weighted average
        return tf.sparse.matmul(instance_sample, w * instance_features) / tf.sparse.matmul(instance_sample, w), mil_weights
    else:  # weighted sum
        return tf.sparse.matmul(instance_sample, w * instance_features), mil_weights


class DVQ:
    def __init__(self):
        self.graph = None
        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True

        self.model_ckpt = './ckpt/model_final'

    def _graph_input(self):
        # inputs
        self.instances_sequences = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.instances_counts = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.input_sample = tf.sparse.placeholder(dtype=tf.float32, shape=[None, None])

        # labels
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, ])

    def graph_loss_opt(self, learning_rate):
        # optimization op
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=self.loss)
        # set graph init_op
        self.init_op = tf.global_variables_initializer()
        # set saver
        self.saver = tf.train.Saver()

    def centroid_graph_nd(self, alphabet_dim, n_classes=2, learning_rate=0.001, kernels_init=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._graph_input()

            self.instances_sequences_embedded = seq_embed(self.instances_sequences, dim=alphabet_dim, trainable=False)
            self.instances_sequences_features = seq_conv2d(self.instances_sequences_embedded, filters=4, kernel_size=7, activation=anlu, kernels_init=kernels_init)
            self.instances_sequences_features = tf.reduce_max(self.instances_sequences_features, axis=1)

            # "rbf" layer
            self.gvq, self.c = gvq(self.instances_sequences_features, n_vectors=8, vector_activation=isru)

            # aggregate mixture activations in each sample
            self.agg = anlu(tf.sparse.matmul(self.input_sample, self.gvq), init_s=0., init_b=0.)
            # normalize aggregation over sample
            self.hist = self.agg / tf.reduce_sum(self.agg, axis=1)[:, tf.newaxis]

            # dense to classification on normalized histogram
            self.logits = tf.layers.dense(self.hist, units=n_classes, activation=None)

            # activation + normalized to class level probabilities
            self.prob = anlu(self.logits)
            self.prob = self.prob / tf.reduce_sum(self.prob, axis=1)[:, tf.newaxis]

            # cross entropy loss
            self.sample_loss = cross_entropy(self.labels, self.prob)
            self.loss = tf.reduce_mean(self.sample_loss)

            # optimization op
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=self.loss)

            # set graph init_op
            self.init_op = tf.global_variables_initializer()
            # set saver
            self.saver = tf.train.Saver()

    def dense_graph_nd(self, n_classes=2, learning_rate=0.001):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._graph_input()

            # aggregate activations in each sample
            self.agg = anlu(tf.sparse.matmul(self.input_sample, self.d))
            # normalize aggregation over sample
            self.hist = self.agg / tf.reduce_sum(self.agg, axis=1)[:, tf.newaxis]

            # dense on normalized histogram
            self.logits = tf.layers.dense(self.hist, units=n_classes, activation=None)

            # activation + normalized to class level probabilities
            self.prob = anlu(self.logits)
            self.prob = self.prob / tf.reduce_sum(self.prob, axis=1)[:, tf.newaxis]

            # cross entropy loss
            self.sample_loss = cross_entropy(self.labels, self.prob)
            self.loss = tf.reduce_mean(self.sample_loss)

    def mil_graph(self, alphabet_dim, n_classes=2, learning_rate=0.001, sample_pooling='mil', kernels_init=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._graph_input()
            self.instances_sequences_embedded = seq_embed(self.instances_sequences, dim=alphabet_dim, trainable=False)
            self.instances_sequences_features = seq_conv2d(self.instances_sequences_embedded, filters=4, kernel_size=7, activation=anlu, kernels_init=kernels_init)
            # self.instances_sequences_features = tf.layers.dropout(self.instances_sequences_features, 0.25)
            # self.instances_sequences_features = seq_conv2d(self.instances_sequences_features, filters=6, activation=anlu)
            # self.instances_sequences_features = tf.layers.dropout(self.instances_sequences_features, 0.25)
            self.instances_sequences_features = tf.reduce_max(self.instances_sequences_features, axis=1)
            self.instances_sequences_features = tf.layers.dense(self.instances_sequences_features, units=4, activation=anlu)

            if sample_pooling == 'mil':
                # mil dynamic permutation invariant pooling
                self.sample_pooled, self. mil_weights = mil_pool(self.instances_sequences_features, self.instances_counts, self.input_sample, mil_output=1, mil_dim=4, weighted_average=True, activation=anlu)
            else:
                self.sample_pooled = tf.sparse.matmul(self.input_sample, tf.expand_dims(self.instances_counts, -1) * self.instances_sequences_features) / tf.sparse.matmul(self.input_sample, tf.expand_dims(self.instances_counts, -1))

            # dense from mil pooling
            # hidden = tf.layers.dense(self.sample_pooled, units=3, activation=anlu)
            self.logits = tf.layers.dense(self.sample_pooled, units=n_classes, activation=None)

            # activation + normalized to class level probabilities
            self.prob = anlu(self.logits)
            self.prob = self.prob / tf.reduce_sum(self.prob, axis=1)[:, tf.newaxis]
            # self.prob = self.mil_pooling / tf.reduce_sum(self.mil_pooling, axis=1)[:, tf.newaxis]

            # cross entropy loss
            self.sample_loss = cross_entropy(self.labels, self.prob)
            self.loss = tf.reduce_mean(self.sample_loss)
            # self.loss = tf.reduce_mean(tf.square(tf.cast(self.labels, dtype=tf.float32) - self.prob))

            self.graph_loss_opt(learning_rate=learning_rate)

    def train(self, input_instances_sequences, input_instances_counts, input_sample, labels, n_epochs=1, training_proportion=0.75, convergence_threshold=0.001, convergence_size=100, continue_training=False, plot=False, plot_update=100):
        # TensorFlow session for training
        with tf.Session(graph=self.graph, config=self.tf_config) as sess:
            # continue training vs new var initialization
            if continue_training:
                # load previous state
                self.saver.restore(sess, self.model_ckpt)
            else:
                # tensorflow initializer
                sess.run(self.init_op)

            if plot:
                fig, ax = plt.subplots(nrows=4, ncols=4)
                fig.set_size_inches([8, 8])
                plt.tight_layout()

            # internal variables initial states
            train_loss = list()
            test_loss = list()

            p1 = list()
            p2 = list()

            convergence_window = np.nan
            epoch = 1
            global_iter = 1

            while epoch <= n_epochs:
                epoch_iter = 1

                # training samples and dict
                idx_train = np.random.choice(input_sample.shape[0], np.round(training_proportion * input_sample.shape[0]).astype(int), replace=False)
                idx_train_event = input_sample[idx_train, :].nonzero()[1]
                feed_dict_train = {self.instances_sequences: input_instances_sequences[idx_train_event],
                                   self.instances_counts: input_instances_counts[idx_train_event],
                                   self.input_sample: (np.stack(input_sample[idx_train, :][:, idx_train_event].nonzero(), axis=1),
                                                       input_sample[idx_train, :][:, idx_train_event].data,
                                                       np.array([len(idx_train), len(idx_train_event)])),
                                   self.labels: labels[idx_train]}

                # testing samples and dict
                idx_test = np.setdiff1d(np.arange(input_sample.shape[0]), idx_train)
                idx_test_event = input_sample[idx_test, :].nonzero()[1]
                feed_dict_test = {self.instances_sequences: input_instances_sequences[idx_test_event],
                                  self.instances_counts: input_instances_counts[idx_test_event],
                                  self.input_sample: (np.stack(input_sample[idx_test, :][:, idx_test_event].nonzero(), axis=1),
                                                      input_sample[idx_test, :][:, idx_test_event].data,
                                                      np.array([len(idx_test), len(idx_test_event)])),
                                  self.labels: labels[idx_test]}

                # generate figure and plotting axes if needed
                # if plot:
                # d_points = ax[0, 0].plot(input[labels == 0, :, 0], input[labels == 0, :, 1], '.', markersize=4, alpha=0.2, color='blue')
                # d_points = ax[1, 0].plot(input[labels == 1, :, 0], input[labels == 1, :, 1], '.', markersize=4, alpha=0.2, color='tan')
                # ax[1, 0].set(xlim=[-3, 3], ylim=[-3, 3])
                # ax[1, 1].set(xlim=[-3, 3], ylim=[-3, 3])
                # ax[1, 2].set(xlim=[-3, 3], ylim=[-3, 3])
                # c_points = ax[1, 0].plot(0, 0, 'x')
                # Y, X = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))

                # trip loop only if both convergence threshold met and a minimum iteration number
                while not ((convergence_window < convergence_threshold) & (epoch_iter > convergence_size)):
                    # assess training set
                    iter_time = time.time()
                    loss = sess.run(self.loss, feed_dict=feed_dict_train)
                    if (not (loss > 0)) or (convergence_window < 0):
                        if 0:
                            d = sess.run(self.instances_sequences_features, feed_dict=feed_dict_train)
                            m = sess.run(self.input_sample, feed_dict=feed_dict_train)
                            s = sess.run(self.graph.get_tensor_by_name('anlu_s:0'))
                            b = sess.run(self.graph.get_tensor_by_name('anlu_b:0'))

                            f = np.matmul(m, d) + b

                            s_ = 2 ** isru(s, l=-4., h=4.)

                            anlu_func(f, s_, b)

                            p = sess.run(self.pdf, feed_dict=feed_dict_train)
                            P = sess.run(self.prob, feed_dict=feed_dict_train)
                    iter_time = time.time() - iter_time
                    train_loss = np.append(train_loss, [loss], axis=0)

                    # p1.append(sess.run([self.sample_loss, self.support], feed_dict=feed_dict_train))
                    # p2.append(sess.run([self.graph.get_tensor_by_name('kde_bell_a:0'), self.graph.get_tensor_by_name('kde_bell_b:0')]))

                    # assess test set
                    loss = sess.run(self.loss, feed_dict=feed_dict_test)
                    test_loss = np.append(test_loss, [loss], axis=0)

                    # update operation
                    _ = sess.run(self.optimizer, feed_dict=feed_dict_train)

                    # convergence window in training
                    convergence_window = train_loss[-np.min((len(train_loss), convergence_size)):]
                    if len(convergence_window) > 1:
                        convergence_window = -np.median(np.diff(convergence_window) / np.abs(convergence_window[:-1]))
                    else:
                        convergence_window = np.nan

                    print('E: {:03d}'.format(epoch),
                          'E_i: {:03d}'.format(epoch_iter),
                          'E_t: {:.3f}'.format(iter_time),
                          'Train loss: {:.5f}'.format(train_loss[-1]),
                          'Convergence: {:.5f}%'.format(convergence_window * 100),
                          'Test loss: {:.5f}'.format(test_loss[-1]))
                    epoch_iter += 1
                    global_iter += 1

                    # save model
                    self.saver.save(sess, self.model_ckpt)

                    # plot
                    if plot:
                        if epoch_iter % plot_update == 0:
                            ax[0, 0].cla()
                            ax[0, 0].plot(np.stack([train_loss, test_loss], axis=1))

                            # d, c = sess.run([self.d, self.c], feed_dict=feed_dict_train)
                            # idx = labels[idx_train] == 0
                            # ax[1, 0].cla()
                            # ax[1, 0].hexbin(d[:, 0], d[:, 1], mincnt=1, zorder=-1)
                            # ax[1, 0].scatter(c[:, 0], c[:, 1], s=10, c='red', alpha=1, zorder=1)
                            # idx = labels[idx_train] == 1
                            # ax[1, 1].cla()
                            # ax[1, 1].hexbin(d[idx, :, 0], d[idx, :, 1], mincnt=1, zorder=-1)
                            # ax[1, 1].scatter(c[:, 0], c[:, 1], s=40, c=w[:, 1], alpha=1, zorder=1)

                            # X, Y = np.meshgrid(np.linspace(np.min(d[:, :, 0]), np.max(d[:, :, 0]), 20), np.linspace(np.min(d[:, :, 1]), np.max(d[:, :, 1]), 20))
                            # pdf = sess.run(self.pdf, feed_dict={**feed_dict_train, **{self.input_2d: np.stack([X, Y], axis=2).reshape([-1, 1, d.shape[2]]), self.input_2d_flag: [True]}})
                            # ax[2, 0].cla()
                            # ax[2, 0].pcolormesh(X, Y, pdf[:, 0].reshape([X.shape[0], Y.shape[1]]))
                            # ax[2, 1].cla()
                            # ax[2, 1].pcolormesh(X, Y, pdf[:, 1].reshape([X.shape[0], Y.shape[1]]))

                            # ax[0, 1].cla()
                            # ax[0, 1].plot(np.stack(p1, axis=0)[:, 0])
                            # ax[0, 2].cla()
                            # ax[0, 2].plot(np.stack(p1, axis=0)[:, 1])

                            # ax[0, 3].cla()
                            # # ax[0, 3].plot(np.stack(p2, axis=0)[:, 0, :].reshape([-1, np.prod(self.c.shape).value]))
                            # ax[0, 3].plot(np.stack(p2, axis=0)[:, 0, :])
                            # ax[1, 3].cla()
                            # # ax[1, 3].plot(np.stack(p2, axis=0)[:, 1, :].reshape([-1, np.prod(self.c.shape).value]))
                            # ax[1, 3].plot(np.stack(p2, axis=0)[:, 1, :])

                            fig.canvas.draw()

                epoch += 1

    def predict(self, instances_sequences, instances_counts, input_sample):
        # TensorFlow session for training
        with tf.Session(graph=self.graph, config=self.tf_config) as sess:
            self.saver.restore(sess, self.model_ckpt)
            feed_dict = {self.instances_sequences: instances_sequences, self.instances_counts: instances_counts,  self.input_sample: (np.stack(input_sample.nonzero(), axis=1), input_sample.data, input_sample.shape)}
            return sess.run([self.logits, self.prob], feed_dict=feed_dict)
