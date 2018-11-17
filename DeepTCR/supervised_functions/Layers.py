import tensorflow as tf

def Convolutional_Features(inputs,reuse=False,units=12,kernel=5,trainable_embedding=False,name='Convolutional_Features'):
    with tf.variable_scope(name,reuse=reuse):

        if trainable_embedding is True:
            conv = tf.layers.conv2d(inputs, units, (1, kernel), 1, padding='same')
        else:
            conv_weights = tf.get_variable(name='conv_weights', shape=[1, kernel, 20, units])
            conv_weights = tf.abs(conv_weights)
            conv_zero = tf.get_variable(name='conv_zero', shape=[1, kernel, 1, units],initializer=tf.initializers.zeros(), trainable=False)
            conv_weights = tf.concat((conv_zero,conv_weights),axis=2)
            conv_bias = tf.get_variable(name='conv_bias', shape=units, initializer=tf.initializers.zeros())
            conv = tf.nn.conv2d(inputs, conv_weights, padding='SAME', strides=[1, 1, 1, 1]) + conv_bias


        conv = tf.nn.relu(conv)
        indices = tf.cast(tf.argmax(conv, axis=2), tf.float32)
        conv = tf.layers.max_pooling2d(conv,(1,conv.shape[2]),(1,conv.shape[2]))

        return tf.layers.flatten(conv), tf.layers.flatten(indices)

def Convolutional_Features_WF(inputs,reuse=False,units=12,kernel=5,trainable_embedding=False,conv_weights=None,name='Convolutional_Features'):
    with tf.variable_scope(name,reuse=reuse):
        if conv_weights is not None:
            conv_weights = conv_weights[:,:,1:,:]
            units_orig = conv_weights.shape[-1]
            conv_weights = tf.Variable(conv_weights, name='conv_weights', dtype=tf.float32,trainable=True)

            conv_weights_add = tf.get_variable(name='conv_weights_add', shape=[1, kernel, 20, units])
            conv_weights = tf.concat((conv_weights,conv_weights_add),axis=3)

            conv_weights = tf.abs(conv_weights)
            conv_weights_out = conv_weights
            conv_zero = tf.get_variable(name='conv_zero', shape=[1, kernel, 1, units+units_orig], initializer=tf.initializers.zeros(), trainable=False)
            conv_weights = tf.concat((conv_zero, conv_weights), axis=2)
            bias_val = -kernel+1
            conv_bias = tf.get_variable(name='conv_bias', shape=units, initializer=tf.initializers.constant(bias_val),trainable=True)
            bias_val = 0.0
            conv_bias_add =  tf.get_variable(name='conv_bias_add', shape=units_orig, initializer=tf.initializers.constant(bias_val),trainable=True)
            conv_bias = tf.concat((conv_bias,conv_bias_add),0)


        else:
            if trainable_embedding is True:
                conv_weights = tf.get_variable(name='conv_weights', shape=[1, kernel, inputs.shape[-1], units])
            else:
                conv_weights = tf.get_variable(name='conv_weights',shape=[1,kernel,20,units])
                conv_weights = tf.abs(conv_weights)

            conv_weights_out = conv_weights

            if trainable_embedding is False:
                conv_zero = tf.get_variable(name='conv_zero', shape=[1, kernel, 1, units], initializer=tf.initializers.zeros(), trainable=False)
                conv_weights = tf.concat((conv_zero, conv_weights), axis=2)

            conv_bias = tf.get_variable(name='conv_bias', shape=units, initializer=tf.initializers.constant(0.0),trainable=True)


        conv = tf.nn.conv2d(inputs, conv_weights, padding='SAME', strides=[1, 1, 1, 1]) + conv_bias

        conv = tf.nn.relu(conv)
        indices = tf.cast(tf.argmax(conv, axis=2), tf.float32)
        conv = tf.reduce_max(conv, axis=2)
        return conv, conv_weights_out, indices

def Convolutional_Features_Test(inputs,motifs,bias=0,reuse=False,name='Convolutional_Features'):
    with tf.variable_scope(name,reuse=reuse):
        conv = tf.nn.conv2d(inputs,motifs,padding='SAME',strides=[1,1,1,1]) - bias
        conv = tf.nn.relu(conv)
        conv = tf.squeeze(tf.layers.max_pooling2d(conv,(1,conv.shape[2]),(1,conv.shape[2])),2)

        return conv



