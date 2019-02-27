import tensorflow as tf
from DeepTCR.supervised_functions.common_layers import *

#Layers for VAE
def Convolutional_Features_AE(inputs,reuse=False,training=False,prob=0.0,name='Convolutional_Features'):
    with tf.variable_scope(name,reuse=reuse):
        kernel = 3
        units = 12
        conv = tf.layers.conv2d(inputs, units, (1, kernel), 1, padding='same')
        conv = tf.nn.leaky_relu(conv)
        conv = tf.layers.dropout(conv,prob)

        units = 32
        conv = tf.layers.conv2d(conv, units, (1, kernel), (1, kernel), padding='same')
        conv = tf.nn.leaky_relu(conv)
        conv = tf.layers.dropout(conv, prob)

        units = 64
        conv = tf.layers.conv2d(conv, units, (1, kernel), (1, kernel), padding='same')
        conv = tf.nn.leaky_relu(conv)
        conv = tf.layers.dropout(conv, prob)

        return tf.layers.flatten(conv)

def Recon_Loss(inputs,logits):
    #Calculate Per Sample Reconstruction Loss
    shape_layer_1 = inputs.get_shape().as_list()
    shape_layer_2 = tf.shape(inputs)
    recon_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs, logits=logits)
    recon_loss = tf.reshape(recon_loss,shape=[shape_layer_2[0]*shape_layer_2[1],shape_layer_1[2]])
    w=tf.cast(tf.squeeze(tf.greater(inputs,0),1),tf.float32)
    recon_loss = tf.reduce_mean(w*recon_loss,axis=1)
    return recon_loss

def Latent_Loss(z_log_var,z_mean):
    #Calculate Per Sample Variational Loss
    latent_loss = -1e-9 *tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
    return latent_loss

def Get_Gene_Loss(fc,embedding_layer,X_OH):
    upsample1 = tf.layers.dense(fc, 124, tf.nn.relu)
    upsample2 = tf.layers.dense(upsample1, 64, tf.nn.relu)
    upsample3 = tf.layers.dense(upsample2, embedding_layer.shape[1], tf.nn.relu)
    logits = tf.matmul(upsample3, tf.transpose(embedding_layer))
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=X_OH, logits=logits)

    predicted = tf.argmax(logits,1)
    actual = tf.argmax(X_OH,1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,actual),tf.float32))

    return loss,accuracy

#Layers for GAN

def Convolutional_Features_GAN(inputs,reuse=False,training=False,prob=0.0,kernel=3,units=256,name='Convolutional_Features'):

    with tf.variable_scope(name,reuse=reuse):
        conv = tf.layers.conv2d(inputs, units, (1, kernel), 1, padding='same')
        conv = tf.nn.leaky_relu(conv)

        indices = tf.cast(tf.argmax(conv, axis=2), tf.float32)
        conv = tf.layers.max_pooling2d(conv,(1,conv.shape[2]),(1,conv.shape[2]))

        return tf.layers.flatten(conv),tf.layers.flatten(indices)

def generator(z,embedding_dim_aa=64,prob=0.0,training=True,name='generator',reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        z = tf.layers.dense(z,512)
        z = tf.reshape(z,(-1,1,1,z.shape[-1]))
        z = tf.layers.batch_normalization(z,training=training)
        seq = tf.nn.leaky_relu(z)

        kernel = 3
        stride = 5
        units=512
        logits = tf.layers.conv2d_transpose(seq, units, (1, kernel), strides=(1, stride), padding='same')
        logits = tf.layers.batch_normalization(logits,training=training)
        logits = tf.nn.leaky_relu(logits)
        logits = tf.layers.dropout(logits,prob)

        kernel = 3
        stride = 4
        units=256
        logits = tf.layers.conv2d_transpose(logits, units, (1, kernel), strides=(1, stride), padding='same')
        logits = tf.layers.batch_normalization(logits,training=training)
        logits = tf.nn.leaky_relu(logits)
        logits = tf.layers.dropout(logits,prob)

        kernel = 3
        stride = 2
        logits = tf.layers.conv2d_transpose(logits, embedding_dim_aa, (1, kernel), strides=(1, stride), padding='same')
        logits = tf.nn.tanh(logits)

        return logits

def generator_genes(z,embedding_dim,prob=0.0,training=True,name='generator_genes',reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        z = tf.layers.dense(z,512)
        z = tf.layers.batch_normalization(z,training=training)
        z = tf.nn.leaky_relu(z)

        z1 = tf.layers.dense(z,256)
        z1 = tf.layers.batch_normalization(z1,training=training)
        z1 = tf.nn.leaky_relu(z1)

        z2 = tf.layers.dense(z1, embedding_dim)
        z2 = tf.layers.batch_normalization(z2, training=training)
        z2 = tf.nn.leaky_relu(z2)

        return z2

def model_loss(logits_real,logits_fake,features_real,features_fake):

    #Vanilla GAN
    smooth_1=0.9
    smooth_2=0.0
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=smooth_1 * tf.ones_like(logits_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels= smooth_2*tf.ones_like(logits_fake)))
    d_loss = d_loss_real + d_loss_fake

    mu_real,sig_real = tf.nn.moments(features_real,0)
    mu_fake,sig_fake = tf.nn.moments(features_fake,0)
    fm_loss = tf.reduce_mean(tf.abs(mu_real-mu_fake))
    g_loss = - tf.reduce_mean(logits_fake) + fm_loss

    return d_loss,g_loss

def discriminator(features,indices,gene_features,name='discriminator',reuse=False,use_distances=False,num_fc=None):
    with tf.variable_scope(name,reuse=reuse):
        distances = rbf_layer(indices,64)
        if use_distances is True:
            features = tf.concat((features,distances),axis=1)

        if not isinstance(gene_features, list):
            features = tf.concat((features, gene_features), axis=1)

        if num_fc is not None:
            for n in range(num_fc):
                features = tf.layers.dense(features,256,tf.nn.relu)

        logits = tf.layers.dense(features,1)
        return logits,features

