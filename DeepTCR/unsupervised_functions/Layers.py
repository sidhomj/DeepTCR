import tensorflow as tf
import numpy as np

#Layers for VAE
def Convolutional_Features_AE(inputs,reuse=False,training=False,prob=0.0):
    with tf.variable_scope('Convolutional_Features',reuse=reuse):
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

def AE_Loss(inputs,logits,z_log_var,z_mean):

    #Calculate Per Sample Reconstruction Loss
    shape_layer_1 = inputs.get_shape().as_list()
    shape_layer_2 = tf.shape(inputs)
    recon_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs, logits=logits)
    recon_loss = tf.reshape(recon_loss,shape=[shape_layer_2[0]*shape_layer_2[1],shape_layer_1[2]])
    w=tf.cast(tf.squeeze(tf.greater(inputs,0),1),tf.float32)
    recon_loss = tf.reduce_mean(w*recon_loss,axis=1)

    #Calculate Per Sample Variational Loss
    latent_loss = -1e-9 *tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)

    total_loss = recon_loss + latent_loss
    total_loss = tf.reduce_sum(total_loss)
    recon_loss = tf.reduce_sum(recon_loss)
    latent_loss = tf.reduce_sum(latent_loss)

    return total_loss,recon_loss,latent_loss

#Layers for GAN

def Convolutional_Features_GAN(inputs,reuse=False,training=False,prob=0.0,kernel=3,units=256):

    with tf.variable_scope('Convolutional_Features',reuse=reuse):
        conv = tf.layers.conv2d(inputs, units, (1, kernel), 1, padding='same')
        conv = tf.nn.leaky_relu(conv)

        indices = tf.cast(tf.argmax(conv, axis=2), tf.float32)
        conv = tf.layers.max_pooling2d(conv,(1,conv.shape[2]),(1,conv.shape[2]))

        return tf.layers.flatten(conv),tf.layers.flatten(indices)

def generator(z,embedding_dim_aa=64,prob=0.0,training=True):
    with tf.variable_scope('generator'):
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
