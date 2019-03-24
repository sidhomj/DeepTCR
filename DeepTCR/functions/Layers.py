import tensorflow as tf

class graph_object(object):
    def __init__(self):
        self.init=0

#Common layers
def Get_Gene_Features(self,embedding_dim_genes,gene_features):
    if self.use_v_beta is True:
        X_v_beta = tf.placeholder(tf.int64, shape=[None, ], name='Input_V_Beta')
        X_v_beta_OH = tf.one_hot(X_v_beta, depth=len(self.lb_v_beta.classes_))
        embedding_layer_v_beta = tf.get_variable(name='Embedding_V_beta',
                                                 shape=[len(self.lb_v_beta.classes_), embedding_dim_genes])
        X_v_beta_embed = tf.matmul(X_v_beta_OH, embedding_layer_v_beta)
        gene_features.append(X_v_beta_embed)
    else:
        X_v_beta = None
        X_v_beta_OH = None
        embedding_layer_v_beta = None

    if self.use_d_beta is True:
        X_d_beta = tf.placeholder(tf.int64, shape=[None, ], name='Input_D_Beta')
        X_d_beta_OH = tf.one_hot(X_d_beta, depth=len(self.lb_d_beta.classes_))
        embedding_layer_d_beta = tf.get_variable(name='Embedding_D_beta',
                                                 shape=[len(self.lb_d_beta.classes_), embedding_dim_genes])
        X_d_beta_embed = tf.matmul(X_d_beta_OH, embedding_layer_d_beta)
        gene_features.append(X_d_beta_embed)
    else:
        X_d_beta = None
        X_d_beta_OH = None
        embedding_layer_d_beta = None

    if self.use_j_beta is True:
        X_j_beta = tf.placeholder(tf.int64, shape=[None, ], name='Input_J_Beta')
        X_j_beta_OH = tf.one_hot(X_j_beta, depth=len(self.lb_j_beta.classes_))
        embedding_layer_j_beta = tf.get_variable(name='Embedding_J_Beta',
                                                 shape=[len(self.lb_j_beta.classes_), embedding_dim_genes])
        X_j_beta_embed = tf.matmul(X_j_beta_OH, embedding_layer_j_beta)
        gene_features.append(X_j_beta_embed)
    else:
        X_j_beta = None
        X_j_beta_OH = None
        embedding_layer_j_beta = None

    if self.use_v_alpha is True:
        X_v_alpha = tf.placeholder(tf.int64, shape=[None, ], name='Input_V_Alpha')
        X_v_alpha_OH = tf.one_hot(X_v_alpha, depth=len(self.lb_v_alpha.classes_))
        embedding_layer_v_alpha = tf.get_variable(name='Embedding_V_Alpha',
                                                 shape=[len(self.lb_v_alpha.classes_), embedding_dim_genes])
        X_v_alpha_embed = tf.matmul(X_v_alpha_OH, embedding_layer_v_alpha)
        gene_features.append(X_v_alpha_embed)
    else:
        X_v_alpha = None
        X_v_alpha_OH = None
        embedding_layer_v_alpha = None

    if self.use_j_alpha is True:
        X_j_alpha = tf.placeholder(tf.int64, shape=[None, ], name='Input_J_Alpha')
        X_j_alpha_OH = tf.one_hot(X_j_alpha, depth=len(self.lb_j_alpha.classes_))
        embedding_layer_j_alpha = tf.get_variable(name='Embedding_J_Alpha',
                                                 shape=[len(self.lb_j_alpha.classes_), embedding_dim_genes])
        X_j_alpha_embed = tf.matmul(X_j_alpha_OH, embedding_layer_j_alpha)
        gene_features.append(X_j_alpha_embed)
    else:
        X_j_alpha = None
        X_j_alpha_OH = None
        embedding_layer_j_alpha = None

    if gene_features:
        gene_features = tf.concat(gene_features, axis=1)

    return X_v_beta, X_v_beta_OH, embedding_layer_v_beta,\
            X_d_beta, X_d_beta_OH, embedding_layer_d_beta,\
            X_j_beta, X_j_beta_OH, embedding_layer_j_beta,\
            X_v_alpha,X_v_alpha_OH,embedding_layer_v_alpha,\
            X_j_alpha,X_j_alpha_OH,embedding_layer_j_alpha,\
            gene_features

def Get_Ortho_Loss(x,alpha=1e-6):
    loss = tf.abs(tf.matmul(x,x,transpose_b=True) - tf.eye(tf.shape(x)[-2]))
    indices = tf.constant(list(range(x.shape[1])))
    loss = alpha * tf.reduce_sum(tf.abs(tf.transpose(tf.gather(tf.transpose(loss),indices))))
    return loss

def Get_Ortho_Loss_dep(x,alpha=1.0):
    loss = tf.abs(tf.matmul(x,x,transpose_b=True) - tf.eye(tf.shape(x)[-2]))
    loss = alpha*tf.reduce_sum(loss)
    return loss

#Layers for VAE
def Convolutional_Features_AE(inputs,reuse=False,training=False,prob=0.0,name='Convolutional_Features'):
    with tf.variable_scope(name,reuse=reuse):
        kernel = 3
        units = 32
        conv = tf.layers.conv2d(inputs, units, (1, kernel), 1, padding='same')
        indices = tf.squeeze(tf.cast(tf.argmax(conv, axis=2), tf.float32),1)
        conv = tf.nn.leaky_relu(conv)
        conv = tf.layers.dropout(conv,prob)

        units = 64
        conv = tf.layers.conv2d(conv, units, (1, kernel), (1, kernel), padding='same')
        conv = tf.nn.leaky_relu(conv)
        conv = tf.layers.dropout(conv, prob)

        units = 128
        conv = tf.layers.conv2d(conv, units, (1, kernel), (1, kernel), padding='same')
        conv = tf.nn.leaky_relu(conv)
        conv = tf.layers.dropout(conv, prob)

        return tf.layers.flatten(conv),indices

def Recon_Loss(inputs,logits):
    #Calculate Per Sample Reconstruction Loss
    shape_layer_1 = inputs.get_shape().as_list()
    shape_layer_2 = tf.shape(inputs)
    recon_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs, logits=logits)
    recon_loss = tf.reshape(recon_loss,shape=[shape_layer_2[0]*shape_layer_2[1],shape_layer_1[2]])
    w=tf.cast(tf.squeeze(tf.greater(inputs,0),1),tf.float32)
    recon_loss = tf.reduce_mean(w*recon_loss,axis=1)
    return recon_loss

def Latent_Loss(z_log_var,z_mean,alpha=1e-3):
    #Calculate Per Sample Variational Loss
    latent_loss = -alpha *tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
    return latent_loss

def Get_Gene_Loss(fc,embedding_layer,X_OH):
    upsample1 = tf.layers.dense(fc, 128, tf.nn.relu)
    upsample2 = tf.layers.dense(upsample1, 64, tf.nn.relu)
    upsample3 = tf.layers.dense(upsample2, embedding_layer.shape[1], tf.nn.relu)
    logits = tf.matmul(upsample3, tf.transpose(embedding_layer))
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=X_OH, logits=logits)

    predicted = tf.argmax(logits,1)
    actual = tf.argmax(X_OH,1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,actual),tf.float32))

    return loss,accuracy

#Layers for supervised functions
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

def Conv_Model(GO,self,trainable_embedding,kernel,units,use_only_seq,use_only_gene,num_fc_layers,units_fc):
    if self.use_alpha is True:
        GO.X_Seq_alpha = tf.placeholder(tf.int64,
                                        shape=[None, self.X_Seq_alpha.shape[1], self.X_Seq_alpha.shape[2]],
                                        name='Input_Alpha')
        GO.X_Seq_alpha_OH = tf.one_hot(GO.X_Seq_alpha, depth=21)

    if self.use_beta is True:
        GO.X_Seq_beta = tf.placeholder(tf.int64,
                                       shape=[None, self.X_Seq_beta.shape[1], self.X_Seq_beta.shape[2]],
                                       name='Input_Beta')
        GO.X_Seq_beta_OH = tf.one_hot(GO.X_Seq_beta, depth=21)

    GO.Y = tf.placeholder(tf.float64, shape=[None, self.Y.shape[1]])
    GO.prob = tf.placeholder_with_default(0.0, shape=(), name='prob')
    GO.sp = tf.sparse.placeholder(dtype=tf.float32, shape=[None, None])
    GO.X_Freq = tf.placeholder(tf.float32, shape=[None, ], name='Freq')

    embedding_dim_genes = 48
    gene_features = []
    GO.X_v_beta, GO.X_v_beta_OH, GO.embedding_layer_v_beta, \
    GO.X_d_beta, GO.X_d_beta_OH, GO.embedding_layer_d_beta, \
    GO.X_j_beta, GO.X_j_beta_OH, GO.embedding_layer_j_beta, \
    GO.X_v_alpha, GO.X_v_alpha_OH, GO.embedding_layer_v_alpha, \
    GO.X_j_alpha, GO.X_j_alpha_OH, GO.embedding_layer_j_alpha, \
    gene_features = Get_Gene_Features(self, embedding_dim_genes, gene_features)

    if trainable_embedding is True:
        # AA Embedding
        with tf.variable_scope('AA_Embedding'):
            embedding_dim_aa = 64
            embedding_layer_seq = tf.get_variable(name='Embedding_Layer_Seq', shape=[21, embedding_dim_aa])
            embedding_layer_seq = tf.expand_dims(tf.expand_dims(embedding_layer_seq, axis=0), axis=0)
            if self.use_alpha is True:
                inputs_seq_embed_alpha = tf.squeeze(
                    tf.tensordot(GO.X_Seq_alpha_OH, embedding_layer_seq, axes=(3, 2)), axis=(3, 4))
            if self.use_beta is True:
                inputs_seq_embed_beta = tf.squeeze(
                    tf.tensordot(GO.X_Seq_beta_OH, embedding_layer_seq, axes=(3, 2)), axis=(3, 4))

    else:
        if self.use_alpha is True:
            inputs_seq_embed_alpha = GO.X_Seq_alpha_OH

        if self.use_beta is True:
            inputs_seq_embed_beta = GO.X_Seq_beta_OH

    # Convolutional Features
    if self.use_alpha is True:
        GO.Seq_Features_alpha, GO.Indices_alpha = Convolutional_Features(inputs_seq_embed_alpha, kernel=kernel,
                                                                         units=units,
                                                                         trainable_embedding=trainable_embedding,
                                                                         name='alpha_conv')
    if self.use_beta is True:
        GO.Seq_Features_beta, GO.Indices_beta = Convolutional_Features(inputs_seq_embed_beta, kernel=kernel,
                                                                       units=units,
                                                                       trainable_embedding=trainable_embedding,
                                                                       name='beta_conv')

    Seq_Features = []
    if self.use_alpha is True:
        Seq_Features.append(GO.Seq_Features_alpha)
    if self.use_beta is True:
        Seq_Features.append(GO.Seq_Features_beta)

    if Seq_Features:
        Seq_Features = tf.concat(Seq_Features, axis=1)

    if not isinstance(Seq_Features, list):
        if not isinstance(gene_features, list):
            Features = tf.concat((Seq_Features, gene_features), axis=1)
        else:
            Features = Seq_Features

        if use_only_seq is True:
            Features = Seq_Features

        if use_only_gene is True:
            Features = gene_features
    else:
        Features = gene_features

    fc = Features
    if num_fc_layers != 0:
        for lyr in range(num_fc_layers):
            fc = tf.layers.dropout(fc, GO.prob)
            fc = tf.layers.dense(fc, units_fc, tf.nn.relu)

    return fc





