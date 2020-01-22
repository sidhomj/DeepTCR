import tensorflow as tf
import numpy as np

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

def Get_HLA_Features(self,GO,embedding_dim):
    GO.X_hla = tf.placeholder(tf.float32, shape=[None, self.hla_data_seq_num.shape[1]], name='HLA')
    GO.embedding_layer_hla = tf.get_variable(name='Embedding_HLA',
                                          shape=[len(self.lb_hla.classes_), embedding_dim])
    GO.HLA_Features = tf.matmul(GO.X_hla,GO.embedding_layer_hla)
    return GO.HLA_Features

def Convolutional_Features(inputs,reuse=False,prob=0.0,name='Convolutional_Features',kernel=3,net='ae',
                           size_of_net = 'medium'):
    with tf.variable_scope(name,reuse=reuse):
        if size_of_net == 'small':
            units = [12,32,64]
        elif size_of_net == 'medium':
            units = [32,64,128]
        elif size_of_net == 'large':
            units = [64,128,256]
        else:
            units = size_of_net

        conv = tf.layers.conv2d(inputs, units[0], (1, kernel), 1, padding='same')
        conv_out = tf.layers.flatten(tf.reduce_max(conv,2))
        indices = tf.squeeze(tf.cast(tf.argmax(conv, axis=2), tf.float32),1)
        conv = tf.nn.leaky_relu(conv)
        conv = tf.layers.dropout(conv,prob)

        kernel = 3
        conv_2 = tf.layers.conv2d(conv, units[1], (1, kernel), (1, kernel), padding='same')
        conv_2 = tf.nn.leaky_relu(conv_2)
        conv_2 = tf.layers.dropout(conv_2, prob)
        conv_2_out = tf.layers.flatten(tf.reduce_max(conv_2,2))

        conv_3 = tf.layers.conv2d(conv_2, units[2], (1, kernel), (1, kernel), padding='same')
        conv_3 = tf.nn.leaky_relu(conv_3)
        conv_3 = tf.layers.dropout(conv_3, prob)
        conv_3_out = tf.layers.flatten(tf.reduce_max(conv_3,axis=2))

        if net == 'ae':
            return tf.layers.flatten(conv_3),conv_out,indices
        else:
            return conv_3_out,conv_out,indices

def Conv_Model(GO, self, trainable_embedding, kernel, use_only_seq,
               use_only_gene,use_only_hla,num_fc_layers=0, units_fc=12):

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

    GO.prob = tf.placeholder_with_default(0.0, shape=(), name='prob')
    GO.sp = tf.sparse.placeholder(dtype=tf.float32, shape=[None, None],name='sp')
    GO.X_Freq = tf.placeholder(tf.float32, shape=[None, ], name='Freq')
    GO.i = tf.placeholder(dtype=tf.int32,shape= [None, ])
    GO.j = tf.placeholder(dtype=tf.int32,shape = [None, ])

    gene_features = []
    GO.X_v_beta, GO.X_v_beta_OH, GO.embedding_layer_v_beta, \
    GO.X_d_beta, GO.X_d_beta_OH, GO.embedding_layer_d_beta, \
    GO.X_j_beta, GO.X_j_beta_OH, GO.embedding_layer_j_beta, \
    GO.X_v_alpha, GO.X_v_alpha_OH, GO.embedding_layer_v_alpha, \
    GO.X_j_alpha, GO.X_j_alpha_OH, GO.embedding_layer_j_alpha, \
    gene_features = Get_Gene_Features(self, GO.embedding_dim_genes, gene_features)

    if trainable_embedding is True:
        # AA Embedding
        with tf.variable_scope('AA_Embedding'):
            GO.embedding_layer_seq = tf.get_variable(name='Embedding_Layer_Seq', shape=[21, GO.embedding_dim_aa])
            GO.embedding_layer_seq = tf.expand_dims(tf.expand_dims(GO.embedding_layer_seq, axis=0), axis=0)
            if self.use_alpha is True:
                inputs_seq_embed_alpha = tf.squeeze(
                    tf.tensordot(GO.X_Seq_alpha_OH, GO.embedding_layer_seq, axes=(3, 2)), axis=(3, 4))
            if self.use_beta is True:
                inputs_seq_embed_beta = tf.squeeze(
                    tf.tensordot(GO.X_Seq_beta_OH, GO.embedding_layer_seq, axes=(3, 2)), axis=(3, 4))

    else:
        if self.use_alpha is True:
            inputs_seq_embed_alpha = GO.X_Seq_alpha_OH

        if self.use_beta is True:
            inputs_seq_embed_beta = GO.X_Seq_beta_OH

    # Convolutional Features
    if self.use_alpha is True:
        GO.Seq_Features_alpha, GO.alpha_out, GO.indices_alpha = Convolutional_Features(inputs_seq_embed_alpha,
                                                                                       kernel=kernel,
                                                                                       name='alpha_conv', prob=GO.prob,
                                                                                       net=GO.net,size_of_net=GO.size_of_net)

    if self.use_beta is True:
        GO.Seq_Features_beta, GO.beta_out, GO.indices_beta = Convolutional_Features(inputs_seq_embed_beta,
                                                                                    kernel=kernel,
                                                                                    name='beta_conv', prob=GO.prob,
                                                                                    net=GO.net,size_of_net=GO.size_of_net)

    Seq_Features = []
    if self.use_alpha is True:
        Seq_Features.append(GO.Seq_Features_alpha)
    if self.use_beta is True:
        Seq_Features.append(GO.Seq_Features_beta)

    if Seq_Features:
        Seq_Features = tf.concat(Seq_Features, axis=1)


    Features = [Seq_Features,gene_features]
    for ii,f in enumerate(Features,0):
        if not isinstance(f,list):
            f_temp = f
            break

    for jj in range(ii+1,len(Features)):
        if not isinstance(Features[jj],list):
            f_temp = tf.concat((f_temp,Features[jj]),axis=1)

    try:
        Features = f_temp
    except:
        pass

    if self.use_hla:
        HLA_Features = Get_HLA_Features(self,GO,GO.embedding_dim_hla)
        Features = tf.concat((Features,HLA_Features),axis=1)

    if use_only_seq:
        Features = Seq_Features

    if use_only_gene:
        Features = gene_features

    if use_only_hla:
        Features = HLA_Features

    GO.Features_Base = Features
    # if (self.use_hla) and (not use_only_hla):
    #     Features = tf.layers.dropout(Features,GO.prob)
    #     Features = tf.layers.dense(Features,Features.shape[1],tf.nn.relu)

    fc = Features
    if num_fc_layers != 0:
        for lyr in range(num_fc_layers):
            fc = tf.layers.dropout(fc, GO.prob)
            fc = tf.layers.dense(fc, units_fc, tf.nn.relu)

    return fc

def Convolutional_Features_LI(inputs,reuse=False,prob=0.0,name='Convolutional_Features',kernel=3,net='ae',
                           size_of_net = 'medium'):
    with tf.variable_scope(name,reuse=reuse):
        if size_of_net == 'small':
            units = [12,32,64]
        elif size_of_net == 'medium':
            units = [32,64,128]
        elif size_of_net == 'large':
            units = [64,128,256]
        else:
            units = size_of_net

        conv_1 = tf.layers.conv2d(inputs, units[0], (1, kernel), 1, padding='same')
        conv_1 = tf.nn.leaky_relu(conv_1)
        conv_1 = tf.layers.dropout(conv_1,prob)
        conv_out = tf.layers.flatten(tf.reduce_max(conv_1,2))
        indices_out = tf.squeeze(tf.cast(tf.argmax(conv_1, axis=2), tf.float32),1)
        features = tf.layers.flatten(tf.reduce_max(conv_1,2))
        indices = tf.squeeze(tf.cast(tf.argmax(conv_1, axis=2), tf.float32),1)
        #
        # kernel = 3
        # conv_2 = tf.layers.conv2d(conv_1, units[1], (1, kernel), (1, kernel), padding='same')
        # conv_2 = tf.nn.leaky_relu(conv_2)
        # conv_2 = tf.layers.dropout(conv_2, prob)
        # features = tf.concat((features,tf.layers.flatten(tf.reduce_max(conv_2,2))),1)
        # indices = tf.concat((indices,tf.squeeze(tf.cast(tf.argmax(conv_2, axis=2), tf.float32),1)),1)
        #
        # conv_3 = tf.layers.conv2d(conv_2, units[2], (1, kernel), (1, kernel), padding='same')
        # conv_3 = tf.nn.leaky_relu(conv_3)
        # conv_3 = tf.layers.dropout(conv_3, prob)
        # features = tf.concat((features,tf.layers.flatten(tf.reduce_max(conv_3,2))),1)
        # indices = tf.concat((indices,tf.squeeze(tf.cast(tf.argmax(conv_3, axis=2), tf.float32),1)),1)
        return features,indices,conv_out,indices_out

def Conv_Model_LI(GO, self, trainable_embedding, kernel, use_only_seq,
               use_only_gene,use_only_hla,num_fc_layers=0, units_fc=12,num_fc_layers_ind=0,units_fc_ind=12):

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

    GO.prob = tf.placeholder_with_default(0.0, shape=(), name='prob')
    GO.sp = tf.sparse.placeholder(dtype=tf.float32, shape=[None, None],name='sp')
    GO.X_Freq = tf.placeholder(tf.float32, shape=[None, ], name='Freq')
    GO.i = tf.placeholder(dtype=tf.int32,shape= [None, ])
    GO.j = tf.placeholder(dtype=tf.int32,shape = [None, ])

    gene_features = []
    GO.X_v_beta, GO.X_v_beta_OH, GO.embedding_layer_v_beta, \
    GO.X_d_beta, GO.X_d_beta_OH, GO.embedding_layer_d_beta, \
    GO.X_j_beta, GO.X_j_beta_OH, GO.embedding_layer_j_beta, \
    GO.X_v_alpha, GO.X_v_alpha_OH, GO.embedding_layer_v_alpha, \
    GO.X_j_alpha, GO.X_j_alpha_OH, GO.embedding_layer_j_alpha, \
    gene_features = Get_Gene_Features(self, GO.embedding_dim_genes, gene_features)

    if trainable_embedding is True:
        # AA Embedding
        with tf.variable_scope('AA_Embedding'):
            GO.embedding_layer_seq = tf.get_variable(name='Embedding_Layer_Seq', shape=[21, GO.embedding_dim_aa])
            GO.embedding_layer_seq = tf.expand_dims(tf.expand_dims(GO.embedding_layer_seq, axis=0), axis=0)
            if self.use_alpha is True:
                inputs_seq_embed_alpha = tf.squeeze(
                    tf.tensordot(GO.X_Seq_alpha_OH, GO.embedding_layer_seq, axes=(3, 2)), axis=(3, 4))
            if self.use_beta is True:
                inputs_seq_embed_beta = tf.squeeze(
                    tf.tensordot(GO.X_Seq_beta_OH, GO.embedding_layer_seq, axes=(3, 2)), axis=(3, 4))

    else:
        if self.use_alpha is True:
            inputs_seq_embed_alpha = GO.X_Seq_alpha_OH

        if self.use_beta is True:
            inputs_seq_embed_beta = GO.X_Seq_beta_OH

    # Convolutional Features
    if self.use_alpha is True:
        GO.Seq_Features_alpha,GO.indices_alpha, GO.alpha_out, GO.indices_alpha_out = Convolutional_Features_LI(inputs_seq_embed_alpha,
                                                                                       kernel=kernel,
                                                                                       name='alpha_conv', prob=GO.prob,
                                                                                       net=GO.net,size_of_net=GO.size_of_net)

    if self.use_beta is True:
        GO.Seq_Features_beta, GO.indices_beta,GO.beta_out,GO.indices_beta_out = Convolutional_Features_LI(inputs_seq_embed_beta,
                                                                                    kernel=kernel,
                                                                                    name='beta_conv', prob=GO.prob,
                                                                                    net=GO.net,size_of_net=GO.size_of_net)

    Seq_Features = []
    if self.use_alpha is True:
        Seq_Features.append(GO.Seq_Features_alpha)
    if self.use_beta is True:
        Seq_Features.append(GO.Seq_Features_beta)

    if Seq_Features:
        Seq_Features = tf.concat(Seq_Features, axis=1)


    Features = [Seq_Features,gene_features]
    for ii,f in enumerate(Features,0):
        if not isinstance(f,list):
            f_temp = f
            break

    for jj in range(ii+1,len(Features)):
        if not isinstance(Features[jj],list):
            f_temp = tf.concat((f_temp,Features[jj]),axis=1)

    try:
        Features = f_temp
    except:
        pass

    if self.use_hla:
        HLA_Features = Get_HLA_Features(self,GO,GO.embedding_dim_hla)
        Features = tf.concat((Features,HLA_Features),axis=1)

    if use_only_seq:
        Features = Seq_Features

    if use_only_gene:
        Features = gene_features

    if use_only_hla:
        Features = HLA_Features

    GO.Features_Base = Features

    fc = Features
    if num_fc_layers != 0:
        for lyr in range(num_fc_layers):
            fc = tf.layers.dropout(fc, GO.prob)
            fc = tf.layers.dense(fc, units_fc, tf.nn.relu)

    Seq_Indices = []
    if self.use_alpha:
        #Seq_Indices.append(tf.one_hot(tf.cast(GO.indices_alpha,tf.int64),self.max_length))
        Seq_Indices.append(GO.indices_alpha)

    if self.use_beta:
        #Seq_Indices.append(tf.one_hot(tf.cast(GO.indices_beta,tf.int64),self.max_length))
        Seq_Indices.append(GO.indices_beta)

    if Seq_Indices:
        num_fc_layers_ind = 6
        Seq_Indices = tf.concat(Seq_Indices, axis=1)
        if num_fc_layers_ind != 0:
            for _ in range(num_fc_layers_ind):
                Seq_Indices = tf.layers.dense(Seq_Indices,units_fc_ind,tf.nn.relu)

    return fc, Seq_Indices

#Layers for VAE
def upper_mask(A):
    ones = tf.ones_like(A)
    mask_a = tf.matrix_band_part(ones, 0, -1)  # Upper triangular matrix of 0s and 1s
    mask_b = tf.matrix_band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  #
    return mask

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

def VAE_Latent_Layer(GO,input,latent_dim,name='Latent_Layer'):
    with tf.variable_scope(name):
        LLO = graph_object()
        fc = tf.layers.dense(input, 256)
        fc = tf.layers.dense(fc, latent_dim)
        z_w = tf.get_variable(name='z_w', shape=[latent_dim, latent_dim])
        LLO.z_w = z_w
        if GO.sparsity_alpha is not None:
            sparsity_cost = tf.linalg.norm(z_w, axis=0)
            sparsity_cost = sparsity_cost / tf.reduce_sum(sparsity_cost)
            sparsity_cost = -tf.reduce_sum(sparsity_cost * tf.log(sparsity_cost))
            sparsity_cost = sparsity_cost / np.log(latent_dim)
            LLO.sparsity_cost = GO.sparsity_alpha * sparsity_cost
        else:
            LLO.sparsity_cost = tf.Variable(0.0)

        if GO.ortho_alpha is not None:
            # Get pairwise dot product
            ortho_a = tf.matmul(z_w, z_w, transpose_a=True)

            # select for all unique pairwise and take dot product
            mask = upper_mask(ortho_a)
            ortho_a = tf.abs(tf.boolean_mask(ortho_a, mask))
            ortho_cost = tf.reduce_mean(ortho_a)
            LLO.ortho_cost = GO.ortho_alpha * ortho_cost

        else:
            LLO.ortho_cost = tf.Variable(0.0)

        z_mean = tf.matmul(fc, z_w, name='z_mean')
        z_log_var = tf.layers.dense(fc, latent_dim, activation=tf.nn.softplus, name='z_log_var')
        latent_cost = Latent_Loss(z_log_var, z_mean, alpha=GO.variational_alpha)
        LLO.latent_cost = latent_cost

        epsilon = tf.random_normal(tf.shape(z_mean))
        z = z_mean + tf.exp(z_log_var / 2) * epsilon
        z = tf.identity(z, name='z')
        LLO.z_mean = z_mean
        LLO.z_log_var = z_log_var
        LLO.z = z
        return LLO



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

def Get_HLA_Loss(fc,embedding_layer,X_OH,alpha=1.0):
    upsample1 = tf.layers.dense(fc, 128, tf.nn.relu)
    upsample2 = tf.layers.dense(upsample1, 64, tf.nn.relu)
    upsample3 = tf.layers.dense(upsample2, embedding_layer.shape[1], tf.nn.relu)
    logits = tf.matmul(upsample3, tf.transpose(embedding_layer))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=X_OH, logits=logits),-1)

    predicted = tf.greater(logits,0.9)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,tf.cast(X_OH,tf.bool)),tf.float32))
    return loss, accuracy


