import tensorflow as tf
import numpy as np
from DeepTCR.functions.act_func import *

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
            return tf.concat((conv_out,conv_2_out,conv_3_out),axis=1),conv_out,indices

def Conv_Model(GO, self, trainable_embedding, kernel, use_only_seq,
               use_only_gene,use_only_hla,on_graph_clustering=False,num_clusters=12,
               num_fc_layers=0, units_fc=12):
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
    GO.sp = tf.sparse.placeholder(dtype=tf.float32, shape=[None, None],name='sp')
    GO.X_Freq = tf.placeholder(tf.float32, shape=[None, ], name='Freq')
    GO.seq_pred = tf.placeholder_with_default(False, shape=())
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

    if on_graph_clustering:
        Features = GCN(GO,Features, num_clusters)
        #Features, GO.centroids, GO.vq_bias, GO.s = DeepVectorQuantization(Features, GO.prob, num_clusters)
        #GO.act_params.extend([GO.centroids,GO.vq_bias,GO.s])


    if self.use_hla:
        HLA_Features = Get_HLA_Features(self,GO,GO.embedding_dim_hla)
        Features = tf.concat((Features,HLA_Features),axis=1)

    if use_only_seq:
        Features = Seq_Features
        if on_graph_clustering:
            #Features, GO.centroids, GO.vq_bias, GO.s = DeepVectorQuantization(Features, GO.prob, num_clusters)
            Features = GCN(GO,Features,num_clusters)

    if use_only_gene:
        Features = gene_features
        if on_graph_clustering:
            Features, GO.centroids, GO.vq_bias, GO.s = DeepVectorQuantization(Features, GO.prob, num_clusters)

    if use_only_hla:
        Features = HLA_Features

    GO.Features_Base = Features
    if (self.use_hla) and (not use_only_hla):
        Features = tf.layers.dropout(Features,GO.prob)
        Features = tf.layers.dense(Features,Features.shape[1],tf.nn.relu)

    fc = Features
    if num_fc_layers != 0:
        for lyr in range(num_fc_layers):
            fc = tf.layers.dropout(fc, GO.prob)
            fc = tf.layers.dense(fc, units_fc, tf.nn.relu)

    return fc

#Layers for VAE
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

def Get_HLA_Loss(fc,embedding_layer,X_OH,alpha=1.0):
    upsample1 = tf.layers.dense(fc, 128, tf.nn.relu)
    upsample2 = tf.layers.dense(upsample1, 64, tf.nn.relu)
    upsample3 = tf.layers.dense(upsample2, embedding_layer.shape[1], tf.nn.relu)
    logits = tf.matmul(upsample3, tf.transpose(embedding_layer))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=X_OH, logits=logits),-1)

    predicted = tf.greater(logits,0.9)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,tf.cast(X_OH,tf.bool)),tf.float32))
    return loss, accuracy

#Layers for Repertoire Classifier

def DeepVectorQuantization(d,prob, n_c, vq_bias_init=0., activation=anlu):
    d = tf.layers.dropout(d,prob)
    d = tf.layers.dense(d,12,tf.nn.relu)

    d = tf.layers.dropout(d,prob)
    # centroids
    c = tf.Variable(name='centroids', initial_value=tf.random_uniform([n_c, d.shape[-1].value]), trainable=True)

    # euclidean distance all rows of d to all rows of c
    seq_to_centroids_dist = tf.reduce_sum(tf.pow(d[:, tf.newaxis, :] - c[tf.newaxis, :, :], 2), axis=2)

    # get trainable bias terms per centroid
    vq_bias = tf.Variable(name='vq_bias', initial_value=tf.zeros([n_c, ]) + vq_bias_init, trainable=True)

    # activation (these have internal parameters also per centroid)'
    seq_to_centroids_act,s,a = activation(vq_bias - seq_to_centroids_dist)

    return seq_to_centroids_act,c,vq_bias,s

def GCN(GO,Features,num_clusters,n_d=12):
    X = Features
    X = tf.layers.dense(X, n_d)
    X = Reshape_X(X,GO.i,GO.j)
    A = Get_Adjacency_Matrix(GO,X)
    GO.A = A
    Features,S = GCN_Features(A,X,num_clusters)
    Features = Flatten_X(Features,GO.i,GO.j)
    GO.S = Flatten_X(S,GO.i,GO.j)
    return Features

def knn_step(D,k=30):
    val,ind = tf.nn.top_k(D,k)
    OH = tf.one_hot(ind,tf.shape(D)[-1])
    OH = tf.reduce_sum(OH,2)
    return D*OH


    # ind = tf.reshape(ind,[tf.shape(ind)[0]*tf.shape(ind)[1],-1])
    # val = tf.reshape(val,[tf.shape(val)[0]*tf.shape(val)[1],-1])
    # return tf.scatter_nd(tf.squeeze(ind,-1),tf.squeeze(val,-1),[tf.shape(D)[0],tf.shape(D)[1],tf.shape(D)[2]])

    # indices = tf.constant([[0], [2]])
    # updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
    #                         [7, 7, 7, 7], [8, 8, 8, 8]],
    #                        [[5, 5, 5, 5], [6, 6, 6, 6],
    #                         [7, 7, 7, 7], [8, 8, 8, 8]]])
    # shape = tf.constant([4, 4, 4])

def Get_Adjacency_Matrix(GO,X):

    #D = Gen_MDistance(X)
    D = Pairwise_Distance_TF(X)
    #A, GO.a = ada_exp(D,init_a=0.0)
    #GO.act_params.extend([GO.a])
    #A, GO.s,GO.a = anlu(-D)
    A, GO.a,GO.b,GO.c = gbell(D)
    #GO.act_params.extend([GO.a,GO.b])
    #A = tf.cond(GO.seq_pred,lambda: D, lambda: knn_step(D,k=30))

    # X = tf.layers.dense(X,6)
    # X = tf.layers.dense(X,3)
    # X = tf.layers.dense(X,1)
    # z = tf.zeros(shape=tf.shape(X)[1])
    # i = X[:, :, tf.newaxis, :] + z[tf.newaxis, tf.newaxis, :, tf.newaxis]
    # j = X[:, tf.newaxis, :, :] + z[tf.newaxis, :, tf.newaxis, tf.newaxis]
    # X_E = tf.concat((i, j), -1)
    # X_E = tf.transpose(X_E, [0, 2, 1, 3]) + X_E
    # #X_E = tf.concat((X_E,D),-1)
    #
    # # FC
    # fc = tf.layers.dense(X_E, 1)
    # fc = tf.squeeze(fc,-1)
    # A,_,_,_ = gbell(fc)

    # val,ind = tf.nn.top_k(D,30)
    # x,y,z  = tf.shape(val)[0],tf.shape(val)[1],tf.shape(val)[2]
    # idx = tf.range(x)
    # idx = tf.tile(idx,[y])
    # idx = tf.tile(idx,[z])
    # GO.x_idx = idx
    #
    # idx = tf.range(y)

    #A = D

    # D = Pairwise_Distance_TF(X)
    # val,ind = tf.nn.top_k(tf.negative(D),30)
    # OH = tf.one_hot(ind,tf.shape(D)[-1])
    # OH = tf.reduce_sum(OH,2)
    #
    # A = tf.expand_dims(tf.matmul(OH,tf.linalg.transpose(OH)),-1)
    # A = tf.layers.dense(A,1)
    # A = tf.squeeze(A,-1)
    #
    # #D = Pairwise_Distance_TF(X)[:,:,:,tf.newaxis]
    # #Reduce dimensionality to 1
    # #act = tf.nn.leaky_relu
    # X = tf.layers.dense(X,6)
    # X = tf.layers.dense(X,3)
    # X = tf.layers.dense(X,1)
    #
    # z = tf.zeros(shape=tf.shape(X)[1])
    # i = X[:, :, tf.newaxis, :] + z[tf.newaxis, tf.newaxis, :, tf.newaxis]
    # j = X[:, tf.newaxis, :, :] + z[tf.newaxis, :, tf.newaxis, tf.newaxis]
    # X_E = tf.concat((i, j), -1)
    # X_E = tf.transpose(X_E, [0, 2, 1, 3]) + X_E
    # #X_E = tf.concat((X_E,D),-1)
    #
    # # FC
    # fc = tf.layers.dense(X_E, 1)
    # fc = tf.squeeze(fc,-1)
    # A = fc
    # #A,pos,neg = Parametric_Step(A,alpha_neg_init=0.0,pos_train=True,neg_train=False)
    # #GO.act_params.extend([pos,neg])
    # #GO.act_params.extend([tf.trainable_variables()[-1]])
    # #GO.act_params.extend([tf.trainable_variables()[-1]])
    # #A,s,a = airsig(A,init_s=1.0)
    # #GO.act_params.extend([s])
    # GO.act_params.extend([tf.trainable_variables()[-1]])
    # A,s,a = anlu(A)
    # GO.act_params.extend([s,a])

    # NN = tf.expand_dims(tf.matmul(A,tf.linalg.transpose(A)),-1)
    # NN = tf.concat((NN,tf.expand_dims(A,-1)),-1)
    # A = tf.squeeze(tf.layers.dense(NN,1),-1)
    # GO.act_params.extend([tf.trainable_variables()[-1]])
    # A,s,a = anlu(A)
    # GO.act_params.extend([s,a])

    #A = tf.nn.sigmoid(A)

    # NN = tf.concat((tf.expand_dims(tf.matmul(A,tf.linalg.transpose(A)),-1),tf.expand_dims(A,-1)),-1)
    # A = tf.squeeze(tf.layers.dense(NN,1),-1)
    #A =  tf.nn.tanh(A)
    return A

def Reshape_X(X,i,j):
    return tf.scatter_nd(tf.concat((i[:, tf.newaxis], j[:, tf.newaxis]), -1),
                  X, [tf.reduce_max(i) + 1, tf.reduce_max(j) + 1,X.shape[-1]] )

def Flatten_X(X,i,j):
    return tf.gather_nd(X,tf.concat((i[:, tf.newaxis], j[:, tf.newaxis]), -1))

def GCN_Features(A,X,num_clusters=12):
    #Norm
    D_norm = tf.sqrt(1 / tf.reduce_sum(A, -1))
    Lap_D = tf.expand_dims(D_norm, -1) * A * tf.expand_dims(D_norm, -2)
    A = Lap_D

    #Hiarachical GCN
    num_gcn_layers = 1
    hierarchial_units = [num_clusters]
    for ii,(i, u) in enumerate(zip(range(num_gcn_layers), hierarchial_units),0):
        for i in range(1):
            X = tf.layers.dense(tf.matmul(A, X), u, activation=tf.nn.relu, use_bias=True)
        Z = X
        S = tf.nn.softmax(Z, -1)
        #GO.reg_losses += 1e6*tf.reduce_mean(tf.norm(GO.S, axis=-1, ord=1))
        # if ii == 0:
        #     X = tf.matmul(tf.linalg.transpose(S), X_Freq*Z)
        # else:
        #     X = tf.matmul(tf.linalg.transpose(S),Z)
        # A = tf.matmul(tf.matmul(tf.linalg.transpose(S), A), S)
        # temp.append(tf.layers.flatten(X))

    #out = tf.concat(temp,1)

    return Z,S

def Pairwise_Distance_TF(A):
   r = tf.reduce_sum(A*A,-1)
   r = tf.expand_dims(r,-1)
   D = tf.sqrt(tf.nn.relu(r - 2 *tf.matmul(A,tf.linalg.transpose(A)) + tf.linalg.transpose(r)))
   return D

def Gen_MDistance(X):
    x = tf.expand_dims(tf.linalg.transpose(X),-1)
    x = x - tf.linalg.transpose(x)
    x = tf.transpose(x,[0,2,3,1])
    xt = tf.transpose(x,[0,2,1,3])
    m = tf.get_variable('M', shape=[X.shape[-1], X.shape[-1]], dtype=tf.float32)
    a = tf.tensordot(xt, m, axes=[[-1], [0]])
    return tf.sqrt(tf.einsum('zjik,zijk->zij',a,x))

