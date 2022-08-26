import tensorflow as tf

class graph_object(object):
    def __init__(self):
        self.init=0

#Common layers
def Get_Gene_Features(self,embedding_dim_genes,gene_features):
    if self.use_v_beta is True:
        X_v_beta = tf.compat.v1.placeholder(tf.int64, shape=[None, ], name='Input_V_Beta')
        X_v_beta_OH = tf.one_hot(X_v_beta, depth=len(self.lb_v_beta.classes_))
        embedding_layer_v_beta = tf.compat.v1.get_variable(name='Embedding_V_beta',
                                                 shape=[len(self.lb_v_beta.classes_)-1, embedding_dim_genes])
        embedding_layer_v_beta = tf.concat([tf.reduce_mean(embedding_layer_v_beta,0,keepdims=True),embedding_layer_v_beta], axis=0)
        X_v_beta_embed = tf.matmul(X_v_beta_OH, embedding_layer_v_beta)
        gene_features.append(X_v_beta_embed)
    else:
        X_v_beta = None
        X_v_beta_OH = None
        embedding_layer_v_beta = None

    if self.use_d_beta is True:
        X_d_beta = tf.compat.v1.placeholder(tf.int64, shape=[None, ], name='Input_D_Beta')
        X_d_beta_OH = tf.one_hot(X_d_beta, depth=len(self.lb_d_beta.classes_))
        embedding_layer_d_beta = tf.compat.v1.get_variable(name='Embedding_D_beta',
                                                 shape=[len(self.lb_d_beta.classes_)-1, embedding_dim_genes])
        embedding_layer_d_beta = tf.concat([tf.reduce_mean(embedding_layer_d_beta,0,keepdims=True),embedding_layer_d_beta], axis=0)
        X_d_beta_embed = tf.matmul(X_d_beta_OH, embedding_layer_d_beta)
        gene_features.append(X_d_beta_embed)
    else:
        X_d_beta = None
        X_d_beta_OH = None
        embedding_layer_d_beta = None

    if self.use_j_beta is True:
        X_j_beta = tf.compat.v1.placeholder(tf.int64, shape=[None, ], name='Input_J_Beta')
        X_j_beta_OH = tf.one_hot(X_j_beta, depth=len(self.lb_j_beta.classes_))
        embedding_layer_j_beta = tf.compat.v1.get_variable(name='Embedding_J_Beta',
                                                 shape=[len(self.lb_j_beta.classes_)-1, embedding_dim_genes])
        embedding_layer_j_beta = tf.concat([tf.reduce_mean(embedding_layer_j_beta,0,keepdims=True),embedding_layer_j_beta], axis=0)
        X_j_beta_embed = tf.matmul(X_j_beta_OH, embedding_layer_j_beta)
        gene_features.append(X_j_beta_embed)
    else:
        X_j_beta = None
        X_j_beta_OH = None
        embedding_layer_j_beta = None

    if self.use_v_alpha is True:
        X_v_alpha = tf.compat.v1.placeholder(tf.int64, shape=[None, ], name='Input_V_Alpha')
        X_v_alpha_OH = tf.one_hot(X_v_alpha, depth=len(self.lb_v_alpha.classes_))
        embedding_layer_v_alpha = tf.compat.v1.get_variable(name='Embedding_V_Alpha',
                                                 shape=[len(self.lb_v_alpha.classes_)-1, embedding_dim_genes])
        embedding_layer_v_alpha = tf.concat([tf.reduce_mean(embedding_layer_v_alpha,0,keepdims=True),embedding_layer_v_alpha], axis=0)
        X_v_alpha_embed = tf.matmul(X_v_alpha_OH, embedding_layer_v_alpha)
        gene_features.append(X_v_alpha_embed)
    else:
        X_v_alpha = None
        X_v_alpha_OH = None
        embedding_layer_v_alpha = None

    if self.use_j_alpha is True:
        X_j_alpha = tf.compat.v1.placeholder(tf.int64, shape=[None, ], name='Input_J_Alpha')
        X_j_alpha_OH = tf.one_hot(X_j_alpha, depth=len(self.lb_j_alpha.classes_))
        embedding_layer_j_alpha = tf.compat.v1.get_variable(name='Embedding_J_Alpha',
                                                 shape=[len(self.lb_j_alpha.classes_)-1, embedding_dim_genes])
        embedding_layer_j_alpha = tf.concat([tf.reduce_mean(embedding_layer_j_alpha,0,keepdims=True),embedding_layer_j_alpha], axis=0)
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
    GO.X_hla = tf.compat.v1.placeholder(tf.float32, shape=[None, self.hla_data_seq_num.shape[1]], name='HLA')
    GO.embedding_layer_hla = tf.compat.v1.get_variable(name='Embedding_HLA',
                                          shape=[len(self.lb_hla.classes_), embedding_dim])
    GO.HLA_Features = tf.matmul(GO.X_hla,GO.embedding_layer_hla)
    return GO.HLA_Features

def Convolutional_Features(inputs,reuse=False,prob=0.0,name='Convolutional_Features',
                           units=[32,64,128], kernel=[5,5,5], stride=[1,1,1],padding='same',
                           agg='max_pool',
                          l2_reg=0.0):

    with tf.compat.v1.variable_scope(name,reuse=reuse):
        assert len(kernel) == len(units) == len(stride), 'kernel, units, and stride must be of same length'

        if inputs.shape[2] > 40:
            check=1
        conv = inputs
        for ii,_ in enumerate(units,0):
            conv = tf.compat.v1.layers.conv2d(conv, units[ii], (1, kernel[ii]), (1,stride[ii]), padding=padding,
                                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
            conv = tf.nn.leaky_relu(conv)
            conv = tf.compat.v1.layers.dropout(conv, prob)

        if agg == 'max_pool':
            return tf.compat.v1.layers.flatten(tf.reduce_max(input_tensor=conv,axis=2))
        else:
            return tf.compat.v1.layers.flatten(conv)

def Conv_Model(GO, self, trainable_embedding,
               units_tcr,units_epitope,units_hla,
               kernel_tcr,kernel_epitope,kernel_hla,
               stride_tcr,stride_epitope,stride_hla,
               padding_tcr,padding_epitope,padding_hla,
               units_fc=[]):

    if self.use_alpha is True:
        GO.X_Seq_alpha = tf.compat.v1.placeholder(tf.int64,
                                        shape=[None, self.X_Seq_alpha.shape[1], self.X_Seq_alpha.shape[2]],
                                        name='Input_Alpha')
        GO.X_Seq_alpha_OH = tf.one_hot(GO.X_Seq_alpha, depth=21)

    if self.use_beta is True:
        GO.X_Seq_beta = tf.compat.v1.placeholder(tf.int64,
                                       shape=[None, self.X_Seq_beta.shape[1], self.X_Seq_beta.shape[2]],
                                       name='Input_Beta')
        GO.X_Seq_beta_OH = tf.one_hot(GO.X_Seq_beta, depth=21)

    if self.use_epitope is True:
        GO.X_Seq_epitope = tf.compat.v1.placeholder(tf.int64,
                                                    shape=[None,self.X_Seq_epitope.shape[1], self.X_Seq_epitope.shape[2]],
                                                    name='Input_Epitope')
        GO.X_Seq_epitope_OH = tf.one_hot(GO.X_Seq_epitope, depth=21)

    if self.use_hla and self.use_hla_seq:
        GO.X_Seq_hla = tf.compat.v1.placeholder(tf.int64,
                                               shape=[None, self.hla_data_seq_num.shape[1], self.hla_data_seq_num.shape[2]],
                                               name='HLA')
        GO.X_Seq_hla_OH = tf.one_hot(GO.X_Seq_hla, depth=21)

    GO.prob = tf.compat.v1.placeholder_with_default(0.0, shape=(), name='prob')
    GO.prob_multisample = tf.compat.v1.placeholder_with_default(0.0, shape=(), name='prob_multisample')
    GO.sp = tf.compat.v1.sparse.placeholder(dtype=tf.float32, shape=[None, None],name='sp')
    GO.X_Freq = tf.compat.v1.placeholder(tf.float32, shape=[None, ], name='Freq')
    GO.X_Counts = tf.compat.v1.placeholder(tf.float32, shape=[None, ], name='Counts')
    GO.i = tf.compat.v1.placeholder(dtype=tf.int32,shape= [None, ])
    GO.j = tf.compat.v1.placeholder(dtype=tf.int32,shape = [None, ])

    gene_features = []
    GO.X_v_beta, GO.X_v_beta_OH, GO.embedding_layer_v_beta, \
    GO.X_d_beta, GO.X_d_beta_OH, GO.embedding_layer_d_beta, \
    GO.X_j_beta, GO.X_j_beta_OH, GO.embedding_layer_j_beta, \
    GO.X_v_alpha, GO.X_v_alpha_OH, GO.embedding_layer_v_alpha, \
    GO.X_j_alpha, GO.X_j_alpha_OH, GO.embedding_layer_j_alpha, \
    gene_features = Get_Gene_Features(self, GO.embedding_dim_genes, gene_features)

    if trainable_embedding is True:
        # AA Embedding
        with tf.compat.v1.variable_scope('AA_Embedding'):
            GO.embedding_layer_seq = tf.compat.v1.get_variable(name='Embedding_Layer_Seq', shape=[21, GO.embedding_dim_aa])
            GO.embedding_layer_seq = tf.expand_dims(tf.expand_dims(GO.embedding_layer_seq, axis=0), axis=0)
            if self.use_alpha is True:
                inputs_seq_embed_alpha = tf.squeeze(
                    tf.tensordot(GO.X_Seq_alpha_OH, GO.embedding_layer_seq, axes=(3, 2)), axis=(3, 4))
            if self.use_beta is True:
                inputs_seq_embed_beta = tf.squeeze(
                    tf.tensordot(GO.X_Seq_beta_OH, GO.embedding_layer_seq, axes=(3, 2)), axis=(3, 4))
            if self.use_epitope is True:
                inputs_seq_embed_epitope = tf.squeeze(
                    tf.tensordot(GO.X_Seq_epitope_OH, GO.embedding_layer_seq, axes=(3, 2)), axis=(3, 4))
            if self.use_hla and self.use_hla_seq:
                inputs_seq_embed_hla = tf.squeeze(
                    tf.tensordot(GO.X_Seq_hla_OH, GO.embedding_layer_seq, axes=(3, 2)), axis=(3, 4))

    else:
        if self.use_alpha is True:
            inputs_seq_embed_alpha = GO.X_Seq_alpha_OH

        if self.use_beta is True:
            inputs_seq_embed_beta = GO.X_Seq_beta_OH

        if self.use_epitope is True:
            inputs_seq_embed_epitope = GO.X_Seq_epitope_OH

        if self.use_hla and self.use_hla_seq:
            inputs_seq_embed_hla = GO.X_Seq_hla_OH

    # Convolutional Features
    if self.use_alpha is True:
        GO.Seq_Features_alpha  = Convolutional_Features(inputs_seq_embed_alpha,
                                                        units = units_tcr,
                                                        kernel=kernel_tcr,
                                                        stride=stride_tcr,
                                                        padding=padding_tcr,
                                                        name='alpha_conv', prob=GO.prob,
                                                        agg='max_pool',
                                                        l2_reg=GO.l2_reg)

    if self.use_beta is True:
        GO.Seq_Features_beta = Convolutional_Features(inputs_seq_embed_beta,
                                                        units = units_tcr,
                                                        kernel=kernel_tcr,
                                                        stride=stride_tcr,
                                                        padding=padding_tcr,
                                                        name='beta_conv', prob=GO.prob,
                                                        agg='max_pool',
                                                        l2_reg = GO.l2_reg)
    if self.use_epitope is True:
        GO.Seq_Features_epitope  = Convolutional_Features(inputs_seq_embed_epitope,
                                                        units = units_epitope,
                                                        kernel=kernel_epitope,
                                                        stride=stride_epitope,
                                                        padding=padding_epitope,
                                                        name='epitope_conv', prob=GO.prob,
                                                        agg='max_pool',
                                                        l2_reg = GO.l2_reg)
    if self.use_hla and self.use_hla_seq:
        GO.Seq_Features_hla  = Convolutional_Features(inputs_seq_embed_hla,
                                                        units = units_hla,
                                                        kernel=kernel_hla,
                                                        stride=stride_hla,
                                                        padding=padding_hla,
                                                        name='hla_conv', prob=GO.prob,
                                                        agg='flat',
                                                        l2_reg = GO.l2_reg)

    Seq_Features = []
    if self.use_alpha is True:
        Seq_Features.append(GO.Seq_Features_alpha)
    if self.use_beta is True:
        Seq_Features.append(GO.Seq_Features_beta)
    if self.use_epitope is True:
        Seq_Features.append(GO.Seq_Features_epitope)
    if self.use_hla and self.use_hla_seq:
        Seq_Features.append(GO.Seq_Features_hla)

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

    if self.use_hla and not self.use_hla_seq:
        HLA_Features = Get_HLA_Features(self,GO,GO.embedding_dim_hla)
        Features = tf.concat((Features,HLA_Features),axis=1)

    GO.Features_Base = Features
    # if (self.use_hla) and (not use_only_hla):
    #     Features = tf.layers.dropout(Features,GO.prob)
    #     Features = tf.layers.dense(Features,Features.shape[1],tf.nn.relu)

    fc = Features
    for u in units_fc:
        fc = tf.compat.v1.layers.dropout(fc, GO.prob)
        fc = tf.compat.v1.layers.dense(fc, u, tf.nn.relu,
                                    kernel_regularizer=tf.keras.regularizers.l2(GO.l2_reg))

    return fc

#Layers for VAE

def determine_kr_str(upsample2_beta,GO,self):
    """ determines the right kernel and stride parameters on the last deconv layer
    to ensure the output is greater than the input length"""
    for kr in list(range(4,100)):
        for str in list(range(2,100)):
            upsample3_beta = tf.compat.v1.layers.conv2d_transpose(upsample2_beta, GO.embedding_dim_aa, (1, kr), (1, str),
                                                        activation=tf.nn.relu)
            if upsample3_beta.shape[2] >= self.max_length:
                break
        else:
            continue
        break

    return kr,str

def Recon_Loss(inputs,logits):
    #Calculate Per Sample Reconstruction Loss
    shape_layer_1 = inputs.get_shape().as_list()
    shape_layer_2 = tf.shape(input=inputs)
    recon_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs, logits=logits)
    recon_loss = tf.reshape(recon_loss,shape=[shape_layer_2[0]*shape_layer_2[1],shape_layer_1[2]])
    w=tf.cast(tf.squeeze(tf.greater(inputs,0),1),tf.float32)
    recon_loss = tf.reduce_mean(input_tensor=w*recon_loss,axis=1)
    return recon_loss

def Latent_Loss(z_log_var,z_mean,alpha=1e-3):
    #Calculate Per Sample Variational Loss
    latent_loss = -alpha *tf.reduce_sum(input_tensor=1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
    return latent_loss

def Get_Gene_Loss(fc,embedding_layer,X_OH):
    upsample1 = tf.compat.v1.layers.dense(fc, 128, tf.nn.relu)
    upsample2 = tf.compat.v1.layers.dense(upsample1, 64, tf.nn.relu)
    upsample3 = tf.compat.v1.layers.dense(upsample2, embedding_layer.shape[1], tf.nn.relu)
    logits = tf.matmul(upsample3, tf.transpose(a=embedding_layer))
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=X_OH, logits=logits)

    predicted = tf.argmax(input=logits,axis=1)
    actual = tf.argmax(input=X_OH,axis=1)
    accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.equal(predicted,actual),tf.float32))

    return loss,accuracy

def Get_HLA_Loss(fc,embedding_layer,X_OH,alpha=1.0):
    upsample1 = tf.compat.v1.layers.dense(fc, 128, tf.nn.relu)
    upsample2 = tf.compat.v1.layers.dense(upsample1, 64, tf.nn.relu)
    upsample3 = tf.compat.v1.layers.dense(upsample2, embedding_layer.shape[1], tf.nn.relu)
    logits = tf.matmul(upsample3, tf.transpose(a=embedding_layer))
    loss = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(labels=X_OH, logits=logits),axis=-1)

    predicted = tf.greater(logits,0.9)
    accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.equal(predicted,tf.cast(X_OH,tf.bool)),tf.float32))
    return loss, accuracy


def calc_entropy(a):
    return -tf.reduce_sum(input_tensor=a * tf.math.log(a))

def calc_norm_entropy(a):
    return calc_entropy(a) / tf.math.log(tf.cast(tf.shape(input=a)[0], tf.float32))

def sparsity_loss(z_w,sparsity_alpha):
    eigen = tf.linalg.norm(tensor=z_w, axis=0)
    eigen_prop = eigen / tf.reduce_sum(input_tensor=eigen)
    sparsity_cost = calc_norm_entropy(eigen_prop)
    sparsity_cost = sparsity_alpha * sparsity_cost
    return sparsity_cost

#Other Layers
def MultiSample_Dropout(X,num_masks=2,activation=tf.nn.relu,use_bias=True,
                       rate=0.25,units=12,name='ml_weights',reg=0.0):
    """
    Multi-Sample Dropout Layer

    Implements Mutli-Sample Dropout layer from "Multi-Sample Dropout for Accelerated Training and Better Generalization"
    https://arxiv.org/abs/1905.09788

    Inputs
    ---------------------------------------
    num_masks: int
        Number of dropout masks to sample from.

    activation: func
        activation function to use on layer

    use_bias: bool
        Whether to incorporate bias.

    rate: float
        dropout rate

    units: int
        Number of output nodes

    name: str
        Name of layer (tensorflow variable scope)

    reg: float
        alpha for l1 regulariization on final layer (feature selection)

    Returns
    ---------------------------------------

    output of layer of dimensionality [?,units]

    """
    out = []
    for i in range(num_masks):
        fc = tf.compat.v1.layers.dropout(X,rate=rate)
        if i==0:
            reuse=False
        else:
            reuse=True

        with tf.compat.v1.variable_scope(name,reuse=reuse):
            out.append(tf.compat.v1.layers.dense(fc,units=units,activation=activation,use_bias=use_bias,
                                       kernel_regularizer=tf.keras.regularizers.l1(reg)))
    return tf.reduce_mean(input_tensor=tf.stack(out),axis=0)

def make_test_pred_object():
    test_pred = graph_object()
    for set in ['train', 'valid', 'test']:
        test_pred.__dict__[set] = graph_object()
        test_pred.__dict__[set].y_test = []
        test_pred.__dict__[set].y_pred = []
    return test_pred