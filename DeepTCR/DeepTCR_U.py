import os
import shutil
import sys
sys.path.append('../')
from DeepTCR.unsupervised_functions.utils import *
from DeepTCR.unsupervised_functions.Layers import *
from DeepTCR.supervised_functions.data_processing import *
from DeepTCR.supervised_functions.common_layers import *
import glob
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool
import pickle
import seaborn as sns
import colorsys
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,fcluster
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import pdist, squareform
import umap
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn import metrics as skmetrics
import sklearn
import phenograph
from scipy.spatial import distance


class DeepTCR_U(object):

    def __init__(self,Name,max_length=40,device='/gpu:0'):
        """
        Initialize Training Object.

        Initializes object and sets initial parameters.

        Inputs
        ---------------------------------------
        Name: str
            Name of the object.

        max_length: int
            maximum length of CDR3 sequence

        device: str
            In the case user is using tensorflow-gpu, one can
            specify the particular device to build the graphs on.

        Returns
        ---------------------------------------


        """
        #Assign parameters
        self.Name = Name
        self.max_length = max_length
        self.use_beta = False
        self.use_alpha = False
        self.device = device
        self.use_v_beta = False
        self.use_d_beta = False
        self.use_j_beta = False
        self.use_v_alpha = False
        self.use_j_alpha = False

        #Create dataframes for assigning AA to ints
        aa_idx, aa_mat = make_aa_df()
        aa_idx_inv = {v: k for k, v in aa_idx.items()}
        self.aa_idx = aa_idx
        self.aa_idx_inv = aa_idx_inv

        #Create directory for results of analysis
        directory = self.Name + '_Results'
        self.directory_results = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        #Create directory for any temporary files
        directory = self.Name
        if not os.path.exists(directory):
            os.makedirs(directory)

    def Get_Data(self,directory,Load_Prev_Data=False,classes=None,type_of_data_cut='Fraction_Response',data_cut=1.0,n_jobs=40,
                    aa_column_alpha = None,aa_column_beta = None, count_column = None,sep='\t',aggregate_by_aa=True,
                    v_alpha_column=None,j_alpha_column=None,
                    v_beta_column=None,j_beta_column=None,d_beta_column=None,p=None):
        """
        Get Data for Unsupervised Deep Learning Methods.

        Parse Data into appropriate inputs for neural network.

        Inputs
        ---------------------------------------
        directory: str
            Path to directory with folders with tsv files are present
            for analysis. Folders names become labels for files within them.

        Load_Prev_Data: bool
            Loads Previous Data.

        classes: list
            Optional selection of input of which sub-directories to use for analysis.


        type_of_data_cut: str
            Method by which one wants to sample from the TCRSeq File.

            Options are:
                Fraction_Response: A fraction (0 - 1) that samples the top fraction of the file by reads. For example,
                if one wants to sample the top 25% of reads, one would use this threshold with a data_cut = 0.25. The idea
                of this sampling is akin to sampling a fraction of cells from the file.

                Frequency_Cut: If one wants to select clones above a given frequency threshold, one would use this threshold.
                For example, if one wanted to only use clones about 1%, one would enter a data_cut value of 0.01.

                Num_Seq: If one wants to take the top N number of clones, one would use this threshold. For example,
                if one wanted to select the top 10 amino acid clones from each file, they would enter a data_cut value of 10.

                Read_Cut: If one wants to take amino acid clones with at least a certain number of reads, one would use
                this threshold. For example, if one wanted to only use clones with at least 10 reads,they would enter a data_cut value of 10.

                Read_Sum: IF one wants to take a given number of reads from each file, one would use this threshold. For example,
                if one wants to use the sequences comprising the top 100 reads of hte file, they would enter a data_cut value of 100.

        data_cut: float or int
            Value  associated with type_of_data_cut parameter.

        n_jobs: int
            Number of processes to use for parallelized operations.

        aa_column_alpha: int
            Column where alpha chain amino acid data is stored. (0-indexed)

        aa_column_beta: int
            Column where beta chain amino acid data is stored.(0-indexed)

        count_column: int
            Column where counts are stored.

        sep: str
            Type of delimiter used in file with TCRSeq data.

        aggregate_by_aa: bool
            Choose to aggregate sequences by unique amino-acid. Defaults to True. If set to False, will allow duplicates
            of the same amino acid sequence given it comes from different nucleotide clones.

        v_alpha_column: int
            Column where v_alpha gene information is stored.

        j_alpha_column: int
            Column where j_alpha gene information is stored.

        v_beta_column: int
            Column where v_beta gene information is stored.

        d_beta_column: int
            Column where d_beta gene information is stored.

        j_beta_column: int
            Column where j_beta gene information is stored.


        Returns
        ---------------------------------------

        """

        if Load_Prev_Data is False:

            if aa_column_alpha is not None:
                self.use_alpha = True

            if aa_column_beta is not None:
                self.use_beta = True

            if v_alpha_column is not None:
                self.use_v_alpha = True

            if j_alpha_column is not None:
                self.use_j_alpha = True

            if v_beta_column is not None:
                self.use_v_beta = True

            if d_beta_column is not None:
                self.use_d_beta = True

            if j_beta_column is not None:
                self.use_j_beta = True


            #Determine classes based on directory names
            if classes is None:
                classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory,d))]
                classes = [f for f in classes if not f.startswith('.')]

            self.lb = LabelEncoder()
            self.lb.fit(classes)
            self.classes = self.lb.classes_

            if p is None:
                p = Pool(n_jobs)

            if sep == '\t':
                ext = '*.tsv'
            elif sep == ',':
                ext = '*.csv'
            else:
                print('Not Valid Delimiter')
                return

            #Get data from tcr-seq files
            alpha_sequences = []
            beta_sequences = []
            v_beta = []
            d_beta = []
            j_beta = []
            v_alpha = []
            j_alpha = []
            label_id = []
            file_id = []
            freq = []
            counts=[]
            file_list = []
            for type in self.classes:
                files_read = glob.glob(os.path.join(directory, type, ext))
                num_ins = len(files_read)
                args = list(zip(files_read,
                                [type_of_data_cut] * num_ins,
                                [data_cut] * num_ins,
                                [aa_column_alpha] * num_ins,
                                [aa_column_beta] * num_ins,
                                [count_column] * num_ins,
                                [sep] * num_ins,
                                [self.max_length]*num_ins,
                                [aggregate_by_aa]*num_ins,
                                [v_beta_column]*num_ins,
                                [d_beta_column]*num_ins,
                                [j_beta_column]*num_ins,
                                [v_alpha_column]*num_ins,
                                [j_alpha_column]*num_ins))

                DF = p.starmap(Get_DF_Data, args)

                for df, file in zip(DF, files_read):
                    if aa_column_alpha is not None:
                        alpha_sequences += df['alpha'].tolist()
                    if aa_column_beta is not None:
                        beta_sequences += df['beta'].tolist()

                    if v_alpha_column is not None:
                        v_alpha += df['v_alpha'].tolist()

                    if j_alpha_column is not None:
                        j_alpha += df['j_alpha'].tolist()

                    if v_beta_column is not None:
                        v_beta += df['v_beta'].tolist()

                    if d_beta_column is not None:
                        d_beta += df['d_beta'].tolist()

                    if j_beta_column is not None:
                        j_beta += df['j_beta'].tolist()

                    label_id += [type] * len(df)
                    file_id += [file.split('/')[-1]] * len(df)
                    file_list.append(file.split('/')[-1])
                    freq += df['Frequency'].tolist()
                    counts += df['counts'].tolist()

            alpha_sequences = np.asarray(alpha_sequences)
            beta_sequences = np.asarray(beta_sequences)
            v_beta = np.asarray(v_beta)
            d_beta = np.asarray(d_beta)
            j_beta = np.asarray(j_beta)
            v_alpha = np.asarray(v_alpha)
            j_alpha = np.asarray(j_alpha)
            label_id = np.asarray(label_id)
            file_id = np.asarray(file_id)
            freq = np.asarray(freq)
            counts = np.asarray(counts)

            #transform sequences into numerical space
            if aa_column_alpha is not None:
                args = list(zip(alpha_sequences, [self.aa_idx] * len(alpha_sequences), [self.max_length] * len(alpha_sequences)))
                result = p.starmap(Embed_Seq_Num, args)
                sequences_num = np.vstack(result)
                X_Seq_alpha = np.expand_dims(sequences_num, 1)

            if aa_column_beta is not None:
                args = list(zip(beta_sequences, [self.aa_idx] * len(beta_sequences),  [self.max_length] * len(beta_sequences)))
                result = p.starmap(Embed_Seq_Num, args)
                sequences_num = np.vstack(result)
                X_Seq_beta = np.expand_dims(sequences_num, 1)


            if p is None:
                p.close()
                p.join()

            if self.use_alpha is False:
                X_Seq_alpha = np.zeros(shape=[len(label_id)])
                alpha_sequences = np.asarray([None]*len(label_id))

            if self.use_beta is False:
                X_Seq_beta = np.zeros(shape=[len(label_id)])
                beta_sequences = np.asarray([None]*len(label_id))

            #transform v/d/j genes into categorical space
            num_seq = X_Seq_alpha.shape[0]
            if self.use_v_beta is True:
                self.lb_v_beta = LabelEncoder()
                v_beta_num = self.lb_v_beta.fit_transform(v_beta)
            else:
                self.lb_v_beta = LabelEncoder()
                v_beta_num = np.zeros(shape=[num_seq])
                v_beta = np.asarray([None]*len(label_id))

            if self.use_d_beta is True:
                self.lb_d_beta = LabelEncoder()
                d_beta_num = self.lb_d_beta.fit_transform(d_beta)
            else:
                self.lb_d_beta = LabelEncoder()
                d_beta_num = np.zeros(shape=[num_seq])
                d_beta = np.asarray([None]*len(label_id))

            if self.use_j_beta is True:
                self.lb_j_beta = LabelEncoder()
                j_beta_num = self.lb_j_beta.fit_transform(j_beta)
            else:
                self.lb_j_beta = LabelEncoder()
                j_beta_num = np.zeros(shape=[num_seq])
                j_beta = np.asarray([None]*len(label_id))

            if self.use_v_alpha is True:
                self.lb_v_alpha = LabelEncoder()
                v_alpha_num = self.lb_v_alpha.fit_transform(v_alpha)
            else:
                self.lb_v_alpha = LabelEncoder()
                v_alpha_num = np.zeros(shape=[num_seq])
                v_alpha = np.asarray([None]*len(label_id))

            if self.use_j_alpha is True:
                self.lb_j_alpha = LabelEncoder()
                j_alpha_num = self.lb_j_alpha.fit_transform(j_alpha)
            else:
                self.lb_j_alpha = LabelEncoder()
                j_alpha_num = np.zeros(shape=[num_seq])
                j_alpha = np.asarray([None]*len(label_id))

            with open(os.path.join(self.Name,self.Name) + '_Data.pkl', 'wb') as f:
                pickle.dump([X_Seq_alpha,X_Seq_beta, alpha_sequences,beta_sequences, label_id, file_id, freq,counts,
                             self.lb,file_list,self.use_alpha,self.use_beta,
                             self.lb_v_beta, self.lb_d_beta, self.lb_j_beta,self.lb_v_alpha,self.lb_j_alpha,
                             v_beta, d_beta,j_beta,v_alpha,j_alpha,
                             v_beta_num, d_beta_num, j_beta_num,v_alpha_num,j_alpha_num,
                             self.use_v_beta,self.use_d_beta,self.use_j_beta,self.use_v_alpha,self.use_j_alpha],f,protocol=4)

        else:
            with open(os.path.join(self.Name,self.Name) + '_Data.pkl', 'rb') as f:
                X_Seq_alpha,X_Seq_beta, alpha_sequences,beta_sequences, label_id, file_id, freq,counts,\
                self.lb,file_list,self.use_alpha,self.use_beta,\
                    self.lb_v_beta, self.lb_d_beta, self.lb_j_beta,self.lb_v_alpha,self.lb_j_alpha,\
                    v_beta, d_beta,j_beta,v_alpha,j_alpha,\
                    v_beta_num, d_beta_num, j_beta_num,v_alpha_num,j_alpha_num,\
                    self.use_v_beta,self.use_d_beta,self.use_j_beta,self.use_v_alpha,self.use_j_alpha = pickle.load(f)

        self.X_Seq_alpha = X_Seq_alpha
        self.X_Seq_beta = X_Seq_beta
        self.alpha_sequences = alpha_sequences
        self.beta_sequences = beta_sequences
        self.label_id = label_id
        self.file_id = file_id
        self.freq = freq
        self.counts = counts
        self.file_list = file_list
        self.v_beta = v_beta
        self.v_beta_num = v_beta_num
        self.d_beta = d_beta
        self.d_beta_num = d_beta_num
        self.j_beta = j_beta
        self.j_beta_num = j_beta_num
        self.v_alpha = v_alpha
        self.v_alpha_num = v_alpha_num
        self.j_alpha = j_alpha
        self.j_alpha_num = j_alpha_num
        print('Data Loaded')

    def Train_VAE(self,latent_dim=256,batch_size=10000,accuracy_min=None,Load_Prev_Data=False,suppress_output = False,
                  trainable_embedding=True,use_only_gene=False,use_only_seq=False,epochs_min=10,stop_criterion=0.0001):
        """
        Train Variational Autoencoder (VAE)

        This method trains the network and saves features values for sequences
        to create heatmaps.

        Inputs
        ---------------------------------------

        latent_dim: int
            Number of latent dimensions for VAE.

        batch_size: int
            Size of batch to be used for each training iteration of the net.

        accuracy_min: float
            Minimum reconstruction accuracy before terminating training.

        Load_Prev_Data: bool
            Load previous feature data from prior training.

        suppress_output: bool
            To suppress command line output with training statisitcs, set to True.

        trainable_embedding: bool
            Toggle to control whether a trainable embedding layer is used or native
            one-hot representation for convolutional layers.

        use_only_gene: bool
            To only use gene-usage features, set to True. This will turn off features from
            the sequences.

        use_only_seq: bool
            To only use sequence feaures, set to True. This will turn off features learned
            from gene usage.

        epochs_min: int
            The minimum number of epochs to train the autoencoder.

        stop_criterion: float
            Minimum percent decrease in determined interval (below) to continue
            training. Used as early stopping criterion.

        Returns

        self.vae_features: array
            An array that contains n x latent_dim containing features for all sequences

        ---------------------------------------

        """

        if Load_Prev_Data is False:
            with tf.device(self.device):
                graph_model_AE = tf.Graph()
                with graph_model_AE.as_default():
                    if self.use_alpha is True:
                        X_Seq_alpha = tf.placeholder(tf.int64, shape=[None, self.X_Seq_alpha.shape[1], self.X_Seq_alpha.shape[2]],name='Input_Alpha')
                        X_Seq_alpha_OH = tf.one_hot(X_Seq_alpha, depth=21)

                    if self.use_beta is True:
                        X_Seq_beta = tf.placeholder(tf.int64, shape=[None, self.X_Seq_beta.shape[1], self.X_Seq_beta.shape[2]],name='Input_Beta')
                        X_Seq_beta_OH = tf.one_hot(X_Seq_beta, depth=21)

                    embedding_dim_genes = 48
                    gene_features = []
                    X_v_beta, X_v_beta_OH, embedding_layer_v_beta,\
                    X_d_beta, X_d_beta_OH, embedding_layer_d_beta,\
                    X_j_beta, X_j_beta_OH, embedding_layer_j_beta, \
                    X_v_alpha, X_v_alpha_OH, embedding_layer_v_alpha, \
                    X_j_alpha, X_j_alpha_OH, embedding_layer_j_alpha, \
                    gene_features = Get_Gene_Features(self,embedding_dim_genes,gene_features)


                    training = tf.placeholder_with_default(False, shape=())
                    prob = tf.placeholder_with_default(0.0, shape=(), name='prob')

                    if trainable_embedding is True:
                        # AA Embedding
                        with tf.variable_scope('AA_Embedding'):
                            embedding_dim_aa = 64
                            embedding_layer_seq = tf.get_variable(name='Embedding_Layer_Seq',
                                                                  shape=[21, embedding_dim_aa])
                            embedding_layer_seq = tf.expand_dims(tf.expand_dims(embedding_layer_seq, axis=0), axis=0)
                            if self.use_alpha is True:
                                inputs_seq_embed_alpha = tf.squeeze(
                                    tf.tensordot(X_Seq_alpha_OH, embedding_layer_seq, axes=(3, 2)), axis=(3, 4))
                            if self.use_beta is True:
                                inputs_seq_embed_beta = tf.squeeze(
                                    tf.tensordot(X_Seq_beta_OH, embedding_layer_seq, axes=(3, 2)), axis=(3, 4))

                    else:
                        if self.use_alpha is True:
                            inputs_seq_embed_alpha = X_Seq_alpha_OH

                        if self.use_beta is True:
                            inputs_seq_embed_beta = X_Seq_beta_OH


                    # Convolutional Features
                    if self.use_alpha is True:
                        Seq_Features_alpha,indices_alpha = Convolutional_Features_AE(inputs_seq_embed_alpha, training=training, prob=prob,name='alpha_conv')
                    if self.use_beta is True:
                        Seq_Features_beta,indices_beta = Convolutional_Features_AE(inputs_seq_embed_beta, training=training, prob=prob,name='beta_conv')


                    Seq_Features = []
                    if self.use_alpha is True:
                        Seq_Features.append(Seq_Features_alpha)
                    if self.use_beta is True:
                        Seq_Features.append(Seq_Features_beta)

                    if Seq_Features:
                        Seq_Features = tf.concat(Seq_Features,axis=1)

                    if not isinstance(Seq_Features,list):
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


                    fc = tf.layers.dense(Features, 256)
                    fc = tf.layers.dense(fc, 128)
                    z_mean = tf.layers.dense(fc, latent_dim, activation=None, name='z_mean')
                    z_log_var = tf.layers.dense(fc, latent_dim, activation=tf.nn.softplus, name='z_log_var')
                    latent_costs = []
                    latent_costs.append(Latent_Loss(z_log_var,z_mean))

                    z = z_mean + tf.exp(z_log_var / 2) * tf.random_normal(tf.shape(z_mean), 0.0, 1.0, dtype=tf.float32)
                    z = tf.identity(z, name='z')

                    fc_up = tf.layers.dense(z, 128)
                    fc_up = tf.layers.dense(fc_up, 256)
                    fc_up_flat = fc_up
                    fc_up = tf.reshape(fc_up, shape=[-1, 1, 4, 64])

                    recon_losses = []
                    accuracies = []
                    if self.use_beta is True:
                        upsample1_beta = tf.layers.conv2d_transpose(fc_up, 128, (1, 3), (1, 2), activation=tf.nn.relu)
                        upsample2_beta = tf.layers.conv2d_transpose(upsample1_beta, 64, (1, 3), (1, 2), activation=tf.nn.relu)

                        if trainable_embedding is True:
                            upsample3_beta = tf.layers.conv2d_transpose(upsample2_beta, embedding_dim_aa, (1, 4),(1, 2), activation=tf.nn.relu)
                            embedding_layer_seq_back = tf.transpose(embedding_layer_seq, perm=(0, 1, 3, 2))
                            logits_AE_beta = tf.squeeze(tf.tensordot(upsample3_beta, embedding_layer_seq_back, axes=(3, 2)),axis=(3, 4), name='logits')
                        else:
                            logits_AE_beta = tf.layers.conv2d_transpose(upsample2_beta, 21, (1, 4),(1, 2), activation=tf.nn.relu)

                        recon_cost_beta = Recon_Loss(X_Seq_beta, logits_AE_beta)
                        recon_losses.append(recon_cost_beta)

                        predicted_beta = tf.squeeze(tf.argmax(logits_AE_beta, axis=3), axis=1)
                        actual_ae_beta = tf.squeeze(X_Seq_beta, axis=1)
                        w = tf.cast(tf.squeeze(tf.greater(X_Seq_beta, 0), 1), tf.float32)
                        correct_ae_beta = tf.reduce_sum(w * tf.cast(tf.equal(predicted_beta, actual_ae_beta), tf.float32),axis=1) / tf.reduce_sum(w, axis=1)

                        accuracy_beta = tf.reduce_mean(correct_ae_beta, axis=0)
                        accuracies.append(accuracy_beta)

                    if self.use_alpha is True:
                        upsample1_alpha = tf.layers.conv2d_transpose(fc_up, 128, (1, 3), (1, 2), activation=tf.nn.relu)
                        upsample2_alpha = tf.layers.conv2d_transpose(upsample1_alpha, 64, (1, 3), (1, 2),activation=tf.nn.relu)

                        if trainable_embedding is True:
                            upsample3_alpha = tf.layers.conv2d_transpose(upsample2_alpha, embedding_dim_aa, (1, 4), (1, 2),activation=tf.nn.relu)
                            embedding_layer_seq_back = tf.transpose(embedding_layer_seq, perm=(0, 1, 3, 2))
                            logits_AE_alpha = tf.squeeze(tf.tensordot(upsample3_alpha, embedding_layer_seq_back, axes=(3, 2)),axis=(3, 4), name='logits')
                        else:
                            logits_AE_alpha = tf.layers.conv2d_transpose(upsample2_alpha, 21, (1, 4), (1, 2),activation=tf.nn.relu)

                        recon_cost_alpha = Recon_Loss(X_Seq_alpha, logits_AE_alpha)
                        recon_losses.append(recon_cost_alpha)

                        predicted_alpha = tf.squeeze(tf.argmax(logits_AE_alpha, axis=3), axis=1)
                        actual_ae_alpha = tf.squeeze(X_Seq_alpha, axis=1)
                        w = tf.cast(tf.squeeze(tf.greater(X_Seq_alpha, 0), 1), tf.float32)
                        correct_ae_alpha = tf.reduce_sum(w * tf.cast(tf.equal(predicted_alpha, actual_ae_alpha), tf.float32), axis=1) / tf.reduce_sum(w, axis=1)
                        accuracy_alpha = tf.reduce_mean(correct_ae_alpha, axis=0)
                        accuracies.append(accuracy_alpha)

                    gene_loss = []
                    if self.use_v_beta is True:
                        v_beta_loss,v_beta_acc = Get_Gene_Loss(fc_up_flat,embedding_layer_v_beta,X_v_beta_OH)
                        gene_loss.append(v_beta_loss)
                        accuracies.append(v_beta_acc)

                    if self.use_d_beta is True:
                        d_beta_loss, d_beta_acc = Get_Gene_Loss(fc_up_flat,embedding_layer_d_beta,X_d_beta_OH)
                        gene_loss.append(d_beta_loss)
                        accuracies.append(d_beta_acc)

                    if self.use_j_beta is True:
                        j_beta_loss,j_beta_acc = Get_Gene_Loss(fc_up_flat,embedding_layer_j_beta,X_j_beta_OH)
                        gene_loss.append(j_beta_loss)
                        accuracies.append(j_beta_acc)

                    if self.use_v_alpha is True:
                        v_alpha_loss,v_alpha_acc = Get_Gene_Loss(fc_up_flat,embedding_layer_v_alpha,X_v_alpha_OH)
                        gene_loss.append(v_alpha_loss)
                        accuracies.append(v_alpha_acc)

                    if self.use_j_alpha is True:
                        j_alpha_loss,j_alpha_acc = Get_Gene_Loss(fc_up_flat,embedding_layer_j_alpha,X_j_alpha_OH)
                        gene_loss.append(j_alpha_loss)
                        accuracies.append(j_alpha_acc)


                    recon_losses = recon_losses + gene_loss
                    temp = []
                    for l in recon_losses:
                        l = l[:,tf.newaxis]
                        temp.append(l)
                    recon_losses = temp
                    recon_losses = tf.concat(recon_losses,1)

                    recon_cost = tf.reduce_sum(recon_losses)

                    latent_cost = 0
                    for u in latent_costs:
                        latent_cost += u

                    total_cost = [recon_losses,latent_cost[:,tf.newaxis]]
                    total_cost = tf.concat(total_cost,1)
                    total_cost = tf.reduce_sum(total_cost,1)
                    total_cost = tf.reduce_mean(total_cost)
                    num_acc = len(accuracies)
                    accuracy = 0
                    for a in accuracies:
                        accuracy += a
                    accuracy = accuracy/num_acc
                    latent_cost = tf.reduce_sum(latent_cost)


                    opt_ae = tf.train.AdamOptimizer().minimize(total_cost)

                    saver = tf.train.Saver()

            epochs = 50000
            iteration = 0
            tf.reset_default_graph()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(graph=graph_model_AE,config=config) as sess:
                sess.run(tf.global_variables_initializer())
                recon_loss_list = []
                for e in range(epochs):
                    accuracy_list = []
                    Vars = [self.X_Seq_alpha,self.X_Seq_beta,self.v_beta_num,self.d_beta_num,self.j_beta_num,self.v_alpha_num,self.j_alpha_num]
                    for vars in get_batches(Vars, batch_size=batch_size):
                        feed_dict = {training:True}
                        if self.use_alpha is True:
                            feed_dict[X_Seq_alpha] = vars[0]
                        if self.use_beta is True:
                            feed_dict[X_Seq_beta] = vars[1]

                        if self.use_v_beta is True:
                            feed_dict[X_v_beta] = vars[2]

                        if self.use_d_beta is True:
                            feed_dict[X_d_beta] = vars[3]

                        if self.use_j_beta is True:
                            feed_dict[X_j_beta] = vars[4]

                        if self.use_v_alpha is True:
                            feed_dict[X_v_alpha] = vars[5]

                        if self.use_j_alpha is True:
                            feed_dict[X_j_alpha] = vars[6]

                        train_loss, recon_loss, latent_loss, accuracy_check, _ = sess.run([total_cost, recon_cost, latent_cost, accuracy, opt_ae], feed_dict=feed_dict)
                        accuracy_list.append(accuracy_check)
                        iteration += 1
                        recon_loss_list.append(recon_loss)

                    if suppress_output is False:
                        print("Epoch = {}/{}".format(e, epochs),
                              "Total Loss: {:.5f}:".format(train_loss),
                              "Recon Loss: {:.5f}:".format(recon_loss),
                              "Latent Loss: {:5f}:".format(latent_loss),
                              "AE Accuracy: {:.5f}".format(accuracy_check))


                    if e > epochs_min:
                        if accuracy_min is not None:
                            if np.mean(accuracy_list[-10:]) > accuracy_min:
                                break
                        else:
                            a, b, c = -50, -45, -5
                            if (np.mean(recon_loss_list[a:b]) - np.mean(recon_loss_list[c:])) / np.mean(recon_loss_list[a:b]) < stop_criterion:
                                break



                features_list = []
                accuracy_list = []
                Vars = [self.X_Seq_alpha, self.X_Seq_beta, self.v_beta_num, self.d_beta_num, self.j_beta_num,self.v_alpha_num, self.j_alpha_num]

                for vars in get_batches(Vars, batch_size=batch_size, random=False):
                    feed_dict = {training: False}
                    if self.use_alpha is True:
                        feed_dict[X_Seq_alpha] = vars[0]
                    if self.use_beta is True:
                        feed_dict[X_Seq_beta] = vars[1]

                    if self.use_v_beta is True:
                        feed_dict[X_v_beta] = vars[2]

                    if self.use_d_beta is True:
                        feed_dict[X_d_beta] = vars[3]

                    if self.use_j_beta is True:
                        feed_dict[X_j_beta] = vars[4]

                    if self.use_v_alpha is True:
                        feed_dict[X_v_alpha] = vars[5]

                    if self.use_j_alpha is True:
                        feed_dict[X_j_alpha] = vars[6]


                    get = z_mean
                    features_ind, accuracy_check = sess.run([get, accuracy], feed_dict=feed_dict)
                    features_list.append(features_ind)
                    accuracy_list.append(accuracy_check)


                features = np.vstack(features_list)
                accuracy_list = np.hstack(accuracy_list)
                print('Reconstruction Accuracy: {:.5f}'.format(np.nanmean(accuracy_list)))

                embedding_layers = [embedding_layer_v_alpha,embedding_layer_j_alpha,embedding_layer_v_beta,embedding_layer_d_beta,embedding_layer_j_beta]
                embedding_names = ['v_alpha','j_alpha','v_beta','d_beta','j_beta']
                name_keep = []
                embedding_keep = []
                for n,layer in zip(embedding_names,embedding_layers):
                    if layer is not None:
                        embedding_keep.append(layer.eval())
                        name_keep.append(n)

                embed_dict = dict(zip(name_keep,embedding_keep))

                saver.save(sess,os.path.join(self.Name,'model','model.ckpt'))


            with open(os.path.join(self.Name,self.Name) + '_VAE_features.pkl', 'wb') as f:
                pickle.dump([features,embed_dict], f,protocol=4)

        else:
            with open(os.path.join(self.Name,self.Name) + '_VAE_features.pkl', 'rb') as f:
                features,embed_dict = pickle.load(f)


        self.features = features

        self.features = features
        keep=[]
        for i,column in enumerate(self.features.T,0):
            if len(np.unique(column)) > 1:
                keep.append(i)
        keep = np.asarray(keep)
        self.vae_features = self.features[:,keep]
        self.features = self.vae_features
        self.embed_dict = embed_dict
        print('Training Done')

    def Inference(self,alpha_sequences=None,beta_sequences=None,v_beta=None,d_beta=None,j_beta=None,
                  v_alpha=None,j_alpha=None,p=None,batch_size=10000):
        """
        Predicting features on new data

        This method allows a user to take a pre-trained autoencoder and generate feature values
        on new data.

        Inputs
        ---------------------------------------

        alpha_sequences: ndarray of strings
            A 1d array with the sequences for inference for the alpha chain.

        beta_sequences: ndarray of strings
            A 1d array with the sequences for inference for the beta chain.

        v_beta: ndarray of strings
            A 1d array with the v-beta genes for inference.

        d_beta: ndarray of strings
            A 1d array with the d-beta genes for inference.

        j_beta: ndarray of strings
            A 1d array with the j-beta genes for inference.

        v_alpha: ndarray of strings
            A 1d array with the v-alpha genes for inference.

        j_alpha: ndarray of strings
            A 1d array with the j-alpha genes for inference.

        p: multiprocessing pool object
            a pre-formed pool object can be passed to method for multiprocessing tasks.

        batch_size: int
            Batch size for inference.

        Returns

        features: array
            An array that contains n x latent_dim containing features for all sequences

        ---------------------------------------

        """


        inputs = [alpha_sequences,beta_sequences,v_beta,d_beta,j_beta,v_alpha,j_alpha]
        for i in inputs:
            if i is not None:
                len_input = len(i)
                break

        if p is None:
            p = Pool(40)

        if alpha_sequences is not None:
            args = list(zip(alpha_sequences, [self.aa_idx] * len(alpha_sequences), [self.max_length] * len(alpha_sequences)))
            result = p.starmap(Embed_Seq_Num, args)
            sequences_num = np.vstack(result)
            X_Seq_alpha = np.expand_dims(sequences_num, 1)
        else:
            X_Seq_alpha = np.zeros(shape=[len_input])
            alpha_sequences = np.asarray([None] * len_input)

        if beta_sequences is not None:
            args = list(zip(beta_sequences, [self.aa_idx] * len(beta_sequences), [self.max_length] * len(beta_sequences)))
            result = p.starmap(Embed_Seq_Num, args)
            sequences_num = np.vstack(result)
            X_Seq_beta = np.expand_dims(sequences_num, 1)
        else:
            X_Seq_beta = np.zeros(shape=[len_input])
            beta_sequences = np.asarray([None] * len_input)

        if v_beta is not None:
            v_beta_num = self.lb_v_beta.fit_transform(v_beta)
        else:
            v_beta_num = np.zeros(shape=[len_input])
            v_beta = np.asarray([None] * len_input)

        if d_beta is not None:
            d_beta_num = self.lb_d_beta.fit_transform(d_beta)
        else:
            d_beta_num = np.zeros(shape=[len_input])
            d_beta = np.asarray([None] * len_input)

        if j_beta is not None:
            j_beta_num = self.lb_j_beta.fit_transform(j_beta)
        else:
            j_beta_num = np.zeros(shape=[len_input])
            j_beta = np.asarray([None] * len_input)

        if v_alpha is not None:
            v_alpha_num = self.lb_v_alpha.fit_transform(v_alpha)
        else:
            v_alpha_num = np.zeros(shape=[len_input])
            v_alpha = np.asarray([None] * len_input)

        if j_alpha is not None:
            j_alpha_num = self.lb_j_alpha.fit_transform(j_alpha)
        else:
            j_alpha_num = np.zeros(shape=[len_input])
            j_alpha = np.asarray([None] * len_input)

        if p is None:
            p.close()
            p.join()

        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.import_meta_graph(os.path.join(self.Name,'model','model.ckpt.meta'))
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join(self.Name,'model')))
            graph = tf.get_default_graph()

            if self.use_alpha is True:
                X_Seq_alpha_v = graph.get_tensor_by_name('Input_Alpha:0')

            if self.use_beta is True:
                X_Seq_beta_v = graph.get_tensor_by_name('Input_Beta:0')

            if self.use_v_beta is True:
                X_v_beta  = graph.get_tensor_by_name('Input_V_Beta:0')

            if self.use_d_beta is True:
                X_d_beta  = graph.get_tensor_by_name('Input_D_Beta:0')

            if self.use_j_beta is True:
                X_j_beta  = graph.get_tensor_by_name('Input_J_Beta:0')

            if self.use_v_alpha is True:
                X_v_alpha  = graph.get_tensor_by_name('Input_V_Alpha:0')

            if self.use_j_alpha is True:
                X_j_alpha  = graph.get_tensor_by_name('Input_J_Alpha:0')

            z_mean = graph.get_tensor_by_name('z_mean/BiasAdd:0')

            features_list = []
            Vars = [X_Seq_alpha, X_Seq_beta, v_beta_num, d_beta_num, j_beta_num,
                    v_alpha_num, j_alpha_num]

            for vars in get_batches(Vars,batch_size=batch_size):
                feed_dict = {}
                if self.use_alpha is True:
                    feed_dict[X_Seq_alpha_v] = vars[0]
                if self.use_beta is True:
                    feed_dict[X_Seq_beta_v] = vars[1]

                if self.use_v_beta is True:
                    feed_dict[X_v_beta] = vars[2]

                if self.use_d_beta is True:
                    feed_dict[X_d_beta] = vars[3]

                if self.use_j_beta is True:
                    feed_dict[X_j_beta] = vars[4]

                if self.use_v_alpha is True:
                    feed_dict[X_v_alpha] = vars[5]

                if self.use_j_alpha is True:
                    feed_dict[X_j_alpha] = vars[6]

                get = z_mean
                features_ind = sess.run(get, feed_dict=feed_dict)
                features_list.append(features_ind)

            features = np.vstack(features_list)
            return features

    def HeatMap_Sequences(self,filename='Heatmap_Features.tif',sample_num=None,sample_num_per_seq=None,color_dict=None):
        """
        HeatMap of Sequences

        This method creates a heatmap/clustermap for sequences by latent features
        for either supervised deep learming method.

        Inputs
        ---------------------------------------

        filename: str
            Name of file to save heatmap.

        sample_num: int
            Number of events to randomly sample for heatmap.

        color_dict: dict
            Optional dictionary to provide specified colors for classes.

        Returns
        ---------------------------------------

        """

        if sample_num_per_seq is not None and sample_num is not None:
            print("sample_num_per_seq and sample_num cannot be assigned simultaneously")
            return

        if sample_num is not None:
            sel = np.random.choice(range(len(self.features)),sample_num,replace=False)
            self.features = self.features[sel]
            self.label_id = self.label_id[sel]
            self.file_id = self.file_id[sel]
            self.alpha_sequences = self.alpha_sequences[sel]
            self.beta_sequences = self.beta_sequences[sel]


        if sample_num_per_seq is not None:
            features_temp = []
            label_temp = []
            file_temp = []
            seq_temp_alpha = []
            seq_temp_beta = []
            for i in self.lb.classes_:
                sel = np.where(self.label_id==i)[0]
                sel = np.random.choice(sel,sample_num_per_seq,replace=False)
                features_temp.append(self.features[sel])
                label_temp.append(self.label_id[sel])
                file_temp.append(self.file_id[sel])
                seq_temp_alpha.append(self.alpha_sequences[sel])
                seq_temp_beta.append(self.beta_sequences[sel])

            self.features = np.vstack(features_temp)
            self.label_id = np.hstack(label_temp)
            self.file_id = np.hstack(file_temp)
            self.alpha_sequences = np.hstack(seq_temp_alpha)
            self.beta_sequences = np.hstack(seq_temp_beta)


        keep=[]
        for i,column in enumerate(self.features.T,0):
            if len(np.unique(column)) > 1:
                keep.append(i)
        keep = np.asarray(keep)
        self.features = self.features[:,keep]

        if color_dict is None:
            N=len(np.unique(self.label_id))
            HSV_tuples = [(x * 1.0 / N, 1.0, 0.5) for x in range(N)]
            np.random.shuffle(HSV_tuples)
            RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
            color_dict = dict(zip(np.unique(self.label_id), RGB_tuples))

        row_colors = [color_dict[x] for x in self.label_id]
        sns.set(font_scale=0.5)
        CM = sns.clustermap(self.features,standard_scale=1,row_colors=row_colors,cmap='bwr')
        ax = CM.ax_heatmap
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        plt.show()
        plt.savefig(os.path.join(self.directory_results,filename))

    def HeatMap_Files(self,filename='Heatmap_Files.tif',Weight_by_Freq=True,color_dict=None,labels=True,font_scale=1.0):
        """
        HeatMap of Samples

        This method creates a heatmap/clustermap for samples by latent features
        for either supervised deep learming method.

        Inputs
        ---------------------------------------

        filename: str
            Name of file to save heatmap.

        Weight_by_Freq: bool
            Option to weight each sequence used in aggregate measure
            of feature across sample by its frequency.

        color_dict: dict
            Optional dictionary to provide specified colors for classes.

        labels: bool
            Option to show names of samples on y-axis of heatmap.

        font_scale: float
            This parameter controls the font size of the row labels. If there are many rows, one can make this value
            smaller to get better labeling of the rows.

        Returns
        ---------------------------------------

        """
        sample_id = np.unique(self.file_id)

        vector = []
        file_label=[]
        for id in sample_id:
            sel = self.file_id == id
            sel_idx = self.features[sel]
            sel_freq = np.expand_dims(self.freq[sel], 1)
            if Weight_by_Freq is True:
                dist = np.expand_dims(np.sum(sel_idx * sel_freq, 0), 0)
            else:
                dist = np.expand_dims(np.mean(sel_idx, 0), 0)
            file_label.append(np.unique(self.label_id[sel])[0])
            vector.append(dist)

        vector = np.vstack(vector)

        if color_dict is None:
            N=len(np.unique(self.label_id))
            HSV_tuples = [(x * 1.0 / N, 1.0, 0.5) for x in range(N)]
            np.random.shuffle(HSV_tuples)
            RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
            color_dict = dict(zip(np.unique(self.label_id), RGB_tuples))


        row_colors = [color_dict[x] for x in file_label]


        dfs = pd.DataFrame(vector)
        dfs.set_index(sample_id, inplace=True)
        sns.set(font_scale=font_scale)
        CM = sns.clustermap(dfs, standard_scale=1, cmap='bwr', figsize=(12, 10), row_colors=row_colors)
        ax = CM.ax_heatmap
        ax.set_xticklabels('')
        if labels is False:
            ax.set_yticklabels('')
        plt.subplots_adjust(right=0.8)
        plt.show()
        plt.savefig(os.path.join(self.directory_results,filename))

    def Cluster(self,clustering_method = 'phenograph',t=None,criterion='distance',
                linkage_method='ward',write_to_sheets=False,sample=None,n_jobs=1):

        """
        Clustering Sequences by Latent Features

        This method clusters all sequences by learned latent features from
        either the VAE or the GAN. Hierarchical clustering is implemented
        from the scipy package.

        Inputs
        ---------------------------------------

        clustering_method: str
            Clustering algorithm to use to cluster TCR sequences. Options include
            phenograph, dbscan, or hierarchical. When using dbscan or hierarchical clustering,
            a variety of thresholds are used to find an optimimum silhoutte score before using a final
            clustering threshold when t value is not provided.

        t: float
            If t is provided, this is used as a distance threshold for hierarchical clustering or the eps
            value for dbscan.

        criterion: str
            Clustering criterion as allowed by fcluster function
            in scipy.cluster.hierarchy module. (Used in hierarchical clustering).

        linkage_method: str
            method parameter for linkage as allowed by scipy.cluster.hierarchy.linkage

        write_to_sheets: bool
            To write clusters to separate csv files in folder named 'Clusters' under results folder, set to True.
            Additionally, if set to True, a csv file will be written in results directory that contains the frequency contribution
            of each cluster to each sample.

        sample: int
            For large numbers of sequences, to obtain a faster clustering solution, one can sub-sample
            a number of sequences and then use k-nearest neighbors to assign other sequences.

        n_jobs:int
            Number of processes to use for parallel operations.

        Returns

        self.DFs: list of Pandas dataframes
            Clusters by sequences/label

        self.var: list
            Variance of lengths in each cluster

        self.Cluster_Frequencies: Pandas dataframe
            A dataframe containing the frequency contribution of each cluster to each sample.

        self.Cluster_Assignemnts: ndarray
            Array with cluster assignments by number.

        ---------------------------------------

        """
        # Normalize Features
        features = self.features

        if sample is not None:
            idx_sel = np.random.choice(range(len(features)),sample,replace=False)
            features_sel = features[idx_sel]
            distances = squareform(pdist(features_sel))

            if clustering_method == 'hierarchical':
                if t is None:
                    IDX = hierarchical_optimization(distances,features_sel,method=linkage_method,criterion=criterion)
                else:
                    Z = linkage(squareform(distances), method=linkage_method)
                    IDX = fcluster(Z, t, criterion=criterion)

            elif clustering_method == 'dbscan':
                if t is None:
                    IDX = dbscan_optimization(distances,features_sel)
                else:
                    IDX = DBSCAN(eps=t, metric='precomputed').fit_predict(distances)
                    IDX[IDX == -1] = np.max(IDX + 1)

            elif clustering_method == 'phenograph':
                IDX, _, _ = phenograph.cluster(features_sel, k=30,n_jobs=n_jobs)

            knn_class = sklearn.neighbors.KNeighborsClassifier(n_neighbors=30, n_jobs=n_jobs).fit(features_sel, IDX)
            IDX = knn_class.predict(features)
        else:
            distances = squareform(pdist(features))
            if clustering_method == 'hierarchical':
                if t is None:
                    IDX = hierarchical_optimization(distances,features,method=linkage_method,criterion=criterion)
                else:
                    Z = linkage(squareform(distances), method=linkage_method)
                    IDX = fcluster(Z, t, criterion=criterion)

            elif clustering_method =='dbscan':
                if t is None:
                    IDX = dbscan_optimization(distances,features)
                else:
                    IDX = DBSCAN(eps=t, metric='precomputed').fit_predict(distances)
                    IDX[IDX == -1] = np.max(IDX + 1)

            elif clustering_method == 'phenograph':
                IDX, _, _ = phenograph.cluster(features, k=30,n_jobs=n_jobs)


        DFs = []
        DF_Sum = pd.DataFrame()
        DF_Sum['File'] = self.file_list
        DF_Sum.set_index('File', inplace=True)
        var_list_alpha = []
        var_list_beta = []
        for i in np.unique(IDX):
            if i != -1:
                sel = IDX == i
                seq_alpha = self.alpha_sequences[sel]
                seq_beta = self.beta_sequences[sel]
                label = self.label_id[sel]
                file = self.file_id[sel]
                freq = self.freq[sel]

                if self.use_alpha is True:
                    len_sel = [len(x) for x in seq_alpha]
                    var = max(len_sel) - min(len_sel)
                    var_list_alpha.append(var)
                else:
                    var_list_alpha.append(0)


                if self.use_beta is True:
                    len_sel = [len(x) for x in seq_beta]
                    var = max(len_sel) - min(len_sel)
                    var_list_beta.append(var)
                else:
                    var_list_beta.append(0)


                df = pd.DataFrame()
                df['Alpha_Sequences'] = seq_alpha
                df['Beta_Sequences'] = seq_beta
                df['Labels'] = label
                df['File'] = file
                df['Frequency'] = freq
                df['V_alpha'] = self.v_alpha[sel]
                df['J_alpha'] = self.j_alpha[sel]
                df['V_beta'] = self.v_beta[sel]
                df['D_beta'] = self.d_beta[sel]
                df['J_beta'] = self.j_beta[sel]

                df_sum = df.groupby(by='File', sort=False).agg({'Frequency': 'sum'})

                DF_Sum['Cluster_' + str(i)] = df_sum

                DFs.append(df)

        DF_Sum.fillna(0.0, inplace=True)

        if write_to_sheets is True:
            if not os.path.exists(os.path.join(self.directory_results, 'Clusters')):
                os.makedirs(os.path.join(self.directory_results, 'Clusters'))
            else:
                shutil.rmtree(os.path.join(self.directory_results, 'Clusters'))
                os.makedirs(os.path.join(self.directory_results, 'Clusters'))

            for ii, df in enumerate(DFs, 1):
                df.to_csv(os.path.join(self.directory_results, 'Clusters', str(ii) + '.csv'), index=False)

            DF_Sum.to_csv(os.path.join(self.directory_results, 'Cluster_Frequencies_by_Sample.csv'))

        self.DFs = DFs
        self.Cluster_Frequencies = DF_Sum
        self.var_alpha = var_list_alpha
        self.var_beta = var_list_beta
        self.Cluster_Assignments = IDX
        print('Clustering Done')

    def Structural_Diversity(self,sample=None,n_jobs=1):
        """
        Structural Diversity Measurements

        This method first clusters sequences via the phenograph algorithm before computing
        the number of clusters and entropy of the data over these clusters to obtain a measurement
        of the structural diversity within a repertoire.

        Inputs
        ---------------------------------------

        sample: int
            For large numbers of sequences, to obtain a faster clustering solution, one can sub-sample
            a number of sequences and then use k-nearest neighbors to assign other sequences.

        n_jobs:int
            Number of processes to use for parallel operations.

        Returns

        self.Structural_Diversity_DF: Pandas dataframe
            A dataframe containing the number of clusters and entropy in each sample

        ---------------------------------------

        """

        if sample is not None:
            idx_sel = np.random.choice(range(len(self.features)), sample, replace=False)
            features_sel = self.features[idx_sel]
            IDX,_,_ = phenograph.cluster(features_sel,n_jobs=n_jobs)
            knn_class = sklearn.neighbors.KNeighborsClassifier(n_neighbors=30, n_jobs=n_jobs).fit(features_sel, IDX)
            IDX = knn_class.predict(self.features)
        else:
            IDX, _, _ = phenograph.cluster(self.features, k=30,n_jobs=n_jobs)

        DFs = []
        DF_Sum = pd.DataFrame()
        DF_Sum['File'] = self.file_list
        DF_Sum.set_index('File', inplace=True)
        for i in np.unique(IDX):
            if i != -1:
                sel = IDX == i
                seq_alpha = self.alpha_sequences[sel]
                seq_beta = self.beta_sequences[sel]
                label = self.label_id[sel]
                file = self.file_id[sel]
                freq = self.freq[sel]

                df = pd.DataFrame()
                df['Alpha_Sequences'] = seq_alpha
                df['Beta_Sequences'] = seq_beta
                df['Labels'] = label
                df['File'] = file
                df['Frequency'] = freq
                df['V_alpha'] = self.v_alpha[sel]
                df['J_alpha'] = self.j_alpha[sel]
                df['V_beta'] = self.v_beta[sel]
                df['D_beta'] = self.d_beta[sel]
                df['J_beta'] = self.j_beta[sel]

                df_sum = df.groupby(by='File', sort=False).agg({'Frequency': 'sum'})

                DF_Sum['Cluster_' + str(i)] = df_sum

                DFs.append(df)

        DF_Sum.fillna(0.0, inplace=True)

        labels = []
        num_clusters = []
        entropy_list = []
        for file in self.file_list:
            v = np.array(DF_Sum.loc[file].tolist())
            v = v[v > 0.0]
            entropy_list.append(entropy(v))
            num_clusters.append(len(v))
            labels.append(self.label_id[self.file_id == file][0])

        df_out = pd.DataFrame()
        df_out['Label'] = labels
        df_out['Entropy'] = entropy_list
        df_out['Num of Clusters'] = num_clusters

        self.Structural_Diversity_DF = df_out

    def Repertoire_Dendogram(self,distance_metric = 'KL',sample=None,n_jobs=1,color_dict=None,
                             dendrogram_radius = 0.32, repertoire_radius=0.4,linkage_method='ward',
                             gridsize=10,Load_Prev_Data=False):
        """
        Repertoire Dendrogram

        This method creates a visualization that shows and compares the distribution
        of the sample repertoires via UMAP and provided distance metric. The underlying
        algorithm first applied phenograph clustering to determine the proportions of the sample
        within a given cluster. Then a distance metric is used to compare how far two samples are
        based on their cluster proportions. Various metrics can be provided here such as KL-divergence,
        Correlation, and Euclidean.

        Inputs
        ---------------------------------------

        distance_metric = str
            Provided distance metric to determine repertoire-level distance from cluster proportions.
            Options include = (KL,correlation,euclidean,wasserstein,JS).

        sample: int
            For large numbers of sequences, to obtain a faster clustering solution, one can sub-sample
            a number of sequences and then use k-nearest neighbors to assign other sequences.

        n_jobs:int
            Number of processes to use for parallel operations.

        color_dict: dict
            Optional dictionary to provide specified colors for classes.

        dendrogram_radius: float
            The radius of the dendrogram in the figure. This will usually require some adjustment
            given the number of samples.

        repertoire_radius: float
            The radius of the repertoire plots in the figure. This will usually require some adjustment
            given the number of samples.

        linkage_method: str
            linkage method used by scipy's linkage function

        gridsize: int
            This parameter modifies the granularity of the hexbins for the repertoire density plots.

        Load_Prev_Data: bool
            If method has been run before, one can load previous data used to construct the figure for
            faster figure creation. This is helpful when trying to format the figure correctly and will require
            the user to run the method multiple times.

        Returns

        self.pairwise_distances: Pandas dataframe
            Pairwise distances of all samples
        ---------------------------------------

        """

        if Load_Prev_Data is False:
            X_2 = umap.UMAP().fit_transform(self.features)
            self.Cluster(sample=sample,n_jobs=n_jobs)
            prop = self.Cluster_Frequencies
            with open(os.path.join(self.Name,'dendro.pkl'),'wb') as f:
                pickle.dump([X_2,prop],f)
        else:
            with open(os.path.join(self.Name,'dendro.pkl'),'rb') as f:
                X_2,prop = pickle.load(f)

        if distance_metric == 'KL':
            func = sym_KL
        elif distance_metric == 'correlation':
            func = distance.correlation
        elif distance_metric == 'euclidean':
            func = distance.euclidean
        elif distance_metric == 'wasserstein':
            func = wasserstein_distance
        elif distance_metric == 'JS':
            func = distance.jensenshannon

        pairwise_distances = np.zeros(shape=[len(prop), len(prop)])
        eps = 1e-9
        prop += eps
        for ii, i in enumerate(prop.index, 0):
            for jj, j in enumerate(prop.index, 0):
                pairwise_distances[ii, jj] = func(prop.loc[i], prop.loc[j])

        labels = []
        for i in prop.index:
            labels.append(self.label_id[np.where(self.file_id == i)[0][0]])

        samples = prop.index.tolist()

        if color_dict is None:
            N=len(np.unique(self.label_id))
            HSV_tuples = [(x * 1.0 / N, 1.0, 0.5) for x in range(N)]
            np.random.shuffle(HSV_tuples)
            RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
            color_dict = dict(zip(np.unique(self.label_id), RGB_tuples))

        rad_plot(X_2,pairwise_distances,samples,labels,self.file_id,color_dict,
                 gridsize=gridsize,dg_radius=dendrogram_radius,linkage_method=linkage_method,
                 figsize=8,axes_radius=repertoire_radius)

    def KNN_Sequence_Classifier(self,k_values=list(range(1, 500, 25)),rep=5,plot_metrics=False,by_class=False,
                                plot_type='violin',metrics = ['Recall', 'Precision', 'F1_Score', 'AUC']):
        """
        K-Nearest Neighbor Sequence Classifier

        This method uses a K-Nearest Neighbor Classifier to assess the ability to predict a sequence
        label given its sequence features.The method returns AUC,Precision,Recall, and
        F1 Scores for all classes.

        Inputs
        ---------------------------------------

        k_values: list
            List of k for KNN algorithm to assess performance metrics across.

        rep:int
            Number of iterations to train KNN classifier for each k-value.

        plot_metrics: bool
            Toggle to show the performance metrics

        plot_type: str
            Type of plot as taken by seaborn.catplot for kind parameter:
            options include (strip,swarm,box,violin,boxen,point,bar,count)

        by_class: bool
            Toggle to show the performance metrics by class.
        
        metrics: list
            List of performance measures one wants to compute.
            options include AUC, Precision, Recall, F1_Score

        Returns

        self.KNN_Sequence_DF: Pandas dataframe
            Dataframe with all metrics of performance organized by the class label,
            metric (i.e. AUC), k-value (from k-nearest neighbors), and the value of the
            performance metric.
        ---------------------------------------

        """

        distances = squareform(pdist(self.features, metric='euclidean'))

        temp = []
        for v in k_values:
            temp.extend(rep * [v])
        k_values = temp
        class_list = []
        k_list = []
        metric_list = []
        val_list = []

        for k in k_values:
            classes, metric, value, k_l = KNN(distances, self.label_id, k=k, metrics=metrics)
            metric_list.extend(metric)
            val_list.extend(value)
            class_list.extend(classes)
            k_list.extend(k_l)

        df_out = pd.DataFrame()
        df_out['Classes'] = class_list
        df_out['Metric'] = metric_list
        df_out['Value'] = val_list
        df_out['k'] = k_list

        self.KNN_Sequence_DF = df_out

        if plot_metrics is True:
            if by_class is True:
                sns.catplot(data=df_out, x='Metric', y='Value',hue='Classes',kind=plot_type)
            else:
                sns.catplot(data=df_out, x='Metric', y='Value',kind=plot_type)

    def KNN_Repertoire_Classifier(self,distance_metric='KL',sample=None,n_jobs=1,plot_metrics=False,
                                  plot_type='violin',by_class=False,Load_Prev_Data=False,
                                  metrics = ['Recall', 'Precision', 'F1_Score', 'AUC']):
        """
        K-Nearest Neighbor Repertoire Classifier

        This method uses a K-Nearest Neighbor Classifier to assess the ability to predict a repertoire
        label given the structural distribution of the repertoire.The method returns AUC,Precision,Recall, and
        F1 Scores for all classes.

        Inputs
        ---------------------------------------

        distance_metric = str
            Provided distance metric to determine repertoire-level distance from cluster proportions.
            Options include = (KL,correlation,euclidean,wasserstein,JS).

        sample: int
            For large numbers of sequences, to obtain a faster clustering solution, one can sub-sample
            a number of sequences and then use k-nearest neighbors to assign other sequences.

        n_jobs:int
            Number of processes to use for parallel operations.

        plot_metrics: bool
            Toggle to show the performance metrics

        plot_type: str
            Type of plot as taken by seaborn.catplot for kind parameter:
            options include (strip,swarm,box,violin,boxen,point,bar,count)

        by_class: bool
            Toggle to show the performance metrics by class.

        Load_Prev_Data: bool
            If method has been run before, one can load previous data from clustering step to move to KNN
            step faster. Can be useful when trying different distance methods to find optimizal distance metric
            for a given dataset.

        metrics: list
            List of performance measures one wants to compute.
            options include AUC, Precision, Recall, F1_Score

        Returns

        self.KNN_Repertoire_DF: Pandas dataframe
            Dataframe with all metrics of performance organized by the class label,
            metric (i.e. AUC), k-value (from k-nearest neighbors), and the value of the
            performance metric.
        ---------------------------------------

        """
        if Load_Prev_Data is False:
            self.Cluster(sample=sample,n_jobs=n_jobs)
            prop = self.Cluster_Frequencies
            with open(os.path.join(self.Name,'KNN_sample.pkl'),'wb') as f:
                pickle.dump(prop,f,protocol=4)
        else:
            with open(os.path.join(self.Name,'KNN_sample.pkl'),'rb') as f:
                prop = pickle.load(f)

        if distance_metric == 'KL':
            func = sym_KL
        elif distance_metric == 'correlation':
            func = distance.correlation
        elif distance_metric == 'euclidean':
            func = distance.euclidean
        elif distance_metric == 'wasserstein':
            func = wasserstein_distance
        elif distance_metric == 'JS':
            func = distance.jensenshannon

        pairwise_distances = np.zeros(shape=[len(prop), len(prop)])
        eps = 1e-9
        prop += eps
        for ii, i in enumerate(prop.index, 0):
            for jj, j in enumerate(prop.index, 0):
                pairwise_distances[ii, jj] = func(prop.loc[i], prop.loc[j])

        k_values = list(range(1, len(pairwise_distances)))

        labels = []
        for i in prop.index:
            labels.append(self.label_id[np.where(self.file_id == i)[0][0]])

        class_list = []
        k_list = []
        metric_list = []
        val_list = []
        for k in k_values:
            classes, metric, value, k_l = KNN_samples(pairwise_distances, labels, k=k, metrics=metrics)
            metric_list.extend(metric)
            val_list.extend(value)
            class_list.extend(classes)
            k_list.extend(k_l)

        df_out = pd.DataFrame()
        df_out['Classes'] = class_list
        df_out['Metric'] = metric_list
        df_out['Value'] = val_list
        df_out['k'] = k_list

        self.KNN_Repertoire_DF = df_out

        if plot_metrics is True:
            if by_class is True:
                sns.catplot(data=df_out, x='Metric', y='Value',hue='Classes',kind=plot_type)
            else:
                sns.catplot(data=df_out, x='Metric', y='Value',kind=plot_type)

    def UMAP_Plot(self,by_label=False,by_cluster=False,by_file=False,freq_weight=False,show_legend=True,scale=100,
                  Load_Prev_Data=False,alpha=1.0):
        """
        UMAP vizualisation of TCR Sequences

        This method displays the sequences in a 2-dimensional UMAP where the user can color code points by
        prior computing clustering solution or by label. Size of points can also be made to be proportional to
        frequency of sequence within sample.

        Inputs
        ---------------------------------------

        by_label: bool
            To color the points by their label, set to True.

        by_cluster:bool
            To color the points by the prior computed clustering solution, set to True.

        freq_weight: bool
            To scale size of points proportionally to their frequency, set to True.

        show_legend: bool
            To display legend, set to True.

        scale: float
            To change size of points, change scale parameter. Is particularly useful
            when finding good display size when points are scaled by frequency.

        Load_Prev_Data: bool
            If method was run before, one can rerun this method with this parameter set
            to True to bypass recomputing the UMAP projection. Useful for generating
            different versions of the plot on the same UMAP representation.

        alpha: float
            Value between 0-1 that controls transparency of points.


        Returns

        ---------------------------------------

        """

        if Load_Prev_Data is False:
            X_2 = umap.UMAP().fit_transform(self.features)
            with open(os.path.join(self.Name,'umap.pkl'),'wb') as f:
                pickle.dump(X_2,f,protocol=4)
        else:
            with open(os.path.join(self.Name,'umap.pkl'),'rb') as f:
                X_2 = pickle.load(f)

        df_plot = pd.DataFrame()
        df_plot['x'] = X_2[:, 0]
        df_plot['y'] = X_2[:, 1]
        df_plot['Label'] = self.label_id
        df_plot['File'] = self.file_id
        IDX = self.Cluster_Assignments
        IDX[IDX==-1]= np.max(IDX)+1
        IDX = ['Cluster_'+str(I) for I in IDX]
        df_plot['Cluster'] = IDX

        if freq_weight is True:
            freq = self.freq
            s = freq*scale
        else:
            s = scale

        if show_legend is True:
            legend = 'full'
        else:
            legend = False

        if by_label is True:
            hue = 'Label'
        elif by_cluster is True:
            hue = 'Cluster'
        elif by_file is True:
            hue = 'File'
        else:
            hue=None

        sns.scatterplot(data=df_plot,x='x',y='y',s=s,hue=hue,legend=legend,alpha=alpha)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')























































































