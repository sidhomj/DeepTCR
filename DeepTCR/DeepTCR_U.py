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
                    v_beta_column=None,j_beta_column=None,d_beta_column=None):
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

        If both column integers are left to None, column with a header containing 'acid' is used as
            the amino acid column.

        count_column: int
            Column where counts are stored. If set to None, first column with data in integer datatype is used as the
            counts column.

        sep: str
            Type of delimiter used in file with TCRSeq data.

        aggregate_by_aa: bool
            Choose to aggregate sequences by unique amino-acid. Defaults to True. If set to False, will allow duplicates
            of the same amino acid sequence given it comes from different nucleotide clones.

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


            p.close()
            p.join()

            if (self.use_beta is True) and (self.use_alpha is False):
                X_Seq_alpha = np.zeros_like(X_Seq_beta)
                alpha_sequences = np.asarray([None]*len(X_Seq_beta))

            if (self.use_beta is False) and (self.use_alpha is True):
                X_Seq_beta = np.zeros_like(X_Seq_alpha)
                beta_sequences = np.asarray([None]*len(X_Seq_alpha))

            #transform v/d/j genes into categorical space
            num_seq = X_Seq_alpha.shape[0]
            if self.use_v_beta is True:
                self.lb_v_beta = LabelEncoder()
                v_beta_num = self.lb_v_beta.fit_transform(v_beta)
            else:
                self.lb_v_beta = LabelEncoder()
                v_beta_num = np.zeros(shape=[num_seq])

            if self.use_d_beta is True:
                self.lb_d_beta = LabelEncoder()
                d_beta_num = self.lb_d_beta.fit_transform(d_beta)
            else:
                self.lb_d_beta = LabelEncoder()
                d_beta_num = np.zeros(shape=[num_seq])

            if self.use_j_beta is True:
                self.lb_j_beta = LabelEncoder()
                j_beta_num = self.lb_j_beta.fit_transform(j_beta)
            else:
                self.lb_j_beta = LabelEncoder()
                j_beta_num = np.zeros(shape=[num_seq])

            if self.use_v_alpha is True:
                self.lb_v_alpha = LabelEncoder()
                v_alpha_num = self.lb_v_alpha.fit_transform(v_alpha)
            else:
                self.lb_v_alpha = LabelEncoder()
                v_alpha_num = np.zeros(shape=[num_seq])

            if self.use_j_alpha is True:
                self.lb_j_alpha = LabelEncoder()
                j_alpha_num = self.lb_j_alpha.fit_transform(j_alpha)
            else:
                self.lb_j_alpha = LabelEncoder()
                j_alpha_num = np.zeros(shape=[num_seq])


            with open(os.path.join(self.Name,self.Name) + '_Data.pkl', 'wb') as f:
                pickle.dump([X_Seq_alpha,X_Seq_beta, alpha_sequences,beta_sequences, label_id, file_id, freq,
                             self.lb,file_list,self.use_alpha,self.use_beta,
                             self.lb_v_beta, self.lb_d_beta, self.lb_j_beta,self.lb_v_alpha,self.lb_j_alpha,
                             v_beta, d_beta,j_beta,v_alpha,j_alpha,
                             v_beta_num, d_beta_num, j_beta_num,v_alpha_num,j_alpha_num,
                             self.use_v_beta,self.use_d_beta,self.use_j_beta,self.use_v_alpha,self.use_j_alpha],f,protocol=4)

        else:
            with open(os.path.join(self.Name,self.Name) + '_Data.pkl', 'rb') as f:
                X_Seq_alpha,X_Seq_beta, alpha_sequences,beta_sequences, label_id, file_id, freq,\
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

    def Train_VAE(self,latent_dim=256,batch_size=10000,accuracy_min=None,Load_Prev_Data=False,suppress_output = False):
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

                    # AA Embedding
                    with tf.variable_scope('AA_Embedding'):
                        embedding_dim_aa = 64
                        embedding_layer_seq = tf.get_variable(name='Embedding_Layer_Seq', shape=[21, embedding_dim_aa])
                        embedding_layer_seq = tf.expand_dims(tf.expand_dims(embedding_layer_seq, axis=0), axis=0)
                        if self.use_alpha is True:
                            inputs_seq_embed_alpha = tf.squeeze(tf.tensordot(X_Seq_alpha_OH, embedding_layer_seq, axes=(3, 2)), axis=(3, 4))
                        if self.use_beta is True:
                            inputs_seq_embed_beta = tf.squeeze(tf.tensordot(X_Seq_beta_OH, embedding_layer_seq, axes=(3, 2)), axis=(3, 4))


                    # Convolutional Features
                    if self.use_alpha is True:
                        Seq_Features_alpha = Convolutional_Features_AE(inputs_seq_embed_alpha, training=training, prob=prob,name='alpha_conv')
                    if self.use_beta is True:
                        Seq_Features_beta = Convolutional_Features_AE(inputs_seq_embed_beta, training=training, prob=prob,name='beta_conv')


                    if (self.use_alpha is True) and (self.use_beta is True):
                        Seq_Features = tf.concat((Seq_Features_alpha, Seq_Features_beta), axis=1)
                    elif (self.use_alpha is True) and (self.use_beta is False):
                        Seq_Features = Seq_Features_alpha
                    elif (self.use_alpha is False) and (self.use_beta is True):
                        Seq_Features = Seq_Features_beta

                    if not isinstance(gene_features,list):
                        Seq_Features = tf.concat((Seq_Features,gene_features),axis=1)

                    fc = tf.layers.dense(Seq_Features, 256)
                    fc = tf.layers.dense(fc, 128)

                    z_mean = tf.layers.dense(fc, latent_dim, activation=None, name='z_mean')
                    z_log_var = tf.layers.dense(fc, latent_dim, activation=tf.nn.softplus, name='z_log_var')

                    z = z_mean + tf.exp(z_log_var / 2) * tf.random_normal(tf.shape(z_mean), 0.0, 1.0, dtype=tf.float32)
                    z = tf.identity(z, name='z')

                    fc_up = tf.layers.dense(z, 128)
                    fc_up = tf.layers.dense(fc_up, 256)
                    fc_up_flat = fc_up
                    fc_up = tf.reshape(fc_up, shape=[-1, 1, 4, 64])

                    recon_losses = []
                    accuracies = []
                    if self.use_beta is True:
                        upsample1_beta = tf.layers.conv2d_transpose(fc_up, 12, (1, 3), (1, 2), activation=tf.nn.relu)
                        upsample2_beta = tf.layers.conv2d_transpose(upsample1_beta, 32, (1, 3), (1, 2), activation=tf.nn.relu)
                        upsample3_beta = tf.layers.conv2d_transpose(upsample2_beta, embedding_dim_aa, (1, 4), (1, 2),activation=tf.nn.relu)


                        embedding_layer_seq_back = tf.transpose(embedding_layer_seq, perm=(0, 1, 3, 2))
                        logits_AE_beta = tf.squeeze(tf.tensordot(upsample3_beta, embedding_layer_seq_back, axes=(3, 2)),axis=(3, 4), name='logits')

                        recon_cost_beta, latent_cost = AE_Loss(X_Seq_beta, logits_AE_beta, z_mean, z_log_var)
                        recon_losses.append(recon_cost_beta)

                        predicted_beta = tf.squeeze(tf.argmax(logits_AE_beta, axis=3), axis=1)
                        actual_ae_beta = tf.squeeze(X_Seq_beta, axis=1)
                        w = tf.cast(tf.squeeze(tf.greater(X_Seq_beta, 0), 1), tf.float32)
                        correct_ae_beta = tf.reduce_sum(w * tf.cast(tf.equal(predicted_beta, actual_ae_beta), tf.float32),axis=1) / tf.reduce_sum(w, axis=1)

                        accuracy_beta = tf.reduce_mean(correct_ae_beta, axis=0)
                        accuracies.append(accuracy_beta)

                    if self.use_alpha is True:
                        upsample1_alpha = tf.layers.conv2d_transpose(fc_up, 12, (1, 3), (1, 2), activation=tf.nn.relu)
                        upsample2_alpha = tf.layers.conv2d_transpose(upsample1_alpha, 32, (1, 3), (1, 2),activation=tf.nn.relu)
                        upsample3_alpha = tf.layers.conv2d_transpose(upsample2_alpha, embedding_dim_aa, (1, 4), (1, 2),activation=tf.nn.relu)


                        embedding_layer_seq_back = tf.transpose(embedding_layer_seq, perm=(0, 1, 3, 2))
                        logits_AE_alpha = tf.squeeze(tf.tensordot(upsample3_alpha, embedding_layer_seq_back, axes=(3, 2)),axis=(3, 4), name='logits')


                        recon_cost_alpha, latent_cost = AE_Loss(X_Seq_alpha, logits_AE_alpha, z_mean,z_log_var)
                        recon_losses.append(recon_cost_alpha)

                        predicted_alpha = tf.squeeze(tf.argmax(logits_AE_alpha, axis=3), axis=1)
                        actual_ae_alpha = tf.squeeze(X_Seq_alpha, axis=1)
                        w = tf.cast(tf.squeeze(tf.greater(X_Seq_alpha, 0), 1), tf.float32)
                        correct_ae_alpha = tf.reduce_sum(w * tf.cast(tf.equal(predicted_alpha, actual_ae_alpha), tf.float32), axis=1) / tf.reduce_sum(w, axis=1)
                        accuracy_alpha = tf.reduce_mean(correct_ae_alpha, axis=0)
                        accuracies.append(accuracy_alpha)

                    gene_loss = []
                    if self.use_v_beta is True:
                        v_beta_loss = Get_Gene_Loss(fc_up_flat,embedding_layer_v_beta,X_v_beta_OH)
                        gene_loss.append(v_beta_loss)

                    if self.use_d_beta is True:
                        d_beta_loss = Get_Gene_Loss(fc_up_flat,embedding_layer_d_beta,X_d_beta_OH)
                        gene_loss.append(d_beta_loss)

                    if self.use_j_beta is True:
                        j_beta_loss = Get_Gene_Loss(fc_up_flat,embedding_layer_j_beta,X_j_beta_OH)
                        gene_loss.append(j_beta_loss)

                    if self.use_v_alpha is True:
                        v_alpha_loss = Get_Gene_Loss(fc_up_flat,embedding_layer_v_alpha,X_v_alpha_OH)
                        gene_loss.append(v_alpha_loss)

                    if self.use_j_alpha is True:
                        j_alpha_loss = Get_Gene_Loss(fc_up_flat,embedding_layer_j_alpha,X_j_alpha_OH)
                        gene_loss.append(j_alpha_loss)


                    recon_losses = recon_losses + gene_loss
                    temp = []
                    for l in recon_losses:
                        l = l[:,tf.newaxis]
                        temp.append(l)
                    recon_losses = temp
                    recon_losses = tf.concat(recon_losses,1)

                    recon_cost = tf.reduce_sum(recon_losses)

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
                for e in range(epochs):
                    accuracy_list = []
                    Vars = [self.X_Seq_alpha,self.X_Seq_beta,self.v_beta_num,self.d_beta_num,self.j_beta_num,self.v_alpha_num,self.j_alpha_num]
                    for vars in get_batches_gen(Vars, batch_size=batch_size):
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

                    if suppress_output is False:
                        print("Epoch = {}/{}".format(e, epochs),
                              "Total Loss: {:.5f}:".format(train_loss),
                              "Recon Loss: {:.5f}:".format(recon_loss),
                              "Latent Loss: {:5f}:".format(latent_loss),
                              "AE Accuracy: {:.5f}".format(accuracy_check))


                    if accuracy_min is not None:
                        if np.mean(accuracy_list[-10:]) > accuracy_min:
                            break


                features_list = []
                accuracy_list = []
                Vars = [self.X_Seq_alpha, self.X_Seq_beta, self.v_beta_num, self.d_beta_num, self.j_beta_num,self.v_alpha_num, self.j_alpha_num]

                for vars in get_batches_gen(Vars, batch_size=batch_size, random=False):
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

                    features_ind, accuracy_check = sess.run([z_mean, accuracy], feed_dict=feed_dict)
                    features_list.append(features_ind)
                    accuracy_list.append(accuracy_check)


                features = np.vstack(features_list)
                accuracy_list = np.hstack(accuracy_list)
                print('Reconstruction Accuracy: {:.5f}'.format(np.nanmean(accuracy_list)))

            with open(os.path.join(self.Name,self.Name) + '_VAE_features.pkl', 'wb') as f:
                pickle.dump(features, f,protocol=4)

        else:
            with open(os.path.join(self.Name,self.Name) + '_VAE_features.pkl', 'rb') as f:
                features = pickle.load(f)


        self.features = features

        self.features = features
        keep=[]
        for i,column in enumerate(self.features.T,0):
            if len(np.unique(column)) > 1:
                keep.append(i)
        keep = np.asarray(keep)
        self.vae_features = self.features[:,keep]
        self.features = self.vae_features
        print('Training Done')

    def Train_GAN(self,Load_Prev_Data=False,batch_size=10000,it_min=50,latent_dim=256,suppress_output=False):
        """
        Train Generative Adversarial Network (GAN)

        This method trains the network and saves features values for sequences
        to create heatmaps.

        Inputs
        ---------------------------------------

        latent_dim: int
            Number of latent dimensions for GAN.

        batch_size: int
            Size of batch to be used for each training iteration of the net.

        it_min: int
            Minimum number of iterations for training the GAN.

        Load_Prev_Data: bool
            Load previous feature data from prior training.

        suppress_output: bool
            To suppress command line output with training statisitcs, set to True.

        Returns

        self.gan_features: array
            An array that contains n x latent_dim containing features for all sequences

        ---------------------------------------

        """
        if Load_Prev_Data is False:

            if (self.use_alpha is True) and (self.use_beta is True):
                latent_dim = latent_dim//2

            z_dim = 256
            graph_model = tf.Graph()
            j=21
            epochs = 500

            with tf.device(self.device):
                with graph_model.as_default():
                    # Setup Placeholders
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

                    # AA Embedding
                    with tf.variable_scope('AA_Embedding'):
                        embedding_dim_aa = 64
                        embedding_layer_seq = tf.get_variable(name='Embedding_Layer_Seq', shape=[j, embedding_dim_aa])
                        embedding_layer_seq = tf.expand_dims(tf.expand_dims(embedding_layer_seq, axis=0), axis=0)
                        if self.use_alpha is True:
                            inputs_seq_embed_alpha = tf.squeeze(tf.tensordot(X_Seq_alpha_OH, embedding_layer_seq, axes=(3, 2)), axis=(3, 4))
                        if self.use_beta is True:
                            inputs_seq_embed_beta = tf.squeeze(tf.tensordot(X_Seq_beta_OH, embedding_layer_seq, axes=(3, 2)), axis=(3, 4))

                    if self.use_alpha is True:
                        latent_real_alpha,indices_real_alpha = Convolutional_Features_GAN(inputs_seq_embed_alpha,training=training,prob=prob,units=latent_dim,name='alpha_conv')
                    if self.use_beta is True:
                        latent_real_beta,indices_real_beta = Convolutional_Features_GAN(inputs_seq_embed_beta,training=training,prob=prob,units=latent_dim,name='beta_conv')

                    if (self.use_alpha is True) and (self.use_beta is True):
                        latent_real = tf.concat((latent_real_alpha,latent_real_beta),axis=1)
                    elif (self.use_alpha is True) and (self.use_beta is False):
                        latent_real = latent_real_alpha
                    elif (self.use_alpha is False) and (self.use_beta is True):
                        latent_real = latent_real_beta

                    if not isinstance(gene_features, list):
                        latent_real = tf.concat((latent_real, gene_features), axis=1)


                    latent_real = tf.identity(latent_real,'latent_real')
                    ortho_loss = Get_Ortho_Loss(latent_real)
                    logits_real = tf.layers.dense(latent_real,1,name='logits_real')

                    if self.use_alpha is True:
                        inputs_z_alpha = tf.placeholder(tf.float32, shape=[None, z_dim])
                        gen_seq_alpha = generator(inputs_z_alpha, training=training, embedding_dim_aa=embedding_dim_aa, prob=prob,name='generator_alpha')
                        latent_fake_alpha, indices_fake_alpha = Convolutional_Features_GAN(gen_seq_alpha, reuse=True, prob=prob,training=training, units=latent_dim,name='alpha_conv')

                    if self.use_beta is True:
                        inputs_z_beta = tf.placeholder(tf.float32, shape=[None, z_dim])
                        gen_seq_beta = generator(inputs_z_beta, training=training, embedding_dim_aa=embedding_dim_aa,prob=prob, name='generator_beta')
                        latent_fake_beta, indices_fake_beta = Convolutional_Features_GAN(gen_seq_beta, reuse=True,prob=prob, training=training,units=latent_dim,name='beta_conv')

                    gene_features = []
                    if self.use_v_alpha is True:
                        gen_v_alpha = generator_genes(inputs_z_alpha,embedding_dim_genes,training=training,prob=prob,name='generator_v_alpha')
                        gene_features.append(gen_v_alpha)

                    if self.use_j_alpha is True:
                        gen_j_alpha = generator_genes(inputs_z_alpha,embedding_dim_genes,training=training,prob=prob,name='generator_j_alpha')
                        gene_features.append(gen_j_alpha)

                    if self.use_v_beta is True:
                        gen_v_beta = generator_genes(inputs_z_beta, embedding_dim_genes, training=training, prob=prob,
                                                      name='generator_v_beta')
                        gene_features.append(gen_v_beta)

                    if self.use_d_beta is True:
                        gen_d_beta = generator_genes(inputs_z_beta, embedding_dim_genes, training=training, prob=prob,
                                                     name='generator_d_beta')
                        gene_features.append(gen_d_beta)

                    if self.use_j_beta is True:
                        gen_j_beta = generator_genes(inputs_z_beta, embedding_dim_genes, training=training, prob=prob,
                                                     name='generator_j_beta')
                        gene_features.append(gen_j_beta)

                    if gene_features:
                        gene_features = tf.concat(gene_features, axis=1)

                    if (self.use_alpha is True) and (self.use_beta is True):
                        latent_fake = tf.concat((latent_fake_alpha,latent_fake_beta),axis=1)
                    elif (self.use_alpha is True) and (self.use_beta is False):
                        latent_fake = latent_fake_alpha
                    elif (self.use_alpha is False) and (self.use_beta is True):
                        latent_fake = latent_fake_beta

                    if not isinstance(gene_features, list):
                        latent_fake = tf.concat((latent_fake, gene_features), axis=1)

                    latent_fake = tf.identity(latent_fake, 'latent_fake')
                    logits_fake = tf.layers.dense(latent_fake, 1, name='logits_fake')

                    d_loss, g_loss = model_loss(logits_real, logits_fake,latent_real,latent_fake)

                    var_list = tf.trainable_variables()
                    var_train = [x for x in var_list if not x.name.startswith('generator')]
                    opt_d = tf.train.RMSPropOptimizer(learning_rate=0.0002).minimize(d_loss+ortho_loss, var_list=var_train)

                    var_train = [x for x in var_list if x.name.startswith('generator')]
                    opt_g = tf.train.RMSPropOptimizer(learning_rate=0.0002).minimize(g_loss, var_list=var_train)


                    saver = tf.train.Saver()

            tf.reset_default_graph()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(graph=graph_model,config=config) as sess:
                sess.run(tf.global_variables_initializer())
                drop_out_rate = 0.1

                d_loss_list = []
                g_loss_list = []
                step = 0
                find=False
                for e in range(epochs):
                    if suppress_output is False:
                        print('Epoch: {}'.format(e))

                    Vars =

                    for x_seq_a,x_seq_b in get_batches(self.X_Seq_alpha,self.X_Seq_beta,batch_size=batch_size,random=True):
                        step +=1

                        feed_dict = {training:True,prob:drop_out_rate}
                        if self.use_alpha is True:
                            feed_dict[X_Seq_alpha] = x_seq_a
                            batch_z_alpha = np.random.normal(size=(batch_size, z_dim))
                            feed_dict[inputs_z_alpha] = batch_z_alpha
                        if self.use_beta is True:
                            feed_dict[X_Seq_beta] = x_seq_b
                            batch_z_beta = np.random.normal(size=(batch_size, z_dim))
                            feed_dict[inputs_z_beta] = batch_z_beta

                        d_loss_i,__= sess.run([d_loss, opt_d], feed_dict=feed_dict)
                        d_loss_list.append(d_loss_i)
                        if suppress_output is False:
                            print("D_Loss = {} ".format(d_loss_i), end='', flush=True)

                        g_loss_i, _ = sess.run([g_loss, opt_g], feed_dict=feed_dict)
                        g_loss_list.append(g_loss_i)
                        if suppress_output is False:
                            print("G_Loss = {}".format(g_loss_i))

                        if step > it_min:
                            if np.mean(d_loss_list[-10:]) < 1.0:
                                a, b, c = -30, -20, -10
                                if (np.mean(g_loss_list[a:b]) - np.mean(g_loss_list[c:])) / np.mean(g_loss_list[a:b]) < 0.01:
                                    find = True
                                    break

                                if np.mean(g_loss_list[-10:]) > 2.0:
                                    find = True
                                    break


                    if find is True:
                        break


                latent_features = []
                for x_seq_a, x_seq_b in get_batches(self.X_Seq_alpha, self.X_Seq_beta, batch_size=batch_size):
                    feed_dict = {}
                    if self.use_alpha is True:
                        feed_dict[X_Seq_alpha] = x_seq_a
                    if self.use_beta is True:
                        feed_dict[X_Seq_beta] = x_seq_b

                    latent_i = sess.run(latent_real,feed_dict=feed_dict)
                    latent_features.append(latent_i)
                features = np.vstack(latent_features)


            with open(os.path.join(self.Name,self.Name) + '_GAN_features.pkl','wb') as f:
                pickle.dump(features,f,protocol=4)

        else:

            with open(os.path.join(self.Name,self.Name) + '_GAN_features.pkl','rb') as f:
                features = pickle.load(f)


        self.features = features
        keep=[]
        for i,column in enumerate(self.features.T,0):
            if len(np.unique(column)) > 1:
                keep.append(i)
        keep = np.asarray(keep)
        self.gan_features = self.features[:,keep]
        self.features = self.gan_features
        print('Training Done')

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

    def Cluster(self,t=100,criterion='distance',on=None,write_to_sheets=False,method='ward',metric='euclidean'):
        """
        Clustering Sequences by Latent Features

        This method clusters all sequences by learned latent features from
        either the VAE or the GAN. Hierarchical clustering is implemented
        from the scipy package.

        Inputs
        ---------------------------------------

        t: float/int
            Threshold parameter for clustering algorithm

        criterion: str
            Clustering criterion as allowed by fcluster function
            in scipy.cluster.hierarchy module.

        write_to_sheets: bool
            To write clusters to separate csv files in folder named 'Clusters' under results folder, set to True.
            Additionally, if set to True, a csv file will be written in results directory that contains the frequency contribution
            of each cluster to each sample.

        on: str
            Specificy which feature space to cluster on. Options are 'VAE','GAN','Both',None. If nothing is specified, the
            features from the last algorithm ran are used. If 'Both' is specified, a clustering solution is applied that merges
            the feature space from both unsupervised algorithms before clustering.

        method: str
            method parameter for linkage as allowed by scipy.cluster.hierarchy.linkage

        metric: str
            metric parameter for linkage as allowed by scipy.cluster.hierarchy.linkage

        Returns

        self.DFs: list of Pandas dataframes
            Clusters by sequences/label

        self.var: list
            Variance of lengths in each cluster

        self.Cluster_Frequencies: Pandas dataframe
            A dataframe containing the frequency contribution of each cluster to each sample.

        ---------------------------------------

        """
        SS = StandardScaler()

        # Normalize Features
        if on is None:
            features = SS.fit_transform(self.features)
        elif on is 'VAE':
            features = SS.fit_transform(self.vae_features)
        elif on is 'GAN':
            features = SS.fit_transform(self.gan_features)
        elif on is 'Both':
            vae_features = SS.fit_transform(self.vae_features)
            gan_features = SS.fit_transform(self.gan_features)
            features = np.concatenate((vae_features,gan_features),axis=1)

        # # Hierarchical Clustering
        Z = linkage(features, method=method, metric=metric)
        IDX = fcluster(Z, t, criterion=criterion)

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
        print('Clustering Done')




























































































