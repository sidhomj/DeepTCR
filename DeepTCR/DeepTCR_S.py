import sys
sys.path.append('../')
from DeepTCR.supervised_functions.utils import *
from DeepTCR.supervised_functions.Layers import *
from DeepTCR.supervised_functions.data_processing import *
from DeepTCR.supervised_functions.common_layers import *
import glob
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import itertools
from scipy.stats import ttest_ind

class DeepTCR_S(object):

    def __init__(self, Name, max_length=40,device='/gpu:0'):
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
        self.device = device
        self.condition_kernels = False
        self.use_beta = True
        self.use_alpha = False
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

    def Get_Data_SS(self, directory, Load_Prev_Data=False,classes=None,save_data=True,type_of_data_cut='Fraction_Response',
                    data_cut=1.0,n_jobs=40,aa_column_alpha=None, aa_column_beta=None, count_column = None,sep='\t',aggregate_by_aa=True,
                    v_alpha_column = None, j_alpha_column = None,
                    v_beta_column=None,j_beta_column=None,d_beta_column=None,p=None):

        """
        Get Data for Single Sequence Classification.

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

        save_data: bool
            Whether to save data to pickle file for future re-loading. Useful to set this to False when trying to
            do inference on new data.

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

        p: multiprocessing pool object
            For parellelized operations, one can pass a multiprocessing pool object
            to this method.

        Returns
        ---------------------------------------

        """

        if Load_Prev_Data is False:
            if aa_column_alpha is not None:
                self.use_alpha = True
            else:
                self.use_alpha = False

            if (aa_column_beta is None) and (aa_column_alpha is not None):
                self.use_beta = False
            else:
                self.use_beta = True

            if v_beta_column is not None:
                self.use_v_beta = True

            if d_beta_column is not None:
                self.use_d_beta = True

            if j_beta_column is not None:
                self.use_j_beta = True

            if classes is None:
                # Get names of classes from folders names
                classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory,d))]
                classes = [f for f in classes if not f.startswith('.')]


            self.lb = LabelEncoder()
            self.lb.fit(classes)
            self.classes = self.lb.classes_


            #Load sequences, labels, and files
            alpha_sequences = []
            beta_sequences = []
            v_beta = []
            d_beta = []
            j_beta = []
            v_alpha = []
            j_alpha = []
            label_id = []
            file_id = []
            counts = []
            file_list = []
            freq = []
            if p is None:
                p = Pool(n_jobs)

            if sep == '\t':
                ext = '*.tsv'
            elif sep == ',':
                ext = '*.csv'
            else:
                print('Not Valid Delimiter')
                return

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
                                [aggregate_by_aa] * num_ins,
                                [v_beta_column] * num_ins,
                                [d_beta_column] * num_ins,
                                [j_beta_column] * num_ins,
                                [v_alpha_column] * num_ins,
                                [j_alpha_column] * num_ins))

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

            Y = self.lb.transform(label_id)
            OH = OneHotEncoder(sparse=False)
            Y = OH.fit_transform(Y.reshape(-1,1))

            #Embed sequences into ints
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



            #Save Data
            if save_data is True:
                with open(os.path.join(self.Name, self.Name) + '_Data.pkl', 'wb') as f:
                    pickle.dump([X_Seq_alpha,X_Seq_beta, Y, alpha_sequences,beta_sequences, label_id, file_id,freq,counts,
                                 self.lb,file_list,self.use_alpha,self.use_beta,
                                 self.lb_v_beta, self.lb_d_beta, self.lb_j_beta,self.lb_v_alpha,self.lb_j_alpha,
                                 v_beta, d_beta,j_beta,v_alpha,j_alpha,
                                 v_beta_num, d_beta_num, j_beta_num,v_alpha_num,j_alpha_num,
                                 self.use_v_beta,self.use_d_beta,self.use_j_beta,self.use_v_alpha,self.use_j_alpha], f, protocol=4)

        else:
            #Load Data
            with open(os.path.join(self.Name,self.Name) + '_Data.pkl', 'rb') as f:
                X_Seq_alpha,X_Seq_beta,Y, alpha_sequences,beta_sequences, label_id, file_id,freq,counts,\
                self.lb,file_list,self.use_alpha,self.use_beta,\
                    self.lb_v_beta, self.lb_d_beta, self.lb_j_beta,self.lb_v_alpha,self.lb_j_alpha,\
                    v_beta, d_beta,j_beta,v_alpha,j_alpha,\
                    v_beta_num, d_beta_num, j_beta_num,v_alpha_num,j_alpha_num,\
                    self.use_v_beta,self.use_d_beta,self.use_j_beta,self.use_v_alpha,self.use_j_alpha = pickle.load(f)

        self.X_Seq_alpha = X_Seq_alpha
        self.X_Seq_beta = X_Seq_beta
        self.alpha_sequences = alpha_sequences
        self.beta_sequences = beta_sequences
        self.class_id = label_id
        self.sample_id = file_id
        self.freq = freq
        self.counts = counts
        self.sample_list = file_list
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

    def Get_Train_Valid_Test_SS(self,test_size=0.25,LOO=None):
        """
        Train/Valid/Test Splits.

        Divide data for train, valid, test set. Training is used to
        train model parameters, validation is used to set early stopping,
        and test acts as blackbox independent test set.

        Inputs
        ---------------------------------------
        test_size: float
            Fraction of sample to be used for valid and test set.

        LOO: int
            Number of sequences to leave-out in Leave-One-Out Cross-Validation

        Returns
        ---------------------------------------

        """
        Vars = [self.X_Seq_alpha,self.X_Seq_beta,self.alpha_sequences,self.beta_sequences,self.sample_id,self.class_id,
                self.v_beta_num,self.d_beta_num,self.j_beta_num,self.v_alpha_num,self.j_alpha_num,
                self.v_beta,self.d_beta,self.j_beta,self.v_alpha,self.j_alpha]

        self.train,self.valid,self.test = Get_Train_Valid_Test(Vars=Vars,Y=self.Y,test_size=test_size,regression=False,LOO=LOO)

    def Train_SS(self,batch_size = 1000, epochs_min = 10,stop_criterion=0.001,kernel=5,units=12,trainable_embedding=True,weight_by_class=False,
                 num_fc_layers=0,units_fc=12,drop_out_rate=0.0,suppress_output=False,use_only_seq=False,use_only_gene=False):
        """
        Train Single-Sequence Classifier

        This method trains the network and saves features values at the
        end of training for motif analysis.

        Inputs
        ---------------------------------------
        batch_size: int
            Size of batch to be used for each training iteration of the net.

        epochs_min: int
            Minimum number of epochs for training neural network.

        stop_criterion: float
            Minimum percent decrease in determined interval (below) to continue
            training. Used as early stopping criterion.

        kernel: int
            Size of convolutional kernel.

        units: int
            Number of filters to be used for convolutional kernel.

        trainable_embedding; bool
            Toggle to control whether a trainable embedding layer is used or native
            one-hot representation for convolutional layers.

        num_fc_layers: int
            Number of fully connected layers following convolutional layer.

        units_fc: int
            Number of nodes per fully-connected layers following convolutional layer.

        drop_out_rate: float
            drop out rate for fully connected layers

        suppress_output: bool
            To suppress command line output with training statisitcs, set to True.


        Returns
        ---------------------------------------

        """
        epochs = 10000
        graph_model = tf.Graph()
        with tf.device(self.device):
            with graph_model.as_default():
                if self.use_alpha is True:
                    X_Seq_alpha = tf.placeholder(tf.int64,shape=[None, self.X_Seq_alpha.shape[1], self.X_Seq_alpha.shape[2]],name='Input_Alpha')
                    X_Seq_alpha_OH = tf.one_hot(X_Seq_alpha, depth=21)

                if self.use_beta is True:
                    X_Seq_beta = tf.placeholder(tf.int64,shape=[None, self.X_Seq_beta.shape[1], self.X_Seq_beta.shape[2]],name='Input_Beta')
                    X_Seq_beta_OH = tf.one_hot(X_Seq_beta, depth=21)

                Y = tf.placeholder(tf.float64, shape=[None, self.Y.shape[1]])
                prob = tf.placeholder_with_default(0.0, shape=(), name='prob')

                embedding_dim_genes = 48
                gene_features = []
                X_v_beta, X_v_beta_OH, embedding_layer_v_beta, \
                X_d_beta, X_d_beta_OH, embedding_layer_d_beta, \
                X_j_beta, X_j_beta_OH, embedding_layer_j_beta, \
                X_v_alpha, X_v_alpha_OH, embedding_layer_v_alpha, \
                X_j_alpha, X_j_alpha_OH, embedding_layer_j_alpha, \
                gene_features = Get_Gene_Features(self, embedding_dim_genes, gene_features)


                if trainable_embedding is True:
                    # AA Embedding
                    with tf.variable_scope('AA_Embedding'):
                        embedding_dim_aa = 64
                        embedding_layer_seq = tf.get_variable(name='Embedding_Layer_Seq', shape=[21, embedding_dim_aa])
                        embedding_layer_seq = tf.expand_dims(tf.expand_dims(embedding_layer_seq, axis=0), axis=0)
                        if self.use_alpha is True:
                            inputs_seq_embed_alpha = tf.squeeze(tf.tensordot(X_Seq_alpha_OH, embedding_layer_seq, axes=(3, 2)), axis=(3, 4))
                        if self.use_beta is True:
                            inputs_seq_embed_beta = tf.squeeze(tf.tensordot(X_Seq_beta_OH, embedding_layer_seq, axes=(3, 2)), axis=(3, 4))

                else:
                    if self.use_alpha is True:
                        inputs_seq_embed_alpha = X_Seq_alpha_OH

                    if self.use_beta is True:
                        inputs_seq_embed_beta = X_Seq_beta_OH


                # Convolutional Features
                if self.use_alpha is True:
                    Seq_Features_alpha, Indices_alpha = Convolutional_Features(inputs_seq_embed_alpha, kernel=kernel, units=units,trainable_embedding=trainable_embedding,name='alpha_conv')
                if self.use_beta is True:
                    Seq_Features_beta, Indices_beta = Convolutional_Features(inputs_seq_embed_beta, kernel=kernel, units=units,trainable_embedding=trainable_embedding,name='beta_conv')

                Seq_Features = []
                if self.use_alpha is True:
                    Seq_Features.append(Seq_Features_alpha)
                if self.use_beta is True:
                    Seq_Features.append(Seq_Features_beta)

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
                        fc = tf.layers.dropout(fc,prob)
                        fc = tf.layers.dense(fc,units_fc,tf.nn.relu)


                logits = tf.layers.dense(fc, self.Y.shape[1])
                ortho_loss = Get_Ortho_Loss(Seq_Features)

                if weight_by_class is True:
                    class_weights = tf.constant([(1 / (np.sum(self.Y, 0) / np.sum(self.Y))).tolist()])
                    weights = tf.squeeze(tf.matmul(tf.cast(Y, dtype='float32'), class_weights, transpose_b=True), axis=1)
                    loss = tf.reduce_mean(weights*tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
                else:
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))

                loss = loss+ortho_loss
                opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

                with tf.name_scope('Accuracy_Measurements'):
                    predicted = tf.nn.softmax(logits, name='predicted')
                    correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(Y, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

                saver = tf.train.Saver()


        #Initialize Training
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=graph_model,config=config) as sess:
            sess.run(tf.global_variables_initializer())

            val_loss_total = []
            for e in range(epochs):
                train_loss = []
                train_accuracy = []
                predicted_list = []
                set = self.train
                Vars = [set[0],set[1],set[6],set[7],set[8],set[-1]]
                for vars in get_batches_gen(Vars, batch_size=batch_size, random=True):
                    feed_dict = {Y:vars[-1],prob:drop_out_rate}
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

                    loss_i, accuracy_i, _,predicted_i = sess.run([loss, accuracy, opt,predicted], feed_dict=feed_dict)
                    train_loss.append(loss_i)
                    train_accuracy.append(accuracy_i)
                    predicted_list.append(predicted_i)

                train_loss = np.mean(train_loss)
                train_accuracy = np.mean(train_accuracy)
                predicted_out = np.vstack(predicted_list)
                train_auc = roc_auc_score(self.train[-1],predicted_out)


                val_loss = []
                val_accuracy = []
                set = self.valid
                Vars = [set[0], set[1], set[6], set[7], set[8], set[-1]]
                for vars in get_batches_gen(Vars, batch_size=batch_size, random=False):
                    feed_dict = {Y:vars[-1]}
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

                    loss_i, accuracy_i = sess.run([loss, accuracy], feed_dict=feed_dict)
                    val_loss.append(loss_i)
                    val_accuracy.append(accuracy_i)


                val_loss = np.mean(val_loss)
                val_accuracy = np.mean(val_accuracy)
                val_loss_total.append(val_loss)

                test_loss = []
                test_accuracy = []
                predicted_list = []
                set = self.test
                Vars = [set[0], set[1], set[6], set[7], set[8], set[-1]]
                for vars in get_batches_gen(Vars, batch_size=batch_size, random=False):
                    feed_dict = {Y:vars[-1]}
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

                    loss_i, accuracy_i, predicted_i = sess.run([loss, accuracy, predicted], feed_dict=feed_dict)
                    test_loss.append(loss_i)
                    test_accuracy.append(accuracy_i)
                    predicted_list.append(predicted_i)

                test_loss = np.mean(test_loss)
                test_accuracy = np.mean(test_accuracy)
                predicted_out = np.vstack(predicted_list)
                self.y_pred = predicted_out
                self.y_test = self.test[-1]

                y_test2 = np.vstack(self.y_test)
                if (np.sum(y_test2[:,0])!=len(y_test2)) and (np.sum(y_test2[:,0])!=0):
                    auc = roc_auc_score(np.vstack(self.y_test),np.vstack(self.y_pred))
                else:
                    auc = 0.0

                if suppress_output is False:
                    print("Training_Statistics: \n",
                          "Epoch: {}/{}".format(e + 1, epochs),
                          "Training loss: {:.5f}".format(train_loss),
                          "Validation loss: {:.5f}".format(val_loss),
                          "Testing loss: {:.5f}".format(test_loss),
                          "Training Accuracy: {:.5}".format(train_accuracy),
                          "Validation Accuracy: {:.5}".format(val_accuracy),
                          'Training AUC: {:.5}'.format(train_auc),
                          "Testing AUC: {:.5}".format(auc))


                if e > epochs_min:
                    a, b, c = -10, -7, -3
                    if (np.mean(val_loss_total[a:b]) - np.mean(val_loss_total[c:])) / np.mean(val_loss_total[a:b]) < stop_criterion:
                        break


            alpha_features_list = []
            beta_features_list = []
            alpha_indices_list = []
            beta_indices_list = []
            for x_seq_a,x_seq_b, y in get_batches_model(self.X_Seq_alpha,self.X_Seq_beta, self.Y, batch_size=batch_size, random=False):
                if self.use_alpha is True:
                    feed_dict[X_Seq_alpha] = x_seq_a
                if self.use_beta is True:
                    feed_dict[X_Seq_beta] = x_seq_b

                if (self.use_alpha is True) and (self.use_beta is True):
                    features_i_alpha,indices_i_alpha,features_i_beta,indices_i_beta = sess.run([Seq_Features_alpha,Indices_alpha,Seq_Features_beta,Indices_beta],feed_dict=feed_dict)
                elif (self.use_alpha is True) and (self.use_beta is False):
                    features_i_alpha,indices_i_alpha = sess.run([Seq_Features_alpha,Indices_alpha],feed_dict=feed_dict)
                elif (self.use_alpha is False) and (self.use_beta is True):
                    features_i_beta,indices_i_beta = sess.run([Seq_Features_beta,Indices_beta],feed_dict=feed_dict)

                if self.use_alpha is True:
                    alpha_features_list.append(features_i_alpha)
                    alpha_indices_list.append(indices_i_alpha)

                if self.use_beta is True:
                    beta_features_list.append(features_i_beta)
                    beta_indices_list.append(indices_i_beta)


            if self.use_alpha is True:
                self.alpha_features = np.vstack(alpha_features_list)
                self.alpha_indices = np.vstack(alpha_indices_list)

            if self.use_beta is True:
                self.beta_features = np.vstack(beta_features_list)
                self.beta_indices = np.vstack(beta_indices_list)


            self.predicted[self.test[5]] += self.y_pred

            self.kernel = kernel
            #
            var_save = []
            if self.use_alpha is True:
                var_save.extend([self.alpha_features,self.alpha_indices,self.alpha_sequences])
            if self.use_beta is True:
                var_save.extend([self.beta_features,self.beta_indices,self.beta_sequences])

            var_save.extend([self.y_pred,self.y_test,self.kernel])

            with open(os.path.join(self.Name,self.Name) + '_features.pkl','wb') as f:
                pickle.dump(var_save,f)

            print('Done Training')


    def Representative_Sequences_SS(self,top_seq=10):
        """
        Identify most highly predicted sequences for each class for single sequence classifier.

        This method allows the user to query which sequences were most predicted to belong to a given class.

        Inputs
        ---------------------------------------

        top_seq: int
            The number of top sequences to show for each class.

        Returns

        self.Rep_Seq_SS: dictionary of dataframes
            This dictionary of dataframes holds for each class the top sequences and their respective
            probabiltiies for all classes. These dataframes can also be found in the results folder under Rep_Sequences_SS.

        ---------------------------------------


        """
        dir = 'Rep_Sequences_SS'
        dir = os.path.join(self.directory_results,dir)
        if not os.path.exists(dir):
            os.makedirs(dir)

        file_list = [f for f in os.listdir(dir)]
        [os.remove(os.path.join(dir, f)) for f in file_list]

        Rep_Seq = []
        keep = []
        df_temp = pd.DataFrame()
        df_temp['alpha'] = self.alpha_sequences
        df_temp['beta'] = self.beta_sequences
        df_temp['label'] = self.label_id
        df_temp['file'] = self.file_id
        for ii,sample in enumerate(self.lb.classes_,0):
            df_temp[sample] = self.predicted[:,ii]

        for ii,sample in enumerate(self.lb.classes_,0):
            df_temp.sort_values(by=sample,ascending=False,inplace=True)
            df_sample = df_temp[df_temp['label']==sample][0:top_seq]

            if not df_sample.empty:
                Rep_Seq.append(df_sample)
                df_sample.to_csv(os.path.join(dir,sample+'.csv'),index=False)
                keep.append(ii)

        self.Rep_Seq_SS = dict(zip(self.lb.classes_[keep],Rep_Seq))

    def Motif_Identification_SS(self,group,p_val_threshold=0.05):
        """
        Motif Identification for Single-Sequence Classifier

        This method looks for enriched features in the predetermined gropu
        and returns fasta files in directory to be used with "https://weblogo.berkeley.edu/logo.cgi"
        to produce seqlogos.

        Inputs
        ---------------------------------------
        group: string
            Class for analyzing enriched motifs.

        p_val_threshold: float
            Significance threshold for enriched features/motifs for
            Mann-Whitney UTest.

        Returns
        ---------------------------------------

        self.(alpha/beta)_group_features_ss: Pandas Dataframe
            Sequences used to determine motifs in fasta files
            are stored in this dataframe where column names represent
            the feature number.

        """
        #Get Saved Features, Indices, and Sequences
        with open(os.path.join(self.Name,self.Name) + '_features.pkl', 'rb') as f:
            var_load = pickle.load(f)

        if (self.use_alpha is True) and (self.use_beta is True):
            self.alpha_features, self.alpha_indices, self.alpha_sequences,\
            self.beta_features, self.beta_indices, self.beta_sequences,self.y_pred,self.y_test,self.kernel = var_load

        elif (self.use_alpha is True) and (self.use_beta is False):
            self.alpha_features, self.alpha_indices, self.alpha_sequences,self.y_pred,self.y_test,self.kernel = var_load

        elif (self.use_alpha is False) and (self.use_beta is True):
            self.beta_features, self.beta_indices, self.beta_sequences, self.y_pred, self.y_test, self.kernel = var_load


        group_num = np.where(self.lb.classes_ == group)[0][0]

        # Find diff expressed features
        idx_pos = self.Y[:, group_num] == 1
        idx_neg = self.Y[:, group_num] == 0

        if self.use_alpha is True:
            self.alpha_group_features_ss = Diff_Features(self.alpha_features, self.alpha_indices, self.alpha_sequences, 'alpha',
                                     p_val_threshold, idx_pos, idx_neg, self.directory_results, group, self.kernel)

        if self.use_beta is True:
            self.beta_group_features_ss = Diff_Features(self.beta_features, self.beta_indices, self.beta_sequences, 'beta',
                                     p_val_threshold, idx_pos, idx_neg, self.directory_results, group, self.kernel)


        print('Motif Identification Completed')

    def Monte_Carlo_CrossVal_SS(self,fold=5,test_size=0.25,LOO=None,epochs_min=10,batch_size=1000,stop_criterion=0.001,kernel=5,units=12,
                                trainable_embedding=True,weight_by_class=False,num_fc_layers=0,units_fc=12,drop_out_rate=0.0,suppress_output=False):

        '''
        Monte Carlo Cross-Validation for Single-Sequence Classifier

        If the number of sequences is small but training the single-sequence classifier, one
        can use Monte Carlo Cross Validation to train a number of iterations before assessing
        predictive performance.After this method is run, the AUC_Curve method can be run to
        assess the overall performance.

        Inputs
        ---------------------------------------
        fold: int
            Number of iterations for Cross-Validation

        test_size: float
            Fraction of sample to be used for valid and test set.

        LOO: int
            Number of sequences to leave-out in Leave-One-Out Cross-Validation

        epochs_min: int
            Minimum number of epochs for training neural network.

        batch_size: int
            Size of batch to be used for each training iteration of the net.


        stop_criterion: float
            Minimum percent decrease in determined interval (below) to continue
            training. Used as early stopping criterion.

        kernel: int
            Size of convolutional kernel.

        units: int
            Number of filters to be used for convolutional kernel.

        trainable_embedding; bool
            Toggle to control whether a trainable embedding layer is used or native
            one-hot representation for convolutional layers.

        num_fc_layers: int
            Number of fully connected layers following convolutional layer.

        units_fc: int
            Number of nodes per fully-connected layers following convolutional layer.

        drop_out_rate: float
            drop out rate for fully connected layers

        suppress_output: bool
            To suppress command line output with training statisitcs, set to True.


        Returns
        ---------------------------------------


        '''

        y_pred = []
        y_test = []
        for i in range(0, fold):
            if suppress_output is False:
                print(i)
            self.Get_Train_Valid_Test_SS(test_size=test_size, LOO=LOO)
            self.Train_SS(epochs_min=epochs_min, batch_size=batch_size,stop_criterion=stop_criterion,
                          kernel=kernel,units=units,weight_by_class=weight_by_class,
                          trainable_embedding=trainable_embedding,num_fc_layers=num_fc_layers,
                          units_fc=units_fc,drop_out_rate=drop_out_rate,suppress_output=suppress_output)

            y_test.append(self.y_test)
            y_pred.append(self.y_pred)

            if i == 0:
                predicted = np.zeros_like(self.predicted)
                counts = np.zeros_like(self.predicted)

            predicted[self.test[5]] += self.y_pred
            counts[self.test[5]] += 1

            y_test2 = np.vstack(y_test)
            y_pred2 = np.vstack(y_pred)

            if suppress_output is False:
                print("Accuracy = {}".format(np.average(np.equal(np.argmax(y_pred2,1),np.argmax(y_test2,1)))))

                if self.y_test.shape[1] == 2:
                    if i > 0:
                        y_test2 = np.vstack(y_test)
                        if (np.sum(y_test2[:, 0]) != len(y_test2)) and (np.sum(y_test2[:, 0]) != 0):
                            print("AUC = {}".format(roc_auc_score(np.vstack(y_test), np.vstack(y_pred))))


        self.y_test = np.vstack(y_test)
        self.y_pred = np.vstack(y_pred)
        self.predicted = np.divide(predicted,counts, out = np.zeros_like(predicted), where = counts != 0)
        print('Monte Carlo Simulation Completed')


    def K_Fold_CrossVal_SS(self,folds=None,epochs_min=10,batch_size=1000,stop_criterion=0.001,kernel=5,units=12,
                           trainable_embedding=True,weight_by_class=False,num_fc_layers=0,units_fc=12,drop_out_rate=0.0,suppress_output=False,
                           iterations=None):
        '''
        K_Fold Cross-Validation for Single-Sequence Classifier

        If the number of sequences is small but training the single-sequence classifier, one
        can use K_Fold Cross Validation to train on all but one before assessing
        predictive performance.After this method is run, the AUC_Curve method can be run to
        assess the overall performance.

        Inputs
        ---------------------------------------

        folds: int
            Number of Folds

        epochs_min: int
            Minimum number of epochs for training neural network.

        batch_size: int
            Size of batch to be used for each training iteration of the net.

        stop_criterion: float
            Minimum percent decrease in determined interval (below) to continue
            training. Used as early stopping criterion.

        kernel: int
            Size of convolutional kernel.

        units: int
            Number of filters to be used for convolutional kernel.

        trainable_embedding; bool
            Toggle to control whether a trainable embedding layer is used or native
            one-hot representation for convolutional layers.

        num_fc_layers: int
            Number of fully connected layers following convolutional layer.

        units_fc: int
            Number of nodes per fully-connected layers following convolutional layer.

        drop_out_rate: float
            drop out rate for fully connected layers

        suppress_output: bool
            To suppress command line output with training statisitcs, set to True.

        iterations: int
            Option to specify how many iterations one wants to complete before
            terminating training. Useful for very large datasets.


        Returns
        ---------------------------------------

        '''

        if folds is None:
            folds = len(self.Y)

        #Create Folds
        idx = list(range(len(self.Y)))
        idx_left = idx
        file_per_sample = len(self.Y) // folds
        test_idx = []
        for ii in range(folds):
            if ii != folds-1:
                idx_sel = np.random.choice(idx_left, size=file_per_sample, replace=False)
            else:
                idx_sel = idx_left

            test_idx.append(idx_sel)
            idx_left = np.setdiff1d(idx_left, idx_sel)


        y_test = []
        y_pred = []
        for ii in range(folds):
            if suppress_output is False:
                print(ii)
            train_idx = np.setdiff1d(idx,test_idx[ii])

            Vars = [self.X_Seq_alpha, self.X_Seq_beta, self.alpha_sequences, self.beta_sequences,
                    self.file_id, self.seq_index,
                    self.v_beta_num, self.d_beta_num, self.j_beta_num, self.v_beta, self.d_beta, self.j_beta]
            self.train, self.test = Get_Train_Test(Vars=Vars,train_idx=train_idx,test_idx = test_idx[ii],Y=self.Y)
            self.valid = self.test
            self.LOO = True

            self.Train_SS(epochs_min=epochs_min, batch_size=batch_size,stop_criterion=stop_criterion,
                          kernel=kernel,units=units,weight_by_class=weight_by_class,
                          trainable_embedding=trainable_embedding,num_fc_layers=num_fc_layers,
                          units_fc=units_fc,drop_out_rate=drop_out_rate,suppress_output=suppress_output)


            y_test.append(self.y_test)
            y_pred.append(self.y_pred)

            y_test2 = np.vstack(y_test)
            y_pred2 = np.vstack(y_pred)

            if suppress_output is False:
                print("Accuracy = {}".format(np.average(np.equal(np.argmax(y_pred2, 1), np.argmax(y_test2, 1)))))

                if self.y_test.shape[1] == 2:
                    if ii > 0:
                        if (np.sum(y_test2[:, 0]) != len(y_test2)) and (np.sum(y_test2[:, 0]) != 0):
                            print("AUC = {}".format(roc_auc_score(np.vstack(y_test), np.vstack(y_pred))))


            if iterations is not None:
                if ii > iterations:
                    break

        self.y_test = np.vstack(y_test)
        self.y_pred = np.vstack(y_pred)
        test_idx = np.hstack(test_idx)
        self.predicted = np.zeros_like(self.predicted)
        self.predicted[test_idx] = self.y_pred

        print('K-fold Cross Validation Completed')



    def AUC_Curve(self,show_all=True,filename='AUC.tif',title=None):
        """
        AUC Curve for both Single Sequence and Whole File Models

        Inputs
        ---------------------------------------
        show_all: bool
            In the case there is only two classes, the method defaults
            to producing an curve for only one class. If one desires
            to see curves for all classes, set to True.

        filename: str
            Filename to save tif file of AUC curve.

        title: str
            Optional Title to put on ROC Curve.

        Returns
        ---------------------------------------

        """
        y_test = self.y_test
        y_pred = self.y_pred
        if (self.Y.shape[1] == 2) & (show_all is False):
            plt.figure()
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC_Curves')

            class_name = self.lb.classes_[1]
            roc_score = roc_auc_score(y_test[:, 1], y_pred[:, 1])
            fpr, tpr, _ = roc_curve(y_test[:, 1], y_pred[:, 1])
            plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (class_name, roc_score))


        else:
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')

            for ii, class_name in enumerate(self.lb.classes_, 0):
                roc_score = roc_auc_score(y_test[:, ii], y_pred[:,ii])
                fpr, tpr, _ = roc_curve(y_test[:, ii], y_pred[:,ii])
                plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (class_name, roc_score))

            plt.legend(loc="lower right")

        if title is not None:
            plt.title(title)

        plt.savefig(os.path.join(self.directory_results,filename))
        plt.show(block=False)

    def Get_Data_WF(self,directory,Load_Prev_Data=False,classes=None,save_data=True,type_of_data_cut='Fraction_Response',data_cut=1.0,n_jobs=40,
                    aa_column_alpha = None, aa_column_beta=None, count_column = None,sep='\t',aggregate_by_aa=True):
        """
        Get Data for Whole Sample Classification.

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

        save_data: bool
           Whether to save data to pickle file for later use.

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
            Type of delimiter used in file with TCRSeq data

        aggregate_by_aa: bool
            Choose to aggregate sequences by unique amino-acid. Defaults to True. If set to False, will allow duplicates
            of the same amino acid sequence given it comes from different nucleotide clones.


        Returns
        ---------------------------------------

        """

        if Load_Prev_Data is False:

            if aa_column_alpha is not None:
                self.use_alpha = True
            else:
                self.use_alpha = False

            if (aa_column_beta is None) and (aa_column_alpha is not None):
                self.use_beta = False
            else:
                self.use_beta = True

            if classes is None:
                classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory,d))]
                classes = [f for f in classes if not f.startswith('.')]

            self.lb = LabelEncoder()
            self.lb.fit(classes)
            self.classes = self.lb.classes_

            alpha_sequences = []
            beta_sequences = []
            labels = []
            files = []
            len_file = []
            freq = []
            p = Pool(n_jobs)

            if sep == '\t':
                ext = '*.tsv'
            elif sep == ',':
                ext = '*.csv'
            else:
                print('Not Valid Delimiter')
                return

            for type in self.classes:
                files_read = glob.glob(os.path.join(directory, type, ext))
                num_ins = len(files_read)
                args = list(zip(files_read,
                                [type_of_data_cut] * num_ins,
                                [data_cut] * num_ins,
                                [aa_column_alpha] * num_ins,
                                [aa_column_beta] * num_ins,
                                [count_column]*num_ins,
                                [sep]*num_ins,
                                [self.max_length]*num_ins,
                                [aggregate_by_aa] * num_ins))

                DF = p.starmap(Get_DF_Data,args)

                for df, file in zip(DF, files_read):
                    if aa_column_alpha is not None:
                        alpha_sequences.append(df['alpha'].tolist())
                    if aa_column_beta is not None:
                        beta_sequences.append(df['beta'].tolist())

                    if (aa_column_alpha is None) and (aa_column_beta is None):
                        beta_sequences.append(df['beta'].tolist())

                    freq.append(df['Frequency'].tolist())
                    labels.append(type)
                    files.append(file.split('/')[-1])
                    len_file.append(len(df))


            labels = np.asarray(labels)
            files = np.asarray(files)
            len_file = np.asarray(len_file)

            num_seq_per_instance = np.max(len_file)
            if self.use_alpha is True:
                alpha_sequences = np.asarray(pad_sequences(alpha_sequences,num_seq_per_instance))
            if self.use_beta is True:
                beta_sequences = np.asarray(pad_sequences(beta_sequences,num_seq_per_instance))

            freq = pad_freq(freq,num_seq_per_instance)
            X_Freq = np.vstack(freq)


            #Embed sequences into ints
            if self.use_alpha is True:
                all_seq = np.hstack(alpha_sequences).tolist()
                args = list(zip(all_seq, [self.aa_idx] * len(all_seq), [self.max_length] * len(all_seq)))
                result = p.starmap(Embed_Seq_Num, args)
                X_Seq_alpha = np.vstack(result).reshape(len(alpha_sequences), -1, self.max_length)

            if self.use_beta is True:
                all_seq = np.hstack(beta_sequences).tolist()
                args = list(zip(all_seq, [self.aa_idx] * len(all_seq), [self.max_length] * len(all_seq)))
                result = p.starmap(Embed_Seq_Num, args)
                X_Seq_beta = np.vstack(result).reshape(len(beta_sequences), -1, self.max_length)

            p.close()
            p.join()

            Y = self.lb.transform(labels)
            OH = OneHotEncoder(sparse=False)
            Y = OH.fit_transform(Y.reshape(-1,1))

            if (self.use_beta is True) and (self.use_alpha is False):
                X_Seq_alpha = np.zeros_like(X_Seq_beta)
                alpha_sequences = np.matlib.repmat(np.asarray([None]*X_Seq_beta.shape[1]),X_Seq_beta.shape[0],1)

            if (self.use_beta is False) and (self.use_alpha is True):
                X_Seq_beta = np.zeros_like(X_Seq_alpha)
                beta_sequences = np.matlib.repmat(np.asarray([None]*X_Seq_alpha.shape[1]),X_Seq_alpha.shape[0],1)

            if save_data is True:
                with open(os.path.join(self.Name,self.Name) + '_Data_WF.pkl', 'wb') as f:
                    pickle.dump([X_Seq_alpha,X_Seq_beta,alpha_sequences,beta_sequences,X_Freq, Y, labels, files, self.lb, self.use_alpha,self.use_beta], f, protocol=4)

        else:
            with open(os.path.join(self.Name,self.Name) + '_Data_WF.pkl', 'rb') as f:
                X_Seq_alpha, X_Seq_beta, alpha_sequences, beta_sequences, X_Freq, Y, labels, files, self.lb, self.use_alpha, self.use_beta = pickle.load(f)


        self.X_Seq_alpha = X_Seq_alpha
        self.X_Seq_beta = X_Seq_beta
        self.alpha_sequences = alpha_sequences
        self.beta_sequences = beta_sequences
        self.X_Freq = X_Freq
        self.Y = Y
        self.labels = labels
        self.label_id = np.matlib.repmat(labels,X_Seq_alpha.shape[1],1).T
        self.files = files
        self.files_id = np.matlib.repmat(files,X_Seq_alpha.shape[1],1).T
        seq_index = np.asarray(list(range(X_Seq_alpha.shape[0]*X_Seq_alpha.shape[1])))
        self.seq_index = np.reshape(seq_index,[X_Seq_alpha.shape[0],X_Seq_alpha.shape[1]])
        self.predicted = np.zeros((len(seq_index),len(self.lb.classes_)))
        print('Data Loaded')

    def One_V_All(self,one_v_all):
        """
        One_V_All Binary Classifier

        If one desires to create a binary classifier where they want to compare one cohort against all others, run this method
        to divide the data into two cohorts (class of interest vs all else).

        Inputs
        ---------------------------------------
        one_v_all: str
            To create binary classifier between class and all else, specify cohort to compare against all others.

        Returns
        ---------------------------------------

        """

        new_labels = []
        for i in self.labels:
            if i != one_v_all:
                new_labels.append('all')
            else:
                new_labels.append(i)
        self.labels = new_labels
        classes = list(set(self.labels))
        self.lb.fit(classes)
        self.classes = self.lb.classes_

        Y = self.lb.transform(self.labels)
        OH = OneHotEncoder(sparse=False)
        Y = OH.fit_transform(Y.reshape(-1, 1))
        self.Y = Y

    def Get_Train_Valid_Test_WF(self,test_size=0.2,LOO=None):
        """
        Train/Valid/Test Splits.

        Divide data for train, valid, test set. Training is used to
        train model parameters, validation is used to set early stopping,
        and test acts as blackbox independent test set. In the case that
        Leave-One-Out (LOO) is set to a value, the valid and test sets
        have the same data and early stopping is based on the training loss.

        Inputs
        ---------------------------------------
        test_size: float
            Fraction of sample to be used for valid and test set.

        LOO: int
            Number of samples to leave-out in Leave-One-Out Cross-Validation

        Returns
        ---------------------------------------

        """

        Vars = [self.X_Seq_alpha,self.X_Seq_beta, self.X_Freq,self.files,self.alpha_sequences,self.beta_sequences,self.seq_index]
        self.train, self.valid, self.test = Get_Train_Valid_Test(Vars=Vars, Y=self.Y, test_size=test_size, regression=False,LOO=LOO)
        self.LOO = LOO

    def Condition_Kernels(self,kernel=3,sample_batch_size=50,motif_batch_size=1000,top_kernels=10,weight_by_freq=False,sample=None):
        """
        DEPRECATED
        --------------------------------------
        Condition Kernels with differentially used motifs.

        This method determines in a brute-force heuristic which k-mers are differentially used in each cohort prior to
        training the classifier. This is particularly useful in the case for low-frequency motifs that may be difficult to
        find with gradient descent.

        Inputs
        ---------------------------------------
        kernel: int
            Size of convolutional kernel.

        sample_batch_size: int
            Size of sample batch to pass through network for inference.

        motif_batch_size: int
            Size of motif batch to pass through the network for inference.

        top_kernels: int
            Number of kernels to select per class to condition.

        weight_by_freq: bool
            Whether to use the frequency to determine enriched motifs.

        sample: int
            Number of motifs to sample to query for differential usage. For computationally more tractable results with
            high length kernels, it is best to sample some number of motifs to query.


        Returns
        ---------------------------------------

        """

        print('Function deprecated in this current version')
        return

        #Generate all k-mer motifs
        numbers = list(range(1,21))
        letters = list(self.aa_idx)
        OH = OneHotEncoder(sparse=False, n_values=21)
        motif_letters = [''.join(p) for p in itertools.product(letters,repeat=kernel)]
        motif_inputs = [np.expand_dims(OH.fit_transform(np.asarray(p).reshape(-1,1)),0) for p in itertools.product(numbers,repeat=kernel)]
        motif_inputs = np.vstack(motif_inputs)

        if sample is not None:
            idx = np.random.choice(range(len(motif_inputs)),sample,replace=False)
            motif_inputs = motif_inputs[idx,:,:]
            motif_letters = list(itertools.compress(motif_letters,idx))

        motif_inputs = np.expand_dims(motif_inputs,1)
        motif_inputs = np.transpose(motif_inputs,[1,2,3,0])

        graph_model = tf.Graph()
        with tf.device(self.device):
            with graph_model.as_default():
                X_Seq = tf.placeholder(tf.int64, shape=[None, None, self.X_Seq.shape[2]], name='Input_Seq')
                X_Seq_OH = tf.one_hot(X_Seq, depth=21)
                X_Seq_OH = X_Seq_OH * 2 - 1
                X_Freq = tf.placeholder(tf.float32, shape=[None, None], name='Input_Freq')
                Y = tf.placeholder(tf.int64, shape=[None, self.Y.shape[1]], name='Y')
                motifs = tf.placeholder(tf.float32,shape=[1,kernel,21,None])

                inputs_seq_embed = X_Seq_OH

                # Convolutional Features
                Seq_Features = Convolutional_Features_Test(inputs_seq_embed,motifs,bias=kernel-1)
                if weight_by_freq is True:
                    Seq_Features = tf.expand_dims(X_Freq, 2) * Seq_Features

                Seq_Features_Ind = tf.reshape(Seq_Features,[tf.shape(Seq_Features)[0]*tf.shape(Seq_Features)[1],tf.shape(Seq_Features)[-1]])
                Seq_Features_Agg = tf.reduce_sum(Seq_Features,1)

        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        plt.show(block=False)

        with tf.Session(graph=graph_model, config=config) as sess:
            sess.run(tf.global_variables_initializer())
            Seq_Features_List_1 = []
            for motif_input_batch in get_motif_batches(motif_inputs,batch_size=motif_batch_size,random=False):
                Seq_Features_List_2 = []
                for x_seq, x_freq, y in get_batches_model(self.train[0], self.train[1], self.train[-1], batch_size=sample_batch_size, random=False):
                    feed_dict = {X_Seq: x_seq, X_Freq: x_freq, Y: y,motifs:motif_input_batch}
                    seq_features_i,seq_features_ind_i = sess.run([Seq_Features_Agg,Seq_Features_Ind],feed_dict=feed_dict)
                    Seq_Features_List_2.append(seq_features_i)

                Seq_Features_List_2 = np.vstack(Seq_Features_List_2)
                Seq_Features_List_1.append(Seq_Features_List_2)

            Seq_Features_List_1 = np.hstack(Seq_Features_List_1)


            DFs = []
            IDX = []
            all_idx = list(range(len(self.train[-1])))
            train_1_count = len(self.train[-1])//2
            train_1_idx = np.random.choice(all_idx,train_1_count,replace=False)
            train_2_idx = np.setdiff1d(all_idx,train_1_idx)

            for type in np.unique(np.argmax(self.train[-1],-1)):
                #Diff Motifs in Train_1
                magnitudes = []
                magnitudes_div = []
                P_Val = []
                Seq_Features_Train_1 = Seq_Features_List_1[train_1_idx]
                sel_idx = self.train[-1][train_1_idx, type] == 1
                for ii in range(motif_inputs.shape[-1]):
                    a = Seq_Features_Train_1[sel_idx, ii]
                    b = Seq_Features_Train_1[~sel_idx, ii]
                    mag = np.mean(a) - np.mean(b)
                    mag_div = np.mean(a)/(np.mean(b)+1e-9)
                    stat, p = ttest_ind(a, b)
                    P_Val.append(p)
                    magnitudes.append(mag)
                    magnitudes_div.append(mag_div)

                df_out = pd.DataFrame()
                df_out['Motifs'] = motif_letters
                df_out['Mag'] = magnitudes
                df_out['Mag_Div'] = magnitudes_div
                df_out['P_Val'] = P_Val
                df_out.sort_values(by='P_Val', ascending=True, inplace=True)
                idx = df_out['P_Val'] < 0.05
                df_out = df_out[idx]
                idx = df_out['Mag'] > 0.0
                df_out = df_out[idx]
                df_out = df_out.reset_index()


                #Diff Motifs in Train_2
                magnitudes = []
                magnitudes_div = []
                P_Val = []
                Seq_Features_Train_2 = Seq_Features_List_1[train_2_idx]

                sel_idx = self.train[-1][train_2_idx, type] == 1
                for ii in range(motif_inputs.shape[-1]):
                    a = Seq_Features_Train_2[sel_idx, ii]
                    b = Seq_Features_Train_2[~sel_idx, ii]
                    mag = np.mean(a) - np.mean(b)
                    mag_div = np.mean(a) / (np.mean(b) + 1e-9)
                    stat, p = ttest_ind(a, b)
                    P_Val.append(p)
                    magnitudes.append(mag)
                    magnitudes_div.append(mag_div)

                df_out_val = pd.DataFrame()
                df_out_val['Motifs'] = motif_letters
                df_out_val['Mag'] = magnitudes
                df_out_val['Mag_Div'] = magnitudes_div
                df_out_val['P_Val'] = P_Val
                df_out_val.sort_values(by='P_Val', ascending=True, inplace=True)
                idx = df_out_val['P_Val'] < 0.05
                df_out_val = df_out_val[idx]
                idx = df_out_val['Mag'] > 0.0
                df_out_val = df_out_val[idx]
                df_out_val = df_out_val.reset_index()

                #Get Common Motifs
                df_final = df_out.merge(df_out_val,how='inner',on=['Motifs','index'])
                df_final['Avg_P'] = (df_final['P_Val_x']+df_final['P_Val_y'])/2
                df_final['Avg_Mag'] = (df_final['Mag_x']+df_final['Mag_y'])/2
                df_final.sort_values(by='Avg_Mag',ascending=False,inplace=True)
                df_final.set_index('index',inplace=True)
                idx = np.asarray(df_final.index)
                idx = idx[:top_kernels]

                IDX.extend(idx)
                DFs.append(df_final)

            self.conditioned_motifs = motif_inputs[:,:,:,IDX]
            self.condition_kernels = True
            self.enriched_motifs = DFs
            self.num_conditioned_kernels = len(IDX)
            self.conditioned_kernel = kernel
            print('Kernels Conditioned')

    def Train_WF(self,batch_size = 25, epochs_min = 10,stop_criterion=0.001,kernel=5,units=12,
                 weight_by_class=False,trainable_embedding = True,accuracy_min = None, weight_by_freq=True,plot_loss=False,
                 num_fc_layers=0, units_fc=12, drop_out_rate=0.0,suppress_output=False):


        """
        Train Whole-Sample Classifier

        This method trains the network and saves features values at the
        end of training for motif analysis.

        Inputs
        ---------------------------------------
        batch_size: int
            Size of batch to be used for each training iteration of the net.

        epochs_min: int
            Minimum number of epochs for training neural network.

        stop_criterion: float
            Minimum percent decrease in determined interval (below) to continue
            training. Used as early stopping criterion.

        kernel: int
            Size of convolutional kernel.

        units: int
            Number of filters to be used for convolutional kernel.

        weight_by_class: bool
            Option to weight loss by the inverse of the class frequency. Useful for
            unbalanced classes.

        trainable_embedding; bool
            Toggle to control whether a trainable embedding layer is used or native
            one-hot representation for convolutional layers.

        accuracy_min: float
            Optional parameter to allow alternative training strategy until minimum
            training accuracy is achieved, at which point, training ceases.


        weight_by_freq: bool
            Whether to use frequency to weight each sequence's features.

        plot_loss: bool
            To live plot the train/valid/test losses, set to True.

        num_fc_layers: int
            Number of fully connected layers following convolutional layer.

        units_fc: int
            Number of nodes per fully-connected layers following convolutional layer.

        drop_out_rate: float
            drop out rate for fully connected layers

        suppress_output: bool
            To suppress command line output with training statisitcs, set to True.



        Returns
        ---------------------------------------

        """

        epochs = 100000
        graph_model = tf.Graph()
        with tf.device(self.device):
            with graph_model.as_default():

                # Create All Placeholders
                if self.use_alpha is True:
                    X_Seq_alpha = tf.placeholder(tf.int64,shape=[None, None, self.X_Seq_alpha.shape[2]],name='Input_Alpha')
                    X_Seq_alpha_OH = tf.one_hot(X_Seq_alpha, depth=21)

                if self.use_beta is True:
                    X_Seq_beta = tf.placeholder(tf.int64,shape=[None, None, self.X_Seq_beta.shape[2]],name='Input_Beta')
                    X_Seq_beta_OH = tf.one_hot(X_Seq_beta, depth=21)

                X_Freq = tf.placeholder(tf.float32, shape=[None, None], name='Input_Freq')
                Y = tf.placeholder(tf.int64, shape=[None, self.Y.shape[1]], name='Y')
                prob = tf.placeholder_with_default(0.0, shape=(), name='prob')
                training = tf.placeholder_with_default(False, shape=())


                if trainable_embedding is True:
                    # AA Embedding
                    with tf.variable_scope('AA_Embedding'):
                        embedding_dim_aa = 64
                        embedding_layer_seq = tf.get_variable(name='Embedding_Layer_Seq', shape=[21, embedding_dim_aa],trainable=True)
                        embedding_layer_seq = tf.expand_dims(tf.expand_dims(embedding_layer_seq, axis=0), axis=0)
                        if self.use_alpha is True:
                            inputs_seq_embed_alpha = tf.squeeze(tf.tensordot(X_Seq_alpha_OH, embedding_layer_seq, axes=(3, 2)), axis=(3, 4))
                        if self.use_beta is True:
                            inputs_seq_embed_beta = tf.squeeze(tf.tensordot(X_Seq_beta_OH, embedding_layer_seq, axes=(3, 2)), axis=(3, 4))

                else:
                    if self.use_alpha is True:
                        inputs_seq_embed_alpha = X_Seq_alpha_OH

                    if self.use_beta is True:
                        inputs_seq_embed_beta = X_Seq_beta_OH


                if self.condition_kernels is True:
                    conv_weights = self.conditioned_motifs
                    units = np.abs(units - self.num_conditioned_kernels)
                    kernel = self.conditioned_kernel
                else:
                    conv_weights = None

                if self.use_alpha is True:
                    Seq_Features_alpha, conv_kernel_alpha, Indices_alpha = Convolutional_Features_WF(inputs_seq_embed_alpha,
                        units=units,kernel=kernel,conv_weights=conv_weights,trainable_embedding=trainable_embedding,name='alpha_conv')

                if self.use_beta is True:
                    Seq_Features_beta, conv_kernel_beta, Indices_beta = Convolutional_Features_WF(inputs_seq_embed_beta,
                        units=units,kernel=kernel,conv_weights=conv_weights,trainable_embedding=trainable_embedding,name='beta_conv')

                if (self.use_alpha is True) and (self.use_beta is True):
                    Seq_Features = tf.concat((Seq_Features_alpha,Seq_Features_beta),axis=2)
                elif (self.use_alpha is True) and (self.use_beta is False):
                    Seq_Features = Seq_Features_alpha
                elif (self.use_alpha is False) and (self.use_beta is True):
                    Seq_Features = Seq_Features_beta

                if self.use_alpha is True:
                    if weight_by_freq is True:
                        Seq_Features_Weight_conv = tf.expand_dims(X_Freq, 2) * Seq_Features_alpha
                        Seq_Features_Weight_Sum_conv = tf.reduce_sum(Seq_Features_Weight_conv, axis=1)
                        fc_avg_alpha = Seq_Features_Weight_Sum_conv
                    else:
                        fc_avg_alpha = tf.reduce_sum(Seq_Features_alpha,axis=1)

                if self.use_beta is True:
                    if weight_by_freq is True:
                        Seq_Features_Weight_conv = tf.expand_dims(X_Freq, 2) * Seq_Features_beta
                        Seq_Features_Weight_Sum_conv = tf.reduce_sum(Seq_Features_Weight_conv, axis=1)
                        fc_avg_beta = Seq_Features_Weight_Sum_conv
                    else:
                        fc_avg_beta = tf.reduce_sum(Seq_Features_beta,axis=1)

                conv_variables = tf.trainable_variables()

                fc = Seq_Features

                if num_fc_layers != 0:
                    for lyr in range(num_fc_layers):
                        fc = tf.layers.dropout(fc,prob)
                        fc = tf.layers.dense(fc,units_fc,tf.nn.relu)


                if weight_by_freq is True:
                    Seq_Features_Weight = tf.expand_dims(X_Freq, 2) * fc
                    Seq_Features_Weight_Sum = tf.reduce_sum(Seq_Features_Weight, axis=1)
                    logits = tf.layers.dense(Seq_Features_Weight_Sum,self.Y.shape[1])

                else:
                    Seq_Features_Sum = tf.reduce_sum(fc,axis=1)
                    logits = tf.layers.dense(Seq_Features_Sum,self.Y.shape[1])

                fc_variables = list(set(tf.trainable_variables()) - set(conv_variables))


                if weight_by_class is True:
                    class_weights = tf.constant([(1 / (np.sum(self.Y, 0) / np.sum(self.Y))).tolist()])
                    weights = tf.squeeze(tf.matmul(tf.cast(Y, dtype='float32'), class_weights, transpose_b=True),axis=1)
                    loss = tf.reduce_mean(weights * tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
                else:
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))


                opt_conv = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss, var_list=conv_variables)
                opt_fc = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss, var_list=fc_variables)
                opt = tf.group(opt_conv, opt_fc)


                # Operations for validation/test accuracy
                predicted = tf.nn.softmax(logits, name='predicted')
                correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

                saver = tf.train.Saver()


        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if plot_loss is True:
            fig_loss, ax_loss = plt.subplots()
            plt.show(block=False)
            epoch_list = list(range(epochs))

        with tf.Session(graph=graph_model,config=config) as sess:
            sess.run(tf.global_variables_initializer())

            val_loss_total = []
            train_accuracy_total = []
            train_loss_total = []
            test_loss_total = []

            for e in range(epochs):
                #Train
                train_loss = []
                train_accuracy = []
                for x_seq_a,x_seq_b, x_freq,y in get_batches_model_2(self.train[0],self.train[1],self.train[2],self.train[-1], batch_size=batch_size, random=True):
                    feed_dict = {X_Freq:x_freq,Y:y,prob:drop_out_rate,training:True}
                    if self.use_alpha is True:
                        feed_dict[X_Seq_alpha] = x_seq_a
                    if self.use_beta is True:
                        feed_dict[X_Seq_beta] = x_seq_b

                    loss_i, accuracy_i, _ =  sess.run([loss, accuracy, opt], feed_dict=feed_dict)
                    train_loss.append(loss_i)
                    train_accuracy.append(accuracy_i)

                train_loss = np.mean(train_loss)
                train_accuracy = np.mean(train_accuracy)
                train_accuracy_total.append(train_accuracy)
                train_loss_total.append(train_loss)


                train_predicted_list = []
                for x_seq_a,x_seq_b, x_freq, y in get_batches_model_2(self.train[0], self.train[1],self.train[2], self.train[-1],batch_size=batch_size, random=False):
                    feed_dict = {X_Freq:x_freq,Y:y,prob:drop_out_rate,training:False}
                    if self.use_alpha is True:
                        feed_dict[X_Seq_alpha] = x_seq_a
                    if self.use_beta is True:
                        feed_dict[X_Seq_beta] = x_seq_b
                    predicted_i = sess.run(predicted,feed_dict=feed_dict)
                    train_predicted_list.append(predicted_i)
                y_pred = np.vstack(train_predicted_list)
                y_test = self.train[-1]

                y_test2 = np.vstack(y_test)
                if (np.sum(y_test2[:,0])!=len(y_test2)) and (np.sum(y_test2[:,0])!=0):
                    roc_score_train = roc_auc_score(np.vstack(y_test),np.vstack(y_pred))
                else:
                    roc_score_train = 0.0


                #Validation
                val_loss = []
                val_accuracy = []
                for x_seq_a,x_seq_b,x_freq, y in get_batches_model_2(self.valid[0],self.valid[1],self.valid[2], self.valid[-1], batch_size=batch_size, random=False):
                    feed_dict = {X_Freq:x_freq,Y:y,training:False}
                    if self.use_alpha is True:
                        feed_dict[X_Seq_alpha] = x_seq_a
                    if self.use_beta is True:
                        feed_dict[X_Seq_beta] = x_seq_b

                    loss_i, accuracy_i = sess.run([loss, accuracy], feed_dict=feed_dict)
                    val_loss.append(loss_i)
                    val_accuracy.append(accuracy_i)

                val_loss = np.mean(val_loss)
                val_accuracy = np.mean(val_accuracy)
                val_loss_total.append(val_loss)


                test_loss = []
                test_accuracy = []
                predicted_list = []
                for x_seq_a,x_seq_b,x_freq, y in get_batches_model_2(self.test[0], self.test[1],self.test[2],self.test[-1], batch_size=batch_size, random=False):
                    feed_dict = {X_Freq:x_freq,Y:y,training:False}
                    if self.use_alpha is True:
                        feed_dict[X_Seq_alpha] = x_seq_a
                    if self.use_beta is True:
                        feed_dict[X_Seq_beta] = x_seq_b
                    loss_i, accuracy_i, predicted_i = sess.run([loss, accuracy, predicted], feed_dict=feed_dict)
                    test_loss.append(loss_i)
                    test_accuracy.append(accuracy_i)
                    predicted_list.append(predicted_i)


                test_loss = np.mean(test_loss)
                test_loss_total.append(test_loss)
                test_accuracy = np.mean(test_accuracy)
                predicted_out = np.vstack(predicted_list)
                self.y_pred = predicted_out
                self.y_test = self.test[-1]

                y_test2 = np.vstack(self.y_test)
                if (np.sum(y_test2[:,0])!=len(y_test2)) and (np.sum(y_test2[:,0])!=0):
                    roc_score = roc_auc_score(np.vstack(self.y_test),np.vstack(self.y_pred))
                else:
                    roc_score = 0.0

                if plot_loss is True:
                    ax_loss.clear()
                    ax_loss.plot(epoch_list[0:e + 1], train_loss_total, label='Train_Loss')
                    ax_loss.plot(epoch_list[0:e + 1], val_loss_total, label='Valid_Loss')
                    ax_loss.plot(epoch_list[0:e + 1], test_loss_total, label='Test_Loss')
                    ax_loss.legend()
                    plt.pause(0.05)

                if suppress_output is False:
                    print("Training_Statistics: \n",
                          "Epoch: {}/{}".format(e + 1, epochs),
                          "Training loss: {:.5f}".format(train_loss),
                          "Validation loss: {:.5f}".format(val_loss),
                          "Testing loss: {:.5f}".format(test_loss),
                          "Training Accuracy: {:.5}".format(train_accuracy),
                          "Validation Accuracy: {:.5}".format(val_accuracy),
                          "Testing Accuracy: {:.5}".format(test_accuracy),
                          'Training AUC: {:.5}'.format(roc_score_train),
                          'Testing AUC: {:.5}'.format(roc_score))


                if e > epochs_min:
                    if accuracy_min is not None:
                        if np.mean(train_accuracy_total[-3:]) >= accuracy_min:
                            break

                    else:
                        if self.LOO is None:
                            a, b, c = -10, -7, -3
                            if val_loss_total:
                                if (np.mean(val_loss_total[a:b]) - np.mean(val_loss_total[c:])) / np.mean(val_loss_total[a:b]) < stop_criterion:
                                    break

                        else:
                            a, b, c = -10, -7, -3
                            if train_loss_total:
                                if (np.mean(train_loss_total[a:b]) - np.mean(train_loss_total[c:])) / np.mean(train_loss_total[a:b]) < stop_criterion:
                                    break

                            if np.mean(train_accuracy_total[-100:]) == 1.0:
                                break


            #saver.save(sess, self.Name + '_WF/' + self.Name + '_WF.ckpt')
            alpha_features_WF_list = []
            alpha_features_seq_list = []
            alpha_indices_list = []
            beta_features_WF_list = []
            beta_features_seq_list = []
            beta_indices_list = []
            for x_seq_a,x_seq_b, x_freq, y in get_batches_model_2(self.X_Seq_alpha,self.X_Seq_beta,self.X_Freq, self.Y, batch_size=batch_size, random=False):
                feed_dict = {X_Freq:x_freq}
                if self.use_alpha is True:
                    feed_dict[X_Seq_alpha] = x_seq_a
                if self.use_beta is True:
                    feed_dict[X_Seq_beta] = x_seq_b

                if self.use_alpha is True:
                    features_wf_i, features_i, indices_i = sess.run([fc_avg_alpha, Seq_Features_alpha, Indices_alpha],feed_dict=feed_dict)
                    alpha_features_WF_list.append(features_wf_i)
                    alpha_features_seq_list.append(features_i)
                    alpha_indices_list.append(indices_i)

                if self.use_beta is True:
                    features_wf_i, features_i, indices_i = sess.run([fc_avg_beta, Seq_Features_beta, Indices_beta],feed_dict=feed_dict)
                    beta_features_WF_list.append(features_wf_i)
                    beta_features_seq_list.append(features_i)
                    beta_indices_list.append(indices_i)


            predicted_seq_list = []
            for x_seq_a, x_seq_b, x_freq, y in get_batches_model_2(self.test[0], self.test[1], self.test[2],self.test[-1], batch_size=batch_size, random=False):
                x_freq = np.expand_dims(np.reshape(x_freq,[x_freq.shape[0]*x_freq.shape[1]]),-1)
                feed_dict = {X_Freq: x_freq, Y: y, training: False}
                if self.use_alpha is True:
                    feed_dict[X_Seq_alpha] = np.expand_dims(np.reshape(x_seq_a,[x_seq_a.shape[0]*x_seq_a.shape[1],x_seq_a.shape[-1]]),1)
                if self.use_beta is True:
                    feed_dict[X_Seq_beta] = np.expand_dims(np.reshape(x_seq_b,[x_seq_b.shape[0]*x_seq_b.shape[1],x_seq_b.shape[-1]]),1)

                predicted_i = sess.run(predicted, feed_dict=feed_dict)
                predicted_seq_list.append(predicted_i)

            predicted_seq_list = np.vstack(predicted_seq_list)
            idx = np.reshape(self.test[6],[self.test[6].shape[0]*self.test[6].shape[1]])
            self.predicted[idx] = predicted_seq_list
            self.predicted_seq = predicted_seq_list


            if self.use_alpha is True:
                self.alpha_features_wf = np.vstack(alpha_features_WF_list)
                self.alpha_features_seq = np.vstack(alpha_features_seq_list)
                self.alpha_indices = np.vstack(alpha_indices_list)
                self.kernel = kernel

                with open(os.path.join(self.Name, self.Name) + '_alpha_features_WF.pkl', 'wb') as f:
                    pickle.dump([self.alpha_features_wf, self.alpha_features_seq, self.alpha_indices, self.alpha_sequences,
                                 self.X_Freq, self.y_pred, self.y_test, self.labels, self.Y, self.files, self.kernel],f, protocol=4)

            if self.use_beta is True:
                self.beta_features_wf = np.vstack(beta_features_WF_list)
                self.beta_features_seq = np.vstack(beta_features_seq_list)
                self.beta_indices = np.vstack(beta_indices_list)
                self.kernel = kernel

                with open(os.path.join(self.Name, self.Name) + '_beta_features_WF.pkl', 'wb') as f:
                    pickle.dump([self.beta_features_wf, self.beta_features_seq, self.beta_indices, self.beta_sequences,
                                 self.X_Freq, self.y_pred, self.y_test, self.labels, self.Y, self.files, self.kernel],f, protocol=4)


            if plot_loss is True:
                plt.close()

            print('Done Training')

    def Representative_Sequences_WF(self,top_seq=10):
        """
        Identify most highly predicted sequences for each class for whole sample classifier.

        This method allows the user to query which sequences were most predicted to belong to a given class.

        Inputs
        ---------------------------------------

        top_seq: int
            The number of top sequences to show for each class.

        Returns

        self.Rep_Seq_WF: dictionary of dataframes
            This dictionary of dataframes holds for each class the top sequences and their respective
            probabiltiies for all classes. These dataframes can also be found in the results folder under Rep_Sequences_WF.

        ---------------------------------------


        """
        dir = 'Rep_Sequences_WF'
        dir = os.path.join(self.directory_results,dir)
        if not os.path.exists(dir):
            os.makedirs(dir)

        file_list = [f for f in os.listdir(dir)]
        [os.remove(os.path.join(dir, f)) for f in file_list]

        Rep_Seq = []
        keep = []
        df_temp = pd.DataFrame()
        df_temp['alpha'] = np.reshape(self.alpha_sequences,[self.alpha_sequences.shape[0]*self.alpha_sequences.shape[1]])
        df_temp['beta'] = np.reshape(self.beta_sequences,[self.beta_sequences.shape[0]*self.beta_sequences.shape[1]])
        df_temp['freq'] = np.reshape(self.X_Freq,[self.X_Freq.shape[0]*self.X_Freq.shape[1]])
        df_temp['label'] = np.reshape(self.label_id,[self.label_id.shape[0]*self.label_id.shape[1]])
        df_temp['file'] = np.reshape(self.files_id,[self.files_id.shape[0]*self.files_id.shape[1]])
        for ii,sample in enumerate(self.lb.classes_,0):
            df_temp[sample] = self.predicted[:,ii]

        if self.use_beta is True:
            df_temp = df_temp[df_temp['beta'] != 'null']

        if self.use_alpha is True:
            df_temp = df_temp[df_temp['alpha'] != 'null']

        for ii,sample in enumerate(self.lb.classes_,0):
            df_temp.sort_values(by=sample,ascending=False,inplace=True)
            df_sample = df_temp[df_temp['label']==sample][0:top_seq]

            if not df_sample.empty:
                Rep_Seq.append(df_sample)
                df_sample.to_csv(os.path.join(dir,sample+'.csv'),index=False)
                keep.append(ii)

        self.Rep_Seq_WF = dict(zip(self.lb.classes_[keep],Rep_Seq))

    def Motif_Identification_WF(self,group,p_val_threshold=0.05,cut=95,save_images=False):
        """
        Motif Identification for Whole Sample Classifier

        This method looks for enriched features in the predetermined group
        and returns fasta files in directory to be used with "https://weblogo.berkeley.edu/logo.cgi"
        to produce seqlogos.

        Inputs
        ---------------------------------------
        group: string
            Class for analyzing enriched motifs.

        p_val_threshold: float
            Significance threshold for enriched features/motifs for
            Mann-Whitney UTest.

        cut: float
            Percentile to set threshold for what is considered as a sequence
            positive for a feature.

        save_images: bool
            In order to save violin plots of feature contribution to each cohort,
            set to True.

        Returns
        ---------------------------------------

        self.(alpha/beta)_group_features_wf: Pandas Dataframe
            Sequences used to determine motifs in fasta files
            are stored in this dataframe where column names represent
            the feature number.

        """

        #Get Features

        if self.use_alpha is True:
            with open(os.path.join(self.Name, self.Name) + '_alpha_features_WF.pkl', 'rb') as f:
                self.alpha_features_wf, self.alpha_features_seq, self.alpha_indices, self.alpha_sequences, \
                self.X_Freq, self.y_pred, self.y_test, self.labels, self.Y, self.files, self.kernel = pickle.load(f)

        if self.use_beta is True:
            with open(os.path.join(self.Name, self.Name) + '_beta_features_WF.pkl', 'rb') as f:
                self.beta_features_wf, self.beta_features_seq, self.beta_indices, self.beta_sequences, \
                self.X_Freq, self.y_pred, self.y_test, self.labels, self.Y, self.files, self.kernel = pickle.load(f)


        group_num = np.where(self.lb.classes_ == group)[0][0]

        #Find diff expressed features
        idx_pos = self.Y[:, group_num] == 1
        idx_neg = self.Y[:, group_num] == 0

        if self.use_alpha is True:
            self.alpha_group_features_wf = Diff_Features_WF(self.alpha_features_wf,self.alpha_features_seq,self.alpha_indices,self.alpha_sequences,
                                                            self.X_Freq,self.labels,self.kernel,idx_pos,idx_neg,self.directory_results,p_val_threshold,
                                                            cut,'alpha',group,save_images,self.lb)

        if self.use_beta is True:
            self.beta_group_features_wf = Diff_Features_WF(self.beta_features_wf,self.beta_features_seq,self.beta_indices,self.beta_sequences,
                                                            self.X_Freq,self.labels,self.kernel,idx_pos,idx_neg,self.directory_results,p_val_threshold,
                                                            cut,'beta',group,save_images,self.lb)

        print('Motif Identification Completed')


    def Monte_Carlo_CrossVal_WF(self, fold=5, test_size=0.25, epochs_min=5, batch_size=25, LOO=None,stop_criterion=0.001,
                             kernel=5,units=12,weight_by_class=False, trainable_embedding=True,accuracy_min = None, weight_by_freq = True,plot_loss=False,
                             num_fc_layers=0, units_fc=12, drop_out_rate=0.0,suppress_output=False):


        """
        Monte Carlo Cross-Validation for Whole Sample Classifier

        If the number of samples is small but training the whole sample classifier, one
        can use Monte Carlo Cross Validation to train a number of iterations before assessing
        predictive performance.After this method is run, the AUC_Curve method can be run to
        assess the overall performance.

        Inputs
        ---------------------------------------
        fold: int
            Number of iterations for Cross-Validation

        test_size: float
            Fraction of sample to be used for valid and test set.

        LOO: int
            Number of samples to leave-out in Leave-One-Out Cross-Validation

        batch_size: int
            Size of batch to be used for each training iteration of the net.

        epochs_min: int
            Minimum number of epochs for training neural network.

        stop_criterion: float
            Minimum percent decrease in determined interval (below) to continue
            training. Used as early stopping criterion.

        kernel: int
            Size of convolutional kernel.

        units: int
            Number of filters to be used for convolutional kernel.


        weight_by_class: bool
            Option to weight loss by the inverse of the class frequency. Useful for
            unbalanced classes.

        trainable_embedding; bool
            Toggle to control whether a trainable embedding layer is used or native
            one-hot representation for convolutional layers.

        accuracy_min: float
            Optional parameter to allow alternative training strategy until minimum
            training accuracy is achieved, at which point, training ceases.


        weight_by_freq: bool
            Whether to use frequency to weight each sequence's features.

        plot_loss: bool
            To live plot the train/valid/test losses, set to True.

        num_fc_layers: int
            Number of fully connected layers following convolutional layer.

        units_fc: int
            Number of nodes per fully-connected layers following convolutional layer.

        drop_out_rate: float
            drop out rate for fully connected layers

        suppress_output: bool
            To suppress command line output with training statisitcs, set to True.


        Returns
        ---------------------------------------

        """

        y_pred = []
        y_test = []
        for i in range(0, fold):
            if suppress_output is False:
                print(i)
            self.Get_Train_Valid_Test_WF(test_size=test_size, LOO=LOO)
            self.Train_WF(epochs_min=epochs_min, batch_size=batch_size,stop_criterion=stop_criterion,
                          kernel=kernel,units=units,weight_by_class=weight_by_class,
                          trainable_embedding=trainable_embedding,accuracy_min=accuracy_min,
                          weight_by_freq = weight_by_freq,plot_loss=plot_loss,num_fc_layers=num_fc_layers,
                          units_fc=units_fc,drop_out_rate=drop_out_rate,suppress_output=suppress_output)

            y_test.append(self.y_test)
            y_pred.append(self.y_pred)

            if i == 0:
                predicted = np.zeros_like(self.predicted)
                counts = np.zeros_like(self.predicted)

            idx = np.reshape(self.test[6],[self.test[6].shape[0]*self.test[6].shape[1]])
            predicted[idx] += self.predicted_seq
            counts[idx] += 1

            y_test2 = np.vstack(y_test)
            y_pred2 = np.vstack(y_pred)

            if suppress_output is False:
                print("Accuracy = {}".format(np.average(np.equal(np.argmax(y_pred2,1),np.argmax(y_test2,1)))))

                if self.y_test.shape[1] == 2:
                    if i > 0:
                        y_test2 = np.vstack(y_test)
                        if (np.sum(y_test2[:, 0]) != len(y_test2)) and (np.sum(y_test2[:, 0]) != 0):
                            print("AUC = {}".format(roc_auc_score(np.vstack(y_test), np.vstack(y_pred))))


        self.y_test = np.vstack(y_test)
        self.y_pred = np.vstack(y_pred)
        self.predicted = np.divide(predicted,counts, out = np.zeros_like(predicted), where = counts != 0)
        print('Monte Carlo Simulation Completed')

    def K_Fold_CrossVal_WF(self,folds=None,epochs_min=5,batch_size=25,stop_criterion=0.001, kernel=5,units=12, weight_by_class=False, iterations=None,
                        trainable_embedding=True, accuracy_min = None, weight_by_freq = True, plot_loss=False,
                        num_fc_layers=0, units_fc=12, drop_out_rate=0.0,suppress_output=False):

        """
        K_Fold Cross-Validation for Whole Sample Classifier

        If the number of samples is small but training the whole sample classifier, one
        can use K_Fold Cross Validation to train on all but one before assessing
        predictive performance.After this method is run, the AUC_Curve method can be run to
        assess the overall performance.

        Inputs
        ---------------------------------------
        folds: int
            Number of Folds

        batch_size: int
            Size of batch to be used for each training iteration of the net.

        epochs_min: int
            Minimum number of epochs for training neural network.

        stop_criterion: float
            Minimum percent decrease in determined interval (below) to continue
            training. Used as early stopping criterion.

        kernel: int
            Size of convolutional kernel.

        units: int
            Number of filters to be used for convolutional kernel.

        weight_by_class: bool
            Option to weight loss by the inverse of the class frequency. Useful for
            unbalanced classes.

        iterations: int
            Option to specify how many iterations one wants to complete before
            terminating training. Useful for very large datasets.

        trainable_embedding; bool
            Toggle to control whether a trainable embedding layer is used or native
            one-hot representation for convolutional layers.

        accuracy_min: float
            Optional parameter to allow alternative training strategy until minimum
            training accuracy is achieved, at which point, training ceases.


        weight_by_freq: bool
            Whether to use frequency to weight each sequence's features.

        plot_loss: bool
            To live plot the train/valid/test losses, set to True.

        num_fc_layers: int
            Number of fully connected layers following convolutional layer.

        units_fc: int
            Number of nodes per fully-connected layers following convolutional layer.

        drop_out_rate: float
            drop out rate for fully connected layers

        suppress_output: bool
            To suppress command line output with training statisitcs, set to True.


        Returns
        ---------------------------------------

        """

        if folds is None:
            folds = len(self.files)

        #Create Folds
        idx = list(range(len(self.files)))
        idx_left = idx
        file_per_sample = len(self.files) // folds
        test_idx = []
        for ii in range(folds):
            if ii != folds-1:
                idx_sel = np.random.choice(idx_left, size=file_per_sample, replace=False)
            else:
                idx_sel = idx_left

            test_idx.append(idx_sel)
            idx_left = np.setdiff1d(idx_left, idx_sel)


        y_test = []
        y_pred = []
        for ii in range(folds):
            if suppress_output is False:
                print(ii)
            train_idx = np.setdiff1d(idx,test_idx[ii])

            Vars = [self.X_Seq_alpha,self.X_Seq_beta, self.X_Freq,self.files,self.alpha_sequences,self.beta_sequences,self.seq_index]
            self.train, self.test = Get_Train_Test(Vars=Vars,train_idx=train_idx,test_idx = test_idx[ii],Y=self.Y)
            self.valid = self.test
            self.LOO = True

            self.Train_WF(epochs_min=epochs_min, batch_size=batch_size,
                          stop_criterion=stop_criterion, kernel=kernel,
                          units=units, weight_by_class=weight_by_class,
                          trainable_embedding=trainable_embedding,accuracy_min = accuracy_min,
                          weight_by_freq = weight_by_freq, plot_loss = plot_loss,num_fc_layers=num_fc_layers,units_fc=units_fc,
                          drop_out_rate=drop_out_rate,suppress_output=suppress_output)


            y_test.append(self.y_test)
            y_pred.append(self.y_pred)

            y_test2 = np.vstack(y_test)
            y_pred2 = np.vstack(y_pred)

            if suppress_output is False:
                print("Accuracy = {}".format(np.average(np.equal(np.argmax(y_pred2, 1), np.argmax(y_test2, 1)))))

                if self.y_test.shape[1] == 2:
                    if ii > 0:
                        if (np.sum(y_test2[:, 0]) != len(y_test2)) and (np.sum(y_test2[:, 0]) != 0):
                            print("AUC = {}".format(roc_auc_score(np.vstack(y_test), np.vstack(y_pred))))


            if iterations is not None:
                if ii+1 >= iterations:
                    break

        self.y_test = np.vstack(y_test)
        self.y_pred = np.vstack(y_pred)
        print('K-fold Cross Validation Completed')













































