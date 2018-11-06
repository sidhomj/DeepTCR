import os
import sys
sys.path.append('../')
from DeepTCR.supervised_functions.utils import *
from DeepTCR.supervised_functions.Layers import *
from DeepTCR.supervised_functions.data_processing import *
import glob
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import mannwhitneyu
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import seaborn as sns
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

        Returns
        ---------------------------------------


        """
        #Assign parameters
        self.Name = Name
        self.max_length = max_length
        self.device = device
        self.condition_kernels = False

        #Create dataframes for assigning AA to ints
        aa_idx, aa_mat = make_aa_df()
        aa_idx_inv = {v: k for k, v in aa_idx.items()}
        self.aa_idx = aa_idx
        self.aa_idx_inv = aa_idx_inv

        #Create directory for results of analysis
        directory = self.Name + '_Results/'
        self.directory_results = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        #Create directory for any temporary files
        directory = self.Name
        if not os.path.exists(directory):
            os.makedirs(directory)

    def Get_Data_SS(self, directory, Load_Prev_Data=False,classes=None,save_data=True,type_of_data_cut='Fraction_Response',data_cut=1.0,n_jobs=40,
                    aa_column = None, count_column = None,sep='\t'):

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

        aa_column: int
            Columns where amino acid data is stored. If set to None, column with a header containing 'acid' is used as
            the amino acid column.

        count_column: int
            Column where counts are stored. If set to None, first column with data in integer datatype is used as the
            counts column.

        sep: str
            Type of delimiter used in file with TCRSeq data.

        Returns
        ---------------------------------------

        """

        if Load_Prev_Data is False:
            if classes is None:
                # Get names of classes from folders names
                classes = [d for d in os.listdir(directory) if os.path.isdir(directory + d)]
                classes = [f for f in classes if not f.startswith('.')]


            self.lb = LabelEncoder()
            self.lb.fit(classes)
            self.classes = self.lb.classes_


            #Load sequences, labels, and files
            sequences = []
            label_id = []
            file_id = []
            p = Pool(n_jobs)

            if sep == '\t':
                ext = '/*.tsv'
            elif sep == ',':
                ext = '/*.csv'
            else:
                print('Not Valid Delimiter')
                return

            for type in self.classes:
                files_read = glob.glob(directory + type + ext)
                num_ins = len(files_read)
                args = list(zip(files_read,
                                [type_of_data_cut] * num_ins,
                                [data_cut] * num_ins,
                                [aa_column] * num_ins,
                                [count_column] * num_ins,
                                [sep] * num_ins,
                                [self.max_length]*num_ins))

                DF = p.starmap(Get_DF_Data, args)

                for df, file in zip(DF, files_read):
                    sequences += df['aminoAcid'].tolist()
                    label_id += [type] * len(df)
                    file_id += [file.split('/')[-1]] * len(df)

            sequences = np.asarray(sequences)
            label_id = np.asarray(label_id)
            file_id = np.asarray(file_id)

            Y = self.lb.transform(label_id)
            OH = OneHotEncoder(sparse=False)
            Y = OH.fit_transform(Y.reshape(-1,1))

            #Embed sequences into ints
            args = list(zip(sequences, [self.aa_idx] * len(sequences), [self.max_length] * len(sequences)))
            print('Embedding Sequences')
            result = p.starmap(Embed_Seq_Num, args)
            p.close()
            sequences_num = np.vstack(result)
            X_Seq = np.expand_dims(sequences_num, 1)

            #Save Data
            if save_data is True:
                with open(self.Name + '/' + self.Name + '_Data.pkl', 'wb') as f:
                    pickle.dump([X_Seq, Y, sequences, label_id, file_id, self.lb], f, protocol=4)

        else:
            #Load Data
            with open(self.Name+'/'+self.Name + '_Data.pkl', 'rb') as f:
                X_Seq,Y, sequences, label_id, file_id, self.lb = pickle.load(f)

        self.X_Seq = X_Seq
        self.Y = Y
        self.sequences = sequences
        self.label_id = label_id
        self.file_id = file_id

    def Get_Train_Valid_Test_SS(self,test_size=0.2):
        """
        Train/Valid/Test Splits.

        Divide data for train, valid, test set. Training is used to
        train model parameters, validation is used to set early stopping,
        and test acts as blackbox independent test set.

        Inputs
        ---------------------------------------
        test_size: float
            Fraction of sample to be used for valid and test set.

        Returns
        ---------------------------------------

        """
        Vars = [self.X_Seq,self.sequences,self.file_id]
        self.train,self.valid,self.test = Get_Train_Valid_Test(Vars=Vars,Y=self.Y,test_size=test_size,regression=False)

    def Train_SS(self,batch_size = 1000, epochs_min = 10,stop_criterion=0.001,kernel=5,units=12,trainable_embedding=True,weight_by_class=False,
                 num_fc_layers=0,units_fc=12,drop_out_rate=0.0):
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


        Returns
        ---------------------------------------

        """
        epochs = 10000
        graph_model = tf.Graph()
        with tf.device(self.device):
            with graph_model.as_default():
                X_Seq = tf.placeholder(tf.int64, [None, self.X_Seq.shape[1], self.X_Seq.shape[2]], name='Input_Seq')
                X_Seq_OH = tf.one_hot(X_Seq, depth=21)
                Y = tf.placeholder(tf.float64, shape=[None, self.Y.shape[1]])
                training = tf.placeholder_with_default(False, shape=())
                prob = tf.placeholder_with_default(0.0, shape=(), name='prob')

                if trainable_embedding is True:
                    # AA Embedding
                    with tf.variable_scope('AA_Embedding'):
                        embedding_dim_aa = 64
                        embedding_layer_seq = tf.get_variable(name='Embedding_Layer_Seq', shape=[21, embedding_dim_aa])
                        embedding_layer_seq = tf.expand_dims(tf.expand_dims(embedding_layer_seq, axis=0), axis=0)
                        inputs_seq_embed = tf.squeeze(tf.tensordot(X_Seq_OH, embedding_layer_seq, axes=(3, 2)),axis=(3, 4))

                else:
                    inputs_seq_embed = X_Seq_OH


                # Convolutional Features
                Seq_Features, Indices = Convolutional_Features(inputs_seq_embed,kernel=kernel,units=units,trainable_embedding=trainable_embedding)
                Seq_Features = tf.identity(Seq_Features,'Seq_Features')
                fc = Seq_Features

                if num_fc_layers != 0:
                    for lyr in range(num_fc_layers):
                        fc = tf.layers.dense(fc,units_fc,tf.nn.relu)
                        fc = tf.layers.dropout(fc,prob)


                logits = tf.layers.dense(fc, self.Y.shape[1])


                if weight_by_class is True:
                    class_weights = tf.constant([(1 / (np.sum(self.Y, 0) / np.sum(self.Y))).tolist()])
                    weights = tf.squeeze(tf.matmul(tf.cast(Y, dtype='float32'), class_weights, transpose_b=True), axis=1)
                    loss = tf.reduce_mean(weights*tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
                else:
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))

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
                for x_seq, y in get_batches(self.train[0], self.train[-1], batch_size=batch_size, random=True):
                    feed_dict = {X_Seq: x_seq, Y: y, prob: drop_out_rate}
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
                for x_seq, y in get_batches(self.valid[0], self.valid[-1], batch_size=batch_size, random=False):
                    feed_dict = {X_Seq: x_seq, Y: y}
                    loss_i, accuracy_i = sess.run([loss, accuracy], feed_dict=feed_dict)
                    val_loss.append(loss_i)
                    val_accuracy.append(accuracy_i)


                val_loss = np.mean(val_loss)
                val_accuracy = np.mean(val_accuracy)
                val_loss_total.append(val_loss)

                test_loss = []
                test_accuracy = []
                predicted_list = []
                for x_seq, y in get_batches(self.test[0], self.test[-1], batch_size=batch_size, random=False):
                    feed_dict = {X_Seq: x_seq, Y: y}
                    loss_i, accuracy_i, predicted_i = sess.run([loss, accuracy, predicted], feed_dict=feed_dict)
                    test_loss.append(loss_i)
                    test_accuracy.append(accuracy_i)
                    predicted_list.append(predicted_i)

                test_loss = np.mean(test_loss)
                test_accuracy = np.mean(test_accuracy)
                predicted_out = np.vstack(predicted_list)
                self.y_pred = predicted_out
                self.y_test = self.test[-1]

                auc = roc_auc_score(self.test[-1],predicted_out)

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


            print('Done Training')
            features_list = []
            indices_list = []
            for x_seq, y in get_batches(self.X_Seq, self.Y, batch_size=batch_size, random=False):
                feed_dict = {X_Seq: x_seq}
                features_i,indices_i = sess.run([Seq_Features,Indices],feed_dict=feed_dict)
                features_list.append(features_i)
                indices_list.append(indices_i)

            self.features = np.vstack(features_list)
            self.indices = np.vstack(indices_list)
            self.seq_test = self.sequences
            self.kernel = kernel

            with open(self.Name+'/'+self.Name + '_features.pkl','wb') as f:
                pickle.dump([self.features,self.indices,self.seq_test,self.y_pred,self.y_test,self.kernel],f)

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

        self.group_features_ss: Pandas Dataframe
            Sequences used to determine motifs in fasta files
            are stored in this dataframe where column names represent
            the feature number.

        """
        #Get Saved Features, Indices, and Sequences
        with open(self.Name+'/'+self.Name + '_features.pkl', 'rb') as f:
            self.features, self.indices, self.seq_test, self.y_pred, self.y_test,self.kernel = pickle.load(f)

        group_num = np.where(self.lb.classes_ == group)[0][0]

        # Find diff expressed features
        idx_pos = self.Y[:, group_num] == 1
        idx_neg = self.Y[:, group_num] == 0

        pos_mean = []
        neg_mean = []
        p_val = []
        feature_num = list(range(len(self.features.T)))
        for i in feature_num:
            pos = self.features[idx_pos, i]
            neg = self.features[idx_neg, i]
            pos_mean.append(np.mean(pos))
            neg_mean.append(np.mean(neg))
            try:
                stat, p = mannwhitneyu(pos,neg)
                p_val.append(p)
            except:
                p_val.append(1.0)


        df_features = pd.DataFrame()
        df_features['Feature'] = feature_num
        df_features['P_Val'] = p_val
        df_features['Pos'] = pos_mean
        df_features['Neg'] = neg_mean
        df_features['Mag'] = df_features['Pos'] - df_features['Neg']

        df_features = df_features[df_features['P_Val']<p_val_threshold]

        df_features.sort_values(by='Mag',inplace=True,ascending=False)

        # Get motifs for positive features
        dir = self.directory_results+group + '_SS_Motifs'
        if not os.path.exists(dir):
            os.makedirs(dir)

        file_list = [f for f in os.listdir(dir)]
        [os.remove(os.path.join(dir, f)) for f in file_list]

        top_seq = 10
        seq_cluster = []
        feature_keep = []
        for feature in df_features['Feature'].tolist():
            if df_features['Mag'][feature] > 0:
                feature_keep.append(feature)
                sel = np.flip(self.features[:, feature].argsort(), -1)
                sel = sel[0:top_seq]
                seq_sel = self.seq_test[sel]
                ind_sel = self.indices[sel, feature]
                seq_cluster.append(seq_sel)

                motifs = []
                for ii, i in enumerate(ind_sel, 0):
                    motif = seq_sel[ii][int(i):int(i) + self.kernel]
                    if len(motif) < self.kernel:
                        motif = motif + 'X' * (self.kernel - len(motif))
                    motif = motif.lower()
                    motif = SeqRecord(Seq(motif, IUPAC.protein), str(ii))
                    motifs.append(motif)

                SeqIO.write(motifs, dir + '/feature_' + str(feature) + '.fasta', 'fasta')

        seq_features_df_pos = pd.DataFrame()
        for ii, f in enumerate(feature_keep, 0):
            seq_features_df_pos[f] = seq_cluster[ii]

        self.group_features_ss = seq_features_df_pos

    def AUC_Curve(self,show_all=True,filename=None):
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
        if filename is None:
            plt.savefig(self.directory_results + 'AUC.tif')
        else:
            plt.savefig(self.directory_results + filename+'_AUC.tif')

        plt.show(block=False)

    def Get_Data_WF(self,directory,Load_Prev_Data=False,classes=None,save_data=True,type_of_data_cut='Fraction_Response',data_cut=1.0,n_jobs=40,
                    aa_column = None, count_column = None,sep='\t'):
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

        aa_column: int
            Columns where amino acid data is stored. If set to None, column with a header containing 'acid' is used as
            the amino acid column.

        count_column: int
            Column where counts are stored. If set to None, first column with data in integer datatype is used as the
            counts column.

        sep: str
            Type of delimiter used in file with TCRSeq data.


        Returns
        ---------------------------------------

        """
        if Load_Prev_Data is False:
            if classes is None:
                classes = [d for d in os.listdir(directory) if os.path.isdir(directory + d)]
                classes = [f for f in classes if not f.startswith('.')]

            self.lb = LabelEncoder()
            self.lb.fit(classes)
            self.classes = self.lb.classes_

            sequences = []
            labels = []
            files = []
            len_file = []
            freq = []
            p = Pool(n_jobs)

            if sep == '\t':
                ext = '/*.tsv'
            elif sep == ',':
                ext = '/*.csv'
            else:
                print('Not Valid Delimiter')
                return

            for type in self.classes:
                files_read = glob.glob(directory + type + ext)
                num_ins = len(files_read)
                args = list(zip(files_read,
                                [type_of_data_cut] * num_ins,
                                [data_cut] * num_ins,
                                [aa_column]*num_ins,
                                [count_column]*num_ins,
                                [sep]*num_ins,
                                [self.max_length]*num_ins))

                DF = p.starmap(Get_DF_Data,args)

                for df, file in zip(DF, files_read):
                    sequences.append(df['aminoAcid'].tolist())
                    freq.append(df['Frequency'].tolist())
                    labels.append(type)
                    files.append(file.split('/')[-1])
                    len_file.append(len(df))


            labels = np.asarray(labels)
            files = np.asarray(files)
            len_file = np.asarray(len_file)

            num_seq_per_instance = np.max(len_file)

            for ii, sample in enumerate(sequences):
                if len(sample) > num_seq_per_instance:
                    sequences[ii] = sample[0:num_seq_per_instance]
                    freq[ii] = np.asarray(freq[ii][0:num_seq_per_instance])
                elif len(sample) < num_seq_per_instance:
                    sequences[ii] = sample + ['null'] * (num_seq_per_instance - len(sample))
                    freq[ii] = np.pad(freq[ii], (0, num_seq_per_instance - len(sample)), mode='constant')

            all_seq = np.hstack(sequences).tolist()
            args = list(zip(all_seq, [self.aa_idx] * len(all_seq), [self.max_length] * len(all_seq)))
            print('Embedding Sequences')
            result = p.starmap(Embed_Seq_Num, args)
            p.close()
            X_Seq = np.vstack(result).reshape(len(sequences), -1, self.max_length)
            X_Freq = np.vstack(freq)


            Y = self.lb.transform(labels)
            OH = OneHotEncoder(sparse=False)
            Y = OH.fit_transform(Y.reshape(-1,1))


            if save_data is True:
                with open(self.Name + '/' + self.Name + '_Data_WF.pkl', 'wb') as f:
                    pickle.dump([X_Seq, X_Freq, Y, labels, files, self.lb, sequences], f, protocol=4)

        else:
            with open(self.Name+'/'+self.Name + '_Data_WF.pkl', 'rb') as f:
                X_Seq, X_Freq, Y, labels, files,self.lb,sequences = pickle.load(f)


        self.X_Seq = X_Seq
        self.X_Freq = X_Freq
        self.Y = Y
        self.labels = labels
        self.files = files
        self.sequences = sequences

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

        Vars = [self.X_Seq, self.X_Freq,self.files,np.asarray(self.sequences)]
        self.train, self.valid, self.test = Get_Train_Valid_Test(Vars=Vars, Y=self.Y, test_size=test_size, regression=False,LOO=LOO)
        self.LOO = LOO

    def Condition_Kernels(self,kernel=3,sample_batch_size=50,motif_batch_size=1000,top_kernels=10,weight_by_freq=False,sample=None):
        """
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

    def Train_WF(self,batch_size = 25, epochs_min = 10,stop_criterion=0.001,kernel=5,units=12,
                 weight_by_class=False,trainable_embedding = True,accuracy_min = None, weight_by_freq=True,plot_loss=False,
                 num_fc_layers=0, units_fc=12, drop_out_rate=0.0):


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


        Returns
        ---------------------------------------

        """

        epochs = 100000
        graph_model = tf.Graph()
        with tf.device(self.device):
            with graph_model.as_default():

                # Create All Placeholders
                X_Seq = tf.placeholder(tf.int64, shape=[None, None, self.X_Seq.shape[2]], name='Input_Seq')
                X_Seq_OH = tf.one_hot(X_Seq, depth=21)
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

                        inputs_seq_embed = tf.squeeze(tf.tensordot(X_Seq_OH, embedding_layer_seq, axes=(3, 2)), axis=(3, 4))
                else:
                    inputs_seq_embed = X_Seq_OH


                if self.condition_kernels is True:
                    conv_weights = self.conditioned_motifs
                    units = np.abs(units - self.num_conditioned_kernels)
                    kernel = self.conditioned_kernel
                else:
                    conv_weights = None

                Seq_Features,conv_kernel,Indices = Convolutional_Features_WF(inputs_seq_embed, units=units,kernel=kernel, conv_weights=conv_weights,trainable_embedding=trainable_embedding)
                conv_variables = tf.trainable_variables()




                fc = Seq_Features

                if num_fc_layers != 0:
                    for lyr in range(num_fc_layers):
                        fc = tf.layers.dense(fc,units_fc,tf.nn.relu)
                        fc = tf.layers.dropout(fc,prob)


                if weight_by_freq is True:
                    Seq_Features_Weight_conv = tf.expand_dims(X_Freq, 2) * Seq_Features
                    Seq_Features_Weight_Sum_conv = tf.reduce_sum(Seq_Features_Weight_conv, axis=1)
                    fc_avg = Seq_Features_Weight_Sum_conv

                    Seq_Features_Weight = tf.expand_dims(X_Freq, 2) * fc
                    Seq_Features_Weight_Sum = tf.reduce_sum(Seq_Features_Weight, axis=1)
                    logits = tf.layers.dense(Seq_Features_Weight_Sum,self.Y.shape[1])


                else:
                    fc_avg = tf.reduce_sum(Seq_Features,axis=1)

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
                for x_seq, x_freq,y in get_batches_model(self.train[0],self.train[1],self.train[-1], batch_size=batch_size, random=True):
                    feed_dict = {X_Seq: x_seq, X_Freq: x_freq, Y: y, prob: drop_out_rate,training: True}
                    loss_i, accuracy_i, _ =  sess.run([loss, accuracy, opt], feed_dict=feed_dict)
                    train_loss.append(loss_i)
                    train_accuracy.append(accuracy_i)

                train_loss = np.mean(train_loss)
                train_accuracy = np.mean(train_accuracy)
                train_accuracy_total.append(train_accuracy)
                train_loss_total.append(train_loss)


                train_predicted_list = []
                for x_seq, x_freq, y in get_batches_model(self.train[0], self.train[1], self.train[-1],batch_size=batch_size, random=False):
                    feed_dict = {X_Seq: x_seq, X_Freq: x_freq, Y: y, prob: drop_out_rate, training: False}
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
                for x_seq,x_freq, y in get_batches_model(self.valid[0],self.valid[1], self.valid[-1], batch_size=batch_size, random=False):
                    feed_dict = {X_Seq: x_seq, X_Freq: x_freq, Y: y, training: False}
                    loss_i, accuracy_i = sess.run([loss, accuracy], feed_dict=feed_dict)
                    val_loss.append(loss_i)
                    val_accuracy.append(accuracy_i)

                val_loss = np.mean(val_loss)
                val_accuracy = np.mean(val_accuracy)
                val_loss_total.append(val_loss)


                test_loss = []
                test_accuracy = []
                predicted_list = []
                for x_seq,x_freq, y in get_batches_model(self.test[0], self.test[1],self.test[-1], batch_size=batch_size, random=False):
                    feed_dict = {X_Seq: x_seq, X_Freq: x_freq, Y: y}
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
                            print('Done Training')
                            break

                    else:
                        if self.LOO is None:
                            a, b, c = -10, -7, -3
                            if (np.mean(val_loss_total[a:b]) - np.mean(val_loss_total[c:])) / np.mean(val_loss_total[a:b]) < stop_criterion:
                                print('Done Training')
                                break
                        else:
                            a, b, c = -10, -7, -3
                            if (np.mean(train_loss_total[a:b]) - np.mean(train_loss_total[c:])) / np.mean(train_loss_total[a:b]) < stop_criterion:
                                print('Done Training')
                                break

                            if np.mean(train_accuracy_total[-100:]) == 1.0:
                                print('Done Training')
                                break


            #saver.save(sess, self.Name + '_WF/' + self.Name + '_WF.ckpt')
            features_WF_list = []
            features_seq_list = []
            indices_list = []
            for x_seq, x_freq, y in get_batches_model(self.X_Seq,self.X_Freq, self.Y, batch_size=batch_size, random=False):
                feed_dict = {X_Seq: x_seq, X_Freq: x_freq}
                features_wf_i, features_i,indices_i = sess.run([fc_avg,Seq_Features, Indices], feed_dict=feed_dict)
                features_WF_list.append(features_wf_i)
                features_seq_list.append(features_i)
                indices_list.append(indices_i)

            self.features_wf = np.vstack(features_WF_list)
            self.features_seq = np.vstack(features_seq_list)
            self.indices = np.vstack(indices_list)
            self.seq = self.sequences
            self.kernel = kernel

            if plot_loss is True:
                plt.close()


            with open(self.Name+'/'+self.Name + '_features_WF.pkl', 'wb') as f:
                pickle.dump([self.features_wf, self.features_seq,self.indices,self.seq,
                             self.X_Freq, self.y_pred, self.y_test,self.labels,self.Y,self.files,self.kernel], f,protocol=4)

    def Get_Features(self):
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.import_meta_graph(self.Name + '_WF/' + self.Name + '_WF.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint(self.Name + '_WF'))
            graph = tf.get_default_graph()
            input = graph.get_tensor_by_name('Input_Seq:0')
            input_f = graph.get_tensor_by_name('Input_Freq:0')
            Seq_Features = graph.get_tensor_by_name('Seq_Features:0')
            predicted = graph.get_tensor_by_name('predicted:0')

            feed_dict = {input:self.X_Seq,input_f:self.X_Freq}
            Seq_Features_Out,Predicted_Out = sess.run([Seq_Features,predicted],feed_dict=feed_dict)

        write_dir = self.directory_results+'Feature_Values/'
        if not os.path.isdir(write_dir):
            os.makedirs(write_dir)

        p_val = []
        for s in range(Seq_Features_Out.shape[-1]):
            seq_f = []
            seq_labels = []
            for ii, (label,i) in enumerate(zip(self.labels,self.X_Seq)):
                sel = np.sum(i, 1) > 0
                seq_features = Seq_Features_Out[ii, sel, s]
                seq_f+=list(seq_features)
                seq_labels += [label]*len(seq_features)

            df = pd.DataFrame()
            df['Label'] = seq_labels
            df['Feature Value'] = seq_f

            val_1 = df['Feature Value'][df['Label']=='CMV'].tolist()
            val_2 = df['Feature Value'][df['Label']=='EBV'].tolist()
            stat, p = mannwhitneyu(val_1,val_2)
            p_val.append(p)

            plt.figure()
            sns.violinplot(data=df,x='Label',y='Feature Value')
            plt.savefig(write_dir+'Feature_'+str(s)+'.tif')
            plt.close()

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

        self.group_features_wf: Pandas Dataframe
            Sequences used to determine motifs in fasta files
            are stored in this dataframe where column names represent
            the feature number.

        """

        #Get Features
        with open(self.Name+'/'+self.Name + '_features_WF.pkl', 'rb') as f:
            self.features_wf, self.features_seq, self.indices, self.seq,\
            self.X_Freq, self.y_pred, self.y_test,self.labels,self.Y,self.files,self.kernel = pickle.load(f)

        # # #Create HM to visualize results
        # color_dict = {'CMV+':'g','CMV-':'r'}
        # row_colors = [color_dict[x] for x in self.labels]
        #
        # keep=[]
        # for i,column in enumerate(self.features_wf.T,0):
        #     if len(np.unique(column)) > 1:
        #         keep.append(i)
        # keep = np.asarray(keep)
        # self.features_wf = self.features_wf[:, keep]
        #
        # df = pd.DataFrame(self.features_wf)
        # # #df.index = self.files
        # CM = sns.clustermap(df, standard_scale=1,cmap='bwr',row_colors=row_colors)
        # # ax = CM.ax_heatmap
        # # ax.set_yticklabels('')

        group_num = np.where(self.lb.classes_ == group)[0][0]

        #Find diff expressed features
        idx_pos = self.Y[:, group_num] == 1
        idx_neg = self.Y[:, group_num] == 0

        pos_mean = []
        neg_mean = []
        p_val = []
        feature_num = list(range(len(self.features_wf.T)))
        for i in feature_num:
            pos = self.features_wf[idx_pos,i]
            neg = self.features_wf[idx_neg,i]
            pos_mean.append(np.mean(pos))
            neg_mean.append(np.mean(neg))
            try:
                stat, p = mannwhitneyu(pos,neg)
                p_val.append(p)
            except:
                p_val.append(1.0)

        df_features = pd.DataFrame()
        df_features['Feature'] = feature_num
        df_features['P_Val'] = p_val
        df_features['Pos'] = pos_mean
        df_features['Neg'] = neg_mean
        df_features['Mag'] = df_features['Pos'] - df_features['Neg']
        df_features = df_features[df_features['P_Val']< p_val_threshold]

        df_features.sort_values(by='Mag',inplace=True,ascending=False)

        features_seq = np.reshape(self.features_seq, [-1, self.features_seq.shape[-1]])
        sequences = np.asarray(np.hstack(self.seq).tolist())
        indices = np.asarray(np.reshape(self.indices, [-1, self.indices.shape[-1]]))

        #Get motifs for positive features
        dir = self.directory_results+group + '_WF_Motifs'
        if not os.path.exists(dir):
            os.makedirs(dir)

        file_list = [f for f in os.listdir(dir)]
        [os.remove(os.path.join(dir, f)) for f in file_list]

        top_seq = 10
        seq_cluster = []
        feature_keep = []
        seq_thresh_pos = []
        for feature in df_features['Feature'].tolist():
            if df_features['Mag'][feature] > 0:
                feature_keep.append(feature)
                sel = np.flip(features_seq[:, feature].argsort(), -1)
                feature_sort = features_seq[sel,feature]
                seq_thresh_pos.append(np.percentile(feature_sort,cut))
                sel = sel[0:top_seq]
                seq_sel = sequences[sel]
                ind_sel = indices[sel, feature]
                seq_cluster.append(seq_sel)

                motifs = []
                for ii, i in enumerate(ind_sel, 0):
                    motif = seq_sel[ii][int(i):int(i) + self.kernel]
                    if len(motif) < self.kernel:
                        motif = motif + 'X' * (self.kernel - len(motif))
                    motif = motif.lower()
                    motif = SeqRecord(Seq(motif, IUPAC.protein), str(ii))
                    motifs.append(motif)

                SeqIO.write(motifs, dir+'/feature_' + str(feature) + '.fasta', 'fasta')

        seq_features_df_pos = pd.DataFrame()
        for ii, f in enumerate(feature_keep, 0):
            seq_features_df_pos[f] = seq_cluster[ii]

        self.group_features_wf = seq_features_df_pos
        seq_features_df_pos.to_csv(dir+'/feature_sequences.csv',index=False)

        if save_images is True:
            for feature, thresh in zip(feature_keep, seq_thresh_pos):
                labels = []
                values = []
                for g in self.lb.classes_:
                    sel = self.labels == g
                    features_seq_sel = (self.features_seq[sel, :, feature] > thresh) * self.X_Freq[sel]
                    features_contrib = list(np.sum(features_seq_sel, -1))
                    labels += [g] * np.sum(sel)
                    values += features_contrib

                df = pd.DataFrame()
                df['Cohort'] = labels
                df['Fraction of Response'] = values
                plt.figure()
                sns.set(font_scale=1)
                sns.violinplot(data=df, x='Cohort', y='Fraction of Response')
                plt.title('Feature ' + str(feature))
                plt.savefig(dir + '/feature' + str(feature) + '.tif')
                plt.close()

    def Monte_Carlo_CrossVal(self, fold=5, test_size=0.25, epochs_min=5, batch_size=25, LOO=None,stop_criterion=0.001,
                             kernel=5,units=12,weight_by_class=False, trainable_embedding=True,accuracy_min = None, weight_by_freq = True,plot_loss=False):


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


        Returns
        ---------------------------------------

        """

        y_pred = []
        y_test = []
        for i in range(0, fold):
            print(i)
            self.Get_Train_Valid_Test_WF(test_size=test_size, LOO=LOO)
            self.Train_WF(epochs_min=epochs_min, batch_size=batch_size,stop_criterion=stop_criterion,
                          kernel=kernel,units=units,weight_by_class=weight_by_class,
                          trainable_embedding=trainable_embedding,accuracy_min=accuracy_min,
                          weight_by_freq = weight_by_freq,plot_loss=plot_loss)

            y_test.append(self.y_test)
            y_pred.append(self.y_pred)

            y_test2 = np.vstack(y_test)
            y_pred2 = np.vstack(y_pred)
            print("Accuracy = {}".format(np.average(np.equal(np.argmax(y_pred2,1),np.argmax(y_test2,1)))))

            if self.y_test.shape[1]==2:
                if i > 0:
                    y_test2 = np.vstack(y_test)
                    if (np.sum(y_test2[:, 0]) != len(y_test2)) and (np.sum(y_test2[:, 0]) != 0):
                        print("AUC = {}".format(roc_auc_score(np.vstack(y_test), np.vstack(y_pred))))


        self.y_test = np.vstack(y_test)
        self.y_pred = np.vstack(y_pred)

    def K_Fold_CrossVal(self,folds=5,epochs_min=5,batch_size=25,stop_criterion=0.001, kernel=5,units=12, weight_by_class=False, iterations=None,
                        trainable_embedding=True, accuracy_min = None, weight_by_freq = True, plot_loss=False):

        """
        K_Fold Cross-Validation for Whole Sample Classifier

        If the number of samples is small but training the whole sample classifier, one
        can use K_Fold Cross Validation to train on all but one before assessing
        predictive performance.After this method is run, the AUC_Curve method can be run to
        assess the overall performance.

        Inputs
        ---------------------------------------
        fold_size: int
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


        Returns
        ---------------------------------------

        """

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
            print(ii)
            train_idx = np.setdiff1d(idx,test_idx[ii])

            Vars = [self.X_Seq, self.X_Freq,self.files,np.asarray(self.sequences)]
            self.train, self.test = Get_Train_Test(Vars=Vars,train_idx=train_idx,test_idx = test_idx[ii],Y=self.Y)
            self.valid = self.test
            self.LOO = True

            self.Train_WF(epochs_min=epochs_min, batch_size=batch_size,
                          stop_criterion=stop_criterion, kernel=kernel,
                          units=units, weight_by_class=weight_by_class,
                          trainable_embedding=trainable_embedding,accuracy_min = accuracy_min,
                          weight_by_freq = weight_by_freq, plot_loss = plot_loss)


            y_test.append(self.y_test)
            y_pred.append(self.y_pred)

            y_test2 = np.vstack(y_test)
            y_pred2 = np.vstack(y_pred)
            print("Accuracy = {}".format(np.average(np.equal(np.argmax(y_pred2,1),np.argmax(y_test2,1)))))

            if self.y_test.shape[1]==2:
                if ii > 0:
                    if (np.sum(y_test2[:, 0]) != len(y_test2)) and (np.sum(y_test2[:, 0]) != 0):
                        print("AUC = {}".format(roc_auc_score(np.vstack(y_test), np.vstack(y_pred))))


            if iterations is not None:
                if ii > iterations:
                    break

        self.y_test = np.vstack(y_test)
        self.y_pred = np.vstack(y_pred)













































