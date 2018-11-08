import os
import shutil
import sys
sys.path.append('../')
from DeepTCR.unsupervised_functions.utils import *
from DeepTCR.unsupervised_functions.Layers import *
from DeepTCR.supervised_functions.data_processing import *
import glob
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool
import pickle
import seaborn as sns
import colorsys
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,fcluster

class DeepTCR_U(object):

    def __init__(self,Name,max_length=40):
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

    def Get_Data(self,directory,Load_Prev_Data=False,classes=None,type_of_data_cut='Fraction_Response',data_cut=1.0,n_jobs=40,
                    aa_column = None, count_column = None,sep='\t'):
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

            p = Pool(n_jobs)

            if sep == '\t':
                ext = '/*.tsv'
            elif sep == ',':
                ext = '/*.csv'
            else:
                print('Not Valid Delimiter')
                return

            sequences = []
            label_id = []
            file_id = []
            freq = []
            file_list = []
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
                    file_list.append(file.split('/')[-1])
                    freq += df['Frequency'].tolist()

            sequences = np.asarray(sequences)
            label_id = np.asarray(label_id)
            file_id = np.asarray(file_id)
            freq = np.asarray(freq)

            args = list(zip(sequences, [self.aa_idx] * len(sequences), [self.max_length] * len(sequences)))
            result = p.starmap(Embed_Seq_Num, args)
            p.close()
            sequences_num = np.vstack(result)
            X_Seq = np.expand_dims(sequences_num, 1)

            with open(self.Name+'/'+self.Name + '_Data.pkl', 'wb') as f:
                pickle.dump([X_Seq, sequences, label_id, file_id, freq,self.lb,file_list],f,protocol=4)

        else:
            with open(self.Name+'/'+self.Name + '_Data.pkl', 'rb') as f:
                X_Seq, sequences, label_id, file_id, freq,self.lb,file_list = pickle.load(f)

        self.X_Seq = X_Seq
        self.sequences = sequences
        self.label_id = label_id
        self.file_id = file_id
        self.freq = freq
        self.file_list = file_list
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
        ---------------------------------------

        """

        if Load_Prev_Data is False:
            with tf.device('/gpu:0'):
                graph_model_AE = tf.Graph()
                with graph_model_AE.as_default():
                    X_Seq = tf.placeholder(tf.int64, shape=[None, self.X_Seq.shape[1], self.X_Seq.shape[2]], name='Input')
                    X_Seq_OH = tf.one_hot(X_Seq, depth=21)
                    training = tf.placeholder_with_default(False, shape=())
                    prob = tf.placeholder_with_default(0.0, shape=(), name='prob')

                    # AA Embedding
                    with tf.variable_scope('AA_Embedding'):
                        embedding_dim_aa = 64
                        embedding_layer_seq = tf.get_variable(name='Embedding_Layer_Seq', shape=[21, embedding_dim_aa])
                        embedding_layer_seq = tf.expand_dims(tf.expand_dims(embedding_layer_seq, axis=0), axis=0)
                        inputs_seq_embed = tf.squeeze(tf.tensordot(X_Seq_OH, embedding_layer_seq, axes=(3, 2)), axis=(3, 4))

                    # Convolutional Features
                    Seq_Features = Convolutional_Features_AE(inputs_seq_embed, training=training, prob=prob)

                    fc = tf.layers.dense(Seq_Features, 256)
                    fc = tf.layers.dense(fc, 128)

                    z_mean = tf.layers.dense(fc, latent_dim, activation=None, name='z_mean')
                    z_log_var = tf.layers.dense(fc, latent_dim, activation=tf.nn.softplus, name='z_log_var')

                    z = z_mean + tf.exp(z_log_var / 2) * tf.random_normal(tf.shape(z_mean), 0.0, 1.0, dtype=tf.float32)
                    z = tf.identity(z, name='z')

                    fc_up = tf.layers.dense(z, 128)
                    fc_up = tf.layers.dense(fc_up, 256)
                    fc_up = tf.reshape(fc_up, shape=[-1, 1, 4, 64])

                    upsample1 = tf.layers.conv2d_transpose(fc_up, 12, (1, 3), (1, 2), activation=tf.nn.relu)
                    upsample2 = tf.layers.conv2d_transpose(upsample1, 32, (1, 3), (1, 2), activation=tf.nn.relu)
                    upsample3 = tf.layers.conv2d_transpose(upsample2, embedding_dim_aa, (1, 4), (1, 2), activation=tf.nn.relu)

                    embedding_layer_seq_back = tf.transpose(embedding_layer_seq, perm=(0, 1, 3, 2))
                    logits_AE = tf.squeeze(tf.tensordot(upsample3, embedding_layer_seq_back, axes=(3, 2)), axis=(3, 4), name='logits')

                    total_cost, recon_cost, latent_cost = AE_Loss(X_Seq, logits_AE, z_mean, z_log_var)

                    predicted = tf.squeeze(tf.argmax(logits_AE, axis=3), axis=1)
                    actual_ae = tf.squeeze(X_Seq, axis=1)
                    w = tf.cast(tf.squeeze(tf.greater(X_Seq, 0), 1), tf.float32)
                    correct_ae = tf.reduce_sum(w * tf.cast(tf.equal(predicted, actual_ae), tf.float32), axis=1) / tf.reduce_sum(w, axis=1)
                    accuracy = tf.reduce_mean(correct_ae, axis=0)

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
                    for x_ae_seq in get_batches_seq(self.X_Seq, batch_size=batch_size):
                        feed_dict = {X_Seq: x_ae_seq, training: True}
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

                #saver.save(sess, self.Name + '_VAE/' + self.Name + '_VAE.ckpt')

                features_list = []
                recon_list = []
                accuracy_list = []
                for x_ae_seq in get_batches_seq(self.X_Seq, batch_size=batch_size, random=False):
                    feed_dict = {X_Seq: x_ae_seq}
                    features_ind, recon_ind, accuracy_check,embedding_layer = sess.run([z_mean, logits_AE, accuracy,embedding_layer_seq], feed_dict=feed_dict)
                    features_list.append(features_ind)
                    recon_list.append(np.squeeze(recon_ind, 1))
                    accuracy_list.append(accuracy_check)

                # with open('Embedding_Layer_Test.pkl','wb') as f:
                #     pickle.dump(embedding_layer,f)

                features = np.vstack(features_list)
                accuracy_list = np.hstack(accuracy_list)
                recon = np.vstack(recon_list)
                print('Reconstruction Accuracy: {:.5f}'.format(np.nanmean(accuracy_list)))

            with open(self.Name+'/'+self.Name + '_VAE_features.pkl', 'wb') as f:
                pickle.dump([features, recon], f,protocol=4)

        else:
            with open(self.Name+'/'+self.Name + '_VAE_features.pkl', 'rb') as f:
                features,recon = pickle.load(f)


        self.features = features
        self.recon = recon

        self.features = features
        keep=[]
        for i,column in enumerate(self.features.T,0):
            if len(np.unique(column)) > 1:
                keep.append(i)
        keep = np.asarray(keep)
        self.features = self.features[:,keep]
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
        ---------------------------------------

        """
        if Load_Prev_Data is False:
            z_dim = 256
            graph_model = tf.Graph()
            j=21
            epochs = 500

            with tf.device('/gpu:0'):
                with graph_model.as_default():
                    # Setup Placeholders
                    X_Seq = tf.placeholder(tf.int64, [None, self.X_Seq.shape[1], self.X_Seq.shape[2]], name='Input_Seq')
                    X_Seq_OH = tf.one_hot(X_Seq, depth=j)
                    training = tf.placeholder_with_default(False, shape=())
                    prob = tf.placeholder_with_default(0.0, shape=(), name='prob')

                    # AA Embedding
                    with tf.variable_scope('AA_Embedding'):
                        embedding_dim_aa = 64
                        embedding_layer_seq = tf.get_variable(name='Embedding_Layer_Seq', shape=[j, embedding_dim_aa])
                        embedding_layer_seq = tf.expand_dims(tf.expand_dims(embedding_layer_seq, axis=0), axis=0)
                        inputs_seq_embed = tf.squeeze(tf.tensordot(X_Seq_OH, embedding_layer_seq, axes=(3, 2)), axis=(3, 4))

                    latent_real,indices_real = Convolutional_Features_GAN(inputs_seq_embed,training=training,prob=prob,
                                                                          units=latent_dim)
                    latent_real = tf.identity(latent_real,'latent_real')
                    logits_real = tf.layers.dense(latent_real,1,name='logits_real')

                    inputs_z = tf.placeholder(tf.float32, shape=[None, z_dim])
                    gen_seq = generator(inputs_z, training=training,embedding_dim_aa=embedding_dim_aa,prob=prob)
                    gen_seq_out = tf.squeeze(tf.tensordot(gen_seq,tf.transpose(embedding_layer_seq,perm=[0,1,3,2]),axes=(3,2)),axis=(3,4))
                    latent_fake,indices_fake = Convolutional_Features_GAN(gen_seq,reuse=True,prob=prob,training=training,units=latent_dim)
                    latent_fake = tf.identity(latent_fake,'latent_fake')
                    logits_fake = tf.layers.dense(latent_fake,1,name='logits_fake')


                    d_loss, g_loss = model_loss(logits_real, logits_fake,latent_real,latent_fake)

                    var_list = tf.trainable_variables()
                    var_train = [x for x in var_list if not x.name.startswith('generator')]
                    opt_d = tf.train.RMSPropOptimizer(learning_rate=0.0002).minimize(d_loss, var_list=var_train)

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

                    for x_seq,y in get_batches(self.X_Seq,self.X_Seq,batch_size=batch_size,random=True):
                        step +=1
                        batch_z = np.random.normal(size=(batch_size, z_dim))

                        d_loss_i,__= sess.run([d_loss, opt_d], feed_dict={X_Seq: x_seq,inputs_z:batch_z, prob: drop_out_rate, training: True})
                        d_loss_list.append(d_loss_i)
                        if suppress_output is False:
                            print("D_Loss = {} ".format(d_loss_i), end='', flush=True)

                        g_loss_i, _ = sess.run([g_loss, opt_g], feed_dict={X_Seq: x_seq,inputs_z:batch_z, prob: drop_out_rate, training: True})
                        g_loss_list.append(g_loss_i)
                        if suppress_output is False:
                            print("G_Loss = {}".format(g_loss_i))

                        if step % 10 ==0:
                            batch_z = np.random.normal(size=(5, z_dim))
                            feed_dict = {inputs_z: batch_z}
                            gen_out = sess.run(gen_seq_out, feed_dict=feed_dict)
                            check = np.squeeze(np.argmax(gen_out, -1), 1)

                            seq_list = []
                            for seq in check:
                                seq_out = []
                                for i in seq:
                                    if i != 0:
                                        seq_out.append(self.aa_idx_inv[i])
                                seq_list.append(''.join(seq_out))

                            seq_list = np.asarray(seq_list)
                            if suppress_output is False:
                                print(seq_list)

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


                # plt.figure()
                # plt.plot(d_loss_list,label='Discriminator Loss')
                # plt.plot(g_loss_list,label='Generator loss')
                # plt.legend(loc='best')
                # plt.xlabel('Epochs')
                # plt.ylabel('Loss')

                latent_features = []
                latent_indices = []
                for x_seq, y in get_batches(self.X_Seq, self.X_Seq, batch_size=batch_size):
                    latent_i,indices_i = sess.run([latent_real,indices_real],feed_dict={X_Seq:x_seq})
                    latent_features.append(latent_i)
                    latent_indices.append(indices_i)
                features = np.vstack(latent_features)
                indices = np.vstack(latent_indices)


            with open(self.Name+'/'+self.Name + '_GAN_features.pkl','wb') as f:
                pickle.dump([features,indices],f,protocol=4)

        else:

            with open(self.Name+'/'+self.Name + '_GAN_features.pkl','rb') as f:
                features,indices = pickle.load(f)


        self.features = features
        self.indices = indices
        keep=[]
        for i,column in enumerate(self.features.T,0):
            if len(np.unique(column)) > 1:
                keep.append(i)
        keep = np.asarray(keep)
        self.features = self.features[:,keep]
        self.indices = self.indices[:,keep]
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
            self.sequences = self.sequences[sel]

        if sample_num_per_seq is not None:
            features_temp = []
            label_temp = []
            file_temp = []
            seq_temp = []
            for i in self.lb.classes_:
                sel = np.where(self.label_id==i)[0]
                sel = np.random.choice(sel,sample_num_per_seq,replace=False)
                features_temp.append(self.features[sel])
                label_temp.append(self.label_id[sel])
                file_temp.append(self.file_id[sel])
                seq_temp.append(self.sequences[sel])

            self.features = np.vstack(features_temp)
            self.label_id = np.hstack(label_temp)
            self.file_id = np.hstack(file_temp)
            self.sequences = np.hstack(seq_temp)


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
        plt.savefig(self.directory_results+filename)

    def HeatMap_Files(self,filename='Heatmap_Files.tif',Weight_by_Freq=True,color_dict=None,labels=True):
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
        sns.set(font_scale=1.0)
        CM = sns.clustermap(dfs, standard_scale=1, cmap='bwr', figsize=(12, 10), row_colors=row_colors)
        ax = CM.ax_heatmap
        ax.set_xticklabels('')
        if labels is False:
            ax.set_yticklabels('')
        plt.subplots_adjust(right=0.8)
        plt.show()
        plt.savefig(self.directory_results+filename)

    def Cluster(self,t=100,criterion='distance',write_to_sheets=False):
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

        Returns

        self.DFs: list of Pandas dataframes
            Clusters by sequences/label

        self.var: list
            Variance of lengths in each cluster

        self.Cluster_Frequencies: Pandas dataframe
            A dataframe containing the frequency contribution of each cluster to each sample.

        ---------------------------------------

        """
        #Normalize Features
        from sklearn.preprocessing import StandardScaler
        SS= StandardScaler()
        features=SS.fit_transform(self.features)

        #Hierarchical Clustering
        Z = linkage(features, method='ward',metric='euclidean')
        IDX = fcluster(Z,t,criterion=criterion)


        DFs = []
        DF_Sum = pd.DataFrame()
        DF_Sum['File'] = self.file_list
        DF_Sum.set_index('File',inplace=True)
        var_list = []
        for i in np.unique(IDX):
            if i != -1:
                sel = IDX == i
                seq = self.sequences[sel]
                label = self.label_id[sel]
                file = self.file_id[sel]
                freq = self.freq[sel]

                len_sel = [len(x) for x in seq]
                var = max(len_sel) - min(len_sel)
                var_list.append(var)

                df = pd.DataFrame()
                df['Sequences'] = seq
                df['Labels'] = label
                df['File'] = file
                df['Frequency'] = freq

                df_sum = df.groupby(by='File',sort=False).agg({'Frequency':'sum'})

                DF_Sum['Cluster_'+ str(i)] = df_sum

                DFs.append(df)


        if write_to_sheets is True:
            if not os.path.exists(self.directory_results+'Clusters/'):
                os.makedirs(self.directory_results+'Clusters/')
            else:
                shutil.rmtree(self.directory_results+'Clusters/')
                os.makedirs(self.directory_results + 'Clusters/')

            for ii,df in enumerate(DFs,1):
                df.to_csv(self.directory_results+'Clusters/'+str(ii)+'.csv',index=False)

            DF_Sum.to_csv(self.directory_results+'Cluster_Frequencies_by_Sample.csv')

        self.DFs = DFs
        self.Cluster_Frequencies = DF_Sum
        self.var = var_list
        print('Clustering Done')


























































































