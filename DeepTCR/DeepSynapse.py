import sys
sys.path.append('../')
from DeepTCR.functions_syn.Layers import *
from DeepTCR.functions_syn.utils_s import *
from DeepTCR.functions_syn.act_fun import *
from DeepTCR.functions_syn.data_processing import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import shutil
import warnings
from scipy.stats import spearmanr,gaussian_kde
from distinctipy import distinctipy
from tqdm import tqdm

class DeepSynapse(object):

    def __init__(self,Name,
                 device=0,
                 tf_verbosity=3):
        """
        # Initialize Training Object.
        Initializes object and sets initial parameters.

        All DeepTCR algorithms begin with initializing a training object. This object will contain all methods, data, and results during the training process. One can extract learned features, per-sequence predictions, among other outputs from DeepTCR and use those in their own analyses as well.

        This method is included in the three main DeepTCR objects:

        - DeepTCR_U (unsupervised)
        - DeepTCR_SS (supervised sequence classifier/regressor)
        - DeepTCR_WF (supervised repertoire classifier/regressor)


        Args:
            Name (str): Name of the object. This name will be used to create folders with results as well as a folder with parameters and specifications for any models built/trained.

            max_length (int): maximum length of CDR3 sequence.

            device (int): In the case user is using tensorflow-gpu, one can specify the particular device to build the graphs on. This selects which GPU the user wants to put the graph and train on.

            tf_verbosity (str): determines how much tensorflow log output to display while training.
            0 = all messages are logged (default behavior)
            1 = INFO messages are not printed
            2 = INFO and WARNING messages are not printed
            3 = INFO, WARNING, and ERROR messages are not printed

        """

        #Assign parameters
        self.Name = Name
        self.use_beta = False
        self.use_alpha = False
        self.device = '/device:GPU:'+str(device)
        self.use_v_beta = False
        self.use_d_beta = False
        self.use_j_beta = False
        self.use_v_alpha = False
        self.use_j_alpha = False
        self.use_hla = False
        self.use_hla_sup = False
        self.keep_non_supertype_alleles = False
        self.use_hla_seq = False
        self.use_epitope=False
        self.regression = False
        self.use_w = False
        self.ind = None
        self.unknown_str = '__unknown__'

        #Create dataframes for assigning AA to ints
        aa_idx, aa_mat = make_aa_df()
        aa_idx_inv = {v: k for k, v in aa_idx.items()}
        self.aa_idx = aa_idx
        self.aa_idx_inv = aa_idx_inv

        #Create directory for results of analysis
        directory = os.path.join(self.Name,'results')
        self.directory_results = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        #Create directory for any temporary files
        directory = self.Name
        if not os.path.exists(directory):
            os.makedirs(directory)

        tf.compat.v1.disable_eager_execution()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_verbosity)

    def Load_Data(self,alpha_sequences=None,beta_sequences=None,v_beta=None,d_beta=None,j_beta=None,
                  v_alpha=None,j_alpha=None,class_labels=None,sample_labels=None,freq=None,counts=None,Y=None,
                  p=None,hla=None,use_hla_supertype=False,keep_non_supertype_alleles=False,use_hla_seq=False,
                  epitope_sequences=None,
                  w=None):
        """
        # Load Data programatically into DeepTCR.

        DeepTCR allows direct user input of sequence data for DeepTCR analysis. By using this method,
        a user can load numpy arrays with relevant TCRSeq data for analysis.

        Tip: One can load data with the Get_Data command from directories and then reload it into another DeepTCR object with the Load_Data command. This can be useful, for example, if you have different labels you want to train to, and you need to change the label programatically between training each model. In this case, one can load the data first with the Get_Data method and then assign the labels pythonically before feeding them into the DeepTCR object with the Load_Data method.

        Of note, this method DOES NOT combine sequences with the same amino acid sequence. Therefore, if one wants this, one should first do it programatically before feeding the data into DeepTCR with this method.

        Another special use case of this method would be for any type of regression task (sequence or repertoire models). In the case that a per-sequence value is fed into DeepTCR (with Y), this value either becomes the per-sequence regression value or the average of all Y over a sample becomes the per-sample regression value. This is another case where one might want to load data with the Get_Data method and then reload it into DeepTCR with regression values.

        This method is included in the three main DeepTCR objects:

        - DeepTCR_U (unsupervised)
        - DeepTCR_SS (supervised sequence classifier/regressor)
        - DeepTCR_WF (supervised repertoire classifier/regressor)

        Args:

            alpha_sequences (ndarray of strings): A 1d array with the sequences for inference for the alpha chain.

            beta_sequences (ndarray of strings): A 1d array with the sequences for inference for the beta chain.

            v_beta (ndarray of strings): A 1d array with the v-beta genes for inference.

            d_beta (ndarray of strings): A 1d array with the d-beta genes for inference.

            j_beta (ndarray of strings): A 1d array with the j-beta genes for inference.

            v_alpha (ndarray of strings): A 1d array with the v-alpha genes for inference.

            j_alpha (ndarray of strings): A 1d array with the j-alpha genes for inference.

            class_labels (ndarray of strings): A 1d array with class labels for the sequence (i.e. antigen-specificities)

            sample_labels (ndarray of strings): A 1d array with sample labels for the sequence. (i.e. when loading data from different samples)

            counts (ndarray of ints): A 1d array with the counts for each sequence, in the case they come from samples.

            freq (ndarray of float values): A 1d array with the frequencies for each sequence, in the case they come from samples.

            Y (ndarray of float values): In the case one wants to regress TCR sequences or repertoires against a numerical label, one can provide these numerical values for this input. For the TCR sequence regressor, each sequence will be regressed to the value denoted for each sequence. For the TCR repertoire regressor, the average of all instance level values will be used to regress the sample. Therefore, if there is one sample level value for regression, one would just repeat that same value for all the instances/sequences of the sample.

            hla (ndarray of tuples/arrays): To input the hla context for each sequence fed into DeepTCR, this will need to formatted as an ndarray that is (N,) for each sequence where each entry is a tuple or array of strings referring to the alleles seen for that sequence. ('A*01:01', 'A*11:01', 'B*35:01', 'B*35:02', 'C*04:01')

            use_hla_supertype (bool): Given the diversity of the HLA-loci, training with a full allele may cause over-fitting. And while individuals may have different HLA alleles, these different allelees may bind peptide in a functionality similar way. This idea of supertypes of HLA is a method by which assignments of HLA genes can be aggregated to 6 HLA-A and 6 HLA-B supertypes. In roder to convert input of HLA-allele genes to supertypes, a more biologically functional representation, one can se this parameter to True and if the alleles provided are of one of 945 alleles found in the reference below, it will be assigned to a known supertype.

                - For this method to work, alleles must be provided in the following format: A0101 where the first letter of the designation is the HLA loci (A or B) and then the 4 digit gene designation. HLA supertypes only exist for HLA-A and HLA-B. All other alleles will be dropped from the analysis.

                - Sidney, J., Peters, B., Frahm, N., Brander, C., & Sette, A. (2008). HLA class I supertypes: a revised and updated classification. BMC immunology, 9(1), 1.

            keep_non_supertype_alleles (bool): If assigning supertypes to HLA alleles, one can choose to keep HLA-alleles that do not have a known supertype (i.e. HLA-C alleles or certain HLA-A or HLA-B alleles) or discard them for the analysis. In order to keep these alleles, one should set this parameter to True. Default is False and non HLA-A or B alleles will be discarded.

            p (multiprocessing pool object): a pre-formed pool object can be passed to method for multiprocessing tasks.

            w (ndarray): optional set of weights for training of autoencoder

        Returns:
            variables into training object

            - self.alpha_sequences (ndarray): array with alpha sequences (if provided)
            - self.beta_sequences (ndarray): array with beta sequences (if provided)
            - self.label_id (ndarray): array with sequence class labels
            - self.file_id (ndarray): array with sequence file labels
            - self.freq (ndarray): array with sequence frequencies from samples
            - self.counts (ndarray): array with sequence counts from samples
            - self.(v/d/j)_(alpha/beta) (ndarray):array with sequence (v/d/j)-(alpha/beta) usage

        ---------------------------------------

        """

        inputs = [alpha_sequences,beta_sequences,v_beta,d_beta,j_beta,v_alpha,j_alpha,
                  class_labels,sample_labels,counts,freq,Y,hla,w]

        for i in inputs:
            if i is not None:
                assert isinstance(i,np.ndarray),'Inputs into DeepTCR must come in as numpy arrays!'

        inputs = [alpha_sequences,beta_sequences,epitope_sequences,v_beta,d_beta,j_beta,v_alpha,j_alpha,hla]
        for i in inputs:
            if i is not None:
                len_input = len(i)
                break

        if p is None:
            p_ = Pool(40)
        else:
            p_ = p

        if alpha_sequences is not None:
            self.alpha_sequences = alpha_sequences
            self.max_length_alpha = np.max(np.vectorize(len)(alpha_sequences))
            args = list(zip(alpha_sequences, [self.aa_idx] * len(alpha_sequences), [self.max_length_alpha] * len(alpha_sequences)))
            result = p_.starmap(Embed_Seq_Num, args)
            sequences_num = np.vstack(result)
            self.X_Seq_alpha = np.expand_dims(sequences_num, 1)
            self.use_alpha = True
        else:
            self.X_Seq_alpha = np.zeros(shape=[len_input])
            self.alpha_sequences = np.asarray([None] * len_input)

        if beta_sequences is not None:
            self.beta_sequences = beta_sequences
            self.max_length_beta = np.max(np.vectorize(len)(beta_sequences))
            args = list(zip(beta_sequences, [self.aa_idx] * len(beta_sequences), [self.max_length_beta] * len(beta_sequences)))
            result = p_.starmap(Embed_Seq_Num, args)
            sequences_num = np.vstack(result)
            self.X_Seq_beta = np.expand_dims(sequences_num, 1)
            self.use_beta = True
        else:
            self.X_Seq_beta = np.zeros(shape=[len_input])
            self.beta_sequences = np.asarray([None] * len_input)

        if epitope_sequences is not None:
            self.epitope_sequences = epitope_sequences
            self.max_length_epitope = np.max(np.vectorize(len)(epitope_sequences))
            args = list(zip(epitope_sequences, [self.aa_idx] * len(epitope_sequences), [self.max_length_epitope] * len(epitope_sequences)))
            result = p_.starmap(Embed_Seq_Num, args)
            sequences_num = np.vstack(result)
            self.X_Seq_epitope = np.expand_dims(sequences_num, 1)
            self.use_epitope = True
        else:
            self.X_Seq_epitope = np.zeros(shape=[len_input])
            self.epitope = np.asarray([None] * len_input)

        if v_beta is not None:
            self.v_beta = v_beta
            self.lb_v_beta = LabelEncoder()
            self.lb_v_beta.classes_ = np.insert(np.unique(v_beta), 0, self.unknown_str)
            self.v_beta_num = self.lb_v_beta.transform(v_beta)
            self.use_v_beta = True
        else:
            self.lb_v_beta = LabelEncoder()
            self.v_beta_num = np.zeros(shape=[len_input])
            self.v_beta = np.asarray([None] * len_input)

        if d_beta is not None:
            self.d_beta = d_beta
            self.lb_d_beta = LabelEncoder()
            self.lb_d_beta.classes_ = np.insert(np.unique(d_beta), 0, self.unknown_str)
            self.d_beta_num = self.lb_d_beta.transform(d_beta)
            self.use_d_beta = True
        else:
            self.lb_d_beta = LabelEncoder()
            self.d_beta_num = np.zeros(shape=[len_input])
            self.d_beta = np.asarray([None] * len_input)

        if j_beta is not None:
            self.j_beta = j_beta
            self.lb_j_beta = LabelEncoder()
            self.lb_j_beta.classes_ = np.insert(np.unique(j_beta), 0, self.unknown_str)
            self.j_beta_num = self.lb_j_beta.transform(j_beta)
            self.use_j_beta = True
        else:
            self.lb_j_beta = LabelEncoder()
            self.j_beta_num = np.zeros(shape=[len_input])
            self.j_beta = np.asarray([None] * len_input)

        if v_alpha is not None:
            self.v_alpha = v_alpha
            self.lb_v_alpha = LabelEncoder()
            self.lb_v_alpha.classes_ = np.insert(np.unique(v_alpha), 0, self.unknown_str)
            self.v_alpha_num = self.lb_v_alpha.transform(v_alpha)
            self.use_v_alpha = True
        else:
            self.lb_v_alpha = LabelEncoder()
            self.v_alpha_num = np.zeros(shape=[len_input])
            self.v_alpha = np.asarray([None] * len_input)

        if j_alpha is not None:
            self.j_alpha = j_alpha
            self.lb_j_alpha = LabelEncoder()
            self.lb_j_alpha.classes_ = np.insert(np.unique(j_alpha), 0, self.unknown_str)
            self.j_alpha_num = self.lb_j_alpha.transform(j_alpha)
            self.use_j_alpha = True
        else:
            self.lb_j_alpha = LabelEncoder()
            self.j_alpha_num = np.zeros(shape=[len_input])
            self.j_alpha = np.asarray([None] * len_input)



        if counts is not None:
            if sample_labels is not None:
                count_dict={}
                for s in np.unique(sample_labels):
                    idx = sample_labels==s
                    count_dict[s]=np.sum(counts[idx])

                freq = []
                for c,n in zip(counts,sample_labels):
                    freq.append(c/count_dict[n])
                freq = np.asarray(freq)
                self.counts = counts
            else:
                print('Counts need to be provided with sample labels')
                return

        if freq is not None:
            self.freq = freq

        if sample_labels is not None:
            self.sample_id = sample_labels
        else:
            self.sample_id = np.asarray(['None']*len_input)

        if class_labels is not None:
            self.class_id = class_labels
        else:
            self.class_id = np.asarray(['None']*len_input)

        if (counts is None) & (freq is None):
            counts = np.ones(shape=len_input)
            count_dict = {}
            for s in np.unique(self.sample_id):
                idx = self.sample_id == s
                count_dict[s] = int(np.sum(counts[idx]))

            freq = []
            for c, n in zip(counts, self.sample_id):
                freq.append(c / count_dict[n])
            freq = np.asarray(freq)
            self.counts = counts
            self.freq = freq

        self.lb_hla = MultiLabelBinarizer()
        self.hla_data_seq_num = np.zeros([len_input, 1])
        self.hla_data_seq = np.zeros(len_input)
        if hla is not None:
            self.use_hla = True
            if use_hla_supertype:
                hla = supertype_conv_op(hla,keep_non_supertype_alleles)
                self.use_hla_sup = True
                self.keep_non_supertype_alleles = keep_non_supertype_alleles
                self.hla_data_seq_num = self.lb_hla.fit_transform(hla.reshape(-1,1))
                self.hla_data_seq = hla

            elif use_hla_seq:
                df_hla = load_hla_seq()
                hla_sequences = hla_seq_conv_op(hla,df_hla)
                self.hla_data_seq = hla_sequences

                self.max_length_hla = np.max(np.vectorize(len)(hla_sequences))
                args = list(zip(hla_sequences, [self.aa_idx] * len(hla_sequences),[self.max_length_hla] * len(hla_sequences)))
                result = p_.starmap(Embed_Seq_Num, args)
                sequences_num = np.vstack(result)
                self.hla_data_seq_num = np.expand_dims(sequences_num, 1)
                self.use_hla_seq = True
            else:
                self.hla_data_seq_num = self.lb_hla.fit_transform(hla.reshape(-1, 1))
                self.hla_data_seq = hla

        if p is None:
            p_.close()
            p_.join()

        if Y is not None:
            if Y.ndim == 1:
                Y = np.expand_dims(Y,-1)
            self.Y = Y
            self.lb = LabelEncoder()
            self.regression = True
        else:
            self.lb = LabelEncoder()
            Y = self.lb.fit_transform(self.class_id)
            OH = OneHotEncoder(sparse=False, categories='auto')
            Y = OH.fit_transform(Y.reshape(-1, 1))
            self.Y = Y

        if w is not None:
            self.use_w = True
            self.w = w
        else:
            self.w = np.ones(len_input)

        self.seq_index = np.asarray(list(range(len(self.Y))))
        if self.regression is False:
            self.predicted = np.zeros((len(self.Y),len(self.lb.classes_)))
        else:
            self.predicted = np.zeros([len(self.Y),1])
        self.sample_list = np.unique(self.sample_id)
        print('Data Loaded')

    def Get_Train_Valid_Test(self,test_size=0.25,LOO=None,split_by_sample=False,combine_train_valid=False):
        """
        # Train/Valid/Test Splits.

        Divide data for train, valid, test set. Training is used to train model parameters, validation is used to set early stopping, and test acts as blackbox independent test set.

        Args:

            test_size (float): Fraction of sample to be used for valid and test set.

            LOO (int): Number of sequences to leave-out in Leave-One-Out Cross-Validation. For example, when set to 20, 20 sequences will be left out for the validation set and 20 samples will be left out for the test set.

            split_by_sample (int): In the case one wants to train the single sequence classifer but not to mix the train/test sets with sequences from different samples, one can set this parameter to True to do the train/test splits by sample.

            combine_train_valid (bool): To combine the training and validation partitions into one which will be used for training and updating the model parameters, set this to True. This will also set the validation partition to the test partition. In other words, new train set becomes (original train + original valid) and then new valid = original test partition, new test = original test partition. Therefore, if setting this parameter to True, change one of the training parameters to set the stop training criterion (i.e. train_loss_min) to stop training based on the train set. If one does not chanage the stop training criterion, the decision of when to stop training will be based on the test data (which is considered a form of over-fitting).

        """
        Vars = [self.X_Seq_alpha,self.X_Seq_beta,self.alpha_sequences,self.beta_sequences,self.sample_id,self.class_id,self.seq_index,
                self.v_beta_num,self.d_beta_num,self.j_beta_num,self.v_alpha_num,self.j_alpha_num,
                self.v_beta,self.d_beta,self.j_beta,self.v_alpha,self.j_alpha,self.hla_data_seq_num,self.X_Seq_epitope]

        var_names = ['X_Seq_alpha','X_Seq_beta','alpha_sequences','beta_sequences','sample_id','class_id','seq_index',
                     'v_beta_num','d_beta_num','j_beta_num','v_alpha_num','j_alpha_num','v_beta','d_beta','j_beta',
                     'v_alpha','j_alpha','hla_data_seq_num','X_Seq_epitope']

        self.var_dict = dict(zip(var_names,list(range(len(var_names)))))

        if split_by_sample is False:
            self.train,self.valid,self.test = Get_Train_Valid_Test(Vars=Vars,Y=self.Y,test_size=test_size,regression=self.regression,LOO=LOO)

        else:
            sample = np.unique(self.sample_id)
            Y = np.asarray([self.Y[np.where(self.sample_id == x)[0][0]] for x in sample])
            train, valid, test = Get_Train_Valid_Test([sample], Y, test_size=test_size,regression=self.regression,LOO=LOO)

            self.train_idx = np.where(np.isin(self.sample_id, train[0]))[0]
            self.valid_idx = np.where(np.isin(self.sample_id, valid[0]))[0]
            self.test_idx = np.where(np.isin(self.sample_id, test[0]))[0]

            Vars.append(self.Y)

            self.train = [x[self.train_idx] for x in Vars]
            self.valid = [x[self.valid_idx] for x in Vars]
            self.test = [x[self.test_idx] for x in Vars]

        if combine_train_valid:
            for i in range(len(self.train)):
                self.train[i] = np.concatenate((self.train[i],self.valid[i]),axis=0)
                self.valid[i] = self.test[i]

        if (self.valid[0].size == 0) or (self.test[0].size == 0):
            raise Exception('Choose different train/valid/test parameters!')

    def _reset_models(self):
        self.models_dir = os.path.join(self.Name,'models')
        if os.path.exists(self.models_dir):
            shutil.rmtree(self.models_dir)
        os.makedirs(self.models_dir)

    def _build(self,
               units_tcr = [32,64,128],units_epitope = [32,64,128],units_hla = [32,64,128],
               kernel_tcr = [5,5,5],kernel_epitope=[5,5,5],kernel_hla=[30,5,5],
               stride_tcr=[1,1,1],stride_epitope=[1,1,1],stride_hla=[15,1,1],
               padding_tcr='same',padding_epitope='same',padding_hla='same',
               trainable_embedding = True,embedding_dim_aa = 64, embedding_dim_genes = 48, embedding_dim_hla = 12,
               num_fc_layers = 0, units_fc = 12,weight_by_class = False, class_weights = None,
               graph_seed = None,
               drop_out_rate=0.0,multisample_dropout = False, multisample_dropout_rate = 0.50, multisample_dropout_num_masks = 64,
               batch_size = 1000, epochs_min = 10, stop_criterion = 0.001, stop_criterion_window = 10,
               accuracy_min = None, train_loss_min = None, hinge_loss_t = 0.0, convergence = 'validation', learning_rate = 0.001, suppress_output = False):


        graph_model = tf.Graph()
        GO = graph_object()
        GO.on_graph_clustering=False
        GO.embedding_dim_genes = embedding_dim_genes
        GO.embedding_dim_aa = embedding_dim_aa
        GO.embedding_dim_hla = embedding_dim_hla
        GO.l2_reg = 0.0
        train_params = graph_object()
        train_params.batch_size = batch_size
        train_params.epochs_min = epochs_min
        train_params.stop_criterion = stop_criterion
        train_params.stop_criterion_window  = stop_criterion_window
        train_params.accuracy_min = accuracy_min
        train_params.train_loss_min = train_loss_min
        train_params.convergence = convergence
        train_params.suppress_output = suppress_output
        train_params.drop_out_rate = drop_out_rate
        train_params.multisample_dropout_rate = multisample_dropout_rate

        with graph_model.device(self.device):
            with graph_model.as_default():
                if graph_seed is not None:
                    tf.compat.v1.set_random_seed(graph_seed)

                GO.net = 'sup'
                GO.Features = Conv_Model(GO,self,trainable_embedding,
                                         units_tcr, units_epitope, units_hla,
                                         kernel_tcr,kernel_epitope,kernel_hla,
                                         stride_tcr, stride_epitope, stride_hla,
                                         padding_tcr,padding_epitope,padding_hla,
                                         num_fc_layers,units_fc)
                if self.regression is False:
                    GO.Y = tf.compat.v1.placeholder(tf.float64, shape=[None, self.Y.shape[1]])
                else:
                    GO.Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

                if self.regression is False:
                    if multisample_dropout:
                        GO.logits = MultiSample_Dropout(GO.Features,
                                                        num_masks=multisample_dropout_num_masks,
                                                        units=self.Y.shape[1],
                                                        activation=None,
                                                        rate=GO.prob_multisample)
                    else:
                        GO.logits = tf.compat.v1.layers.dense(GO.Features, self.Y.shape[1])

                    per_sample_loss = tf.nn.softmax_cross_entropy_with_logits(labels=GO.Y, logits=GO.logits)
                    per_sample_loss = per_sample_loss - hinge_loss_t
                    per_sample_loss = tf.cast((per_sample_loss > 0), tf.float32) * per_sample_loss
                    if weight_by_class is True:
                        class_weights = tf.constant([(1 / (np.sum(self.Y, 0) / np.sum(self.Y))).tolist()])
                        weights = tf.squeeze(tf.matmul(tf.cast(GO.Y, dtype='float32'), class_weights, transpose_b=True),axis=1)
                        GO.loss = tf.reduce_mean(input_tensor=weights * per_sample_loss)
                    elif class_weights is not None:
                        weights = np.zeros([1, len(self.lb.classes_)]).astype(np.float32)
                        for key in class_weights:
                            weights[:, self.lb.transform([key])[0]] = class_weights[key]
                        class_weights = tf.constant(weights)
                        weights = tf.squeeze(tf.matmul(tf.cast(GO.Y, dtype='float32'), class_weights, transpose_b=True),axis=1)
                        GO.loss = tf.reduce_mean(input_tensor=weights * per_sample_loss)
                    else:
                        GO.loss = tf.reduce_mean(input_tensor=per_sample_loss)

                else:
                    if multisample_dropout:
                        GO.logits = MultiSample_Dropout(GO.Features,
                                                        num_masks=multisample_dropout_num_masks,
                                                        units=1,
                                                        activation=None,
                                                        rate=GO.prob_multisample)
                    else:
                        GO.logits = tf.compat.v1.layers.dense(GO.Features, 1)

                    GO.loss = tf.reduce_mean(input_tensor=tf.square(GO.Y-GO.logits))

                GO.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(GO.loss)

                if self.regression is False:
                    with tf.compat.v1.name_scope('Accuracy_Measurements'):
                        GO.predicted = tf.nn.softmax(GO.logits, name='predicted')
                        correct_pred = tf.equal(tf.argmax(input=GO.predicted, axis=1), tf.argmax(input=GO.Y, axis=1))
                        GO.accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_pred, tf.float32), name='accuracy')
                else:
                    GO.predicted = GO.logits
                    GO.accuracy = GO.loss

                GO.saver = tf.compat.v1.train.Saver(max_to_keep=None)

                self.GO = GO
                self.train_params = train_params
                self.graph_model = graph_model
                self.kernel_tcr = kernel_tcr
                self.kernel_epitope = kernel_epitope
                self.kernel_hla = kernel_hla

    def _train(self,batch_seed=None,iteration=0):

        GO = self.GO
        graph_model = self.graph_model
        train_params = self.train_params

        batch_size = train_params.batch_size
        epochs_min = train_params.epochs_min
        stop_criterion = train_params.stop_criterion
        stop_criterion_window = train_params.stop_criterion_window
        accuracy_min = train_params.accuracy_min
        train_loss_min = train_params.train_loss_min
        convergence = train_params.convergence
        suppress_output = train_params.suppress_output
        drop_out_rate = train_params.drop_out_rate
        multisample_dropout_rate = train_params.multisample_dropout_rate


        #Initialize Training
        tf.compat.v1.reset_default_graph()
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(graph=graph_model,config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            val_loss_total = []
            train_accuracy_total = []
            train_loss_total = []
            stop_check_list = []
            e = 0

            print('')
            while True:
                if batch_seed is not None:
                    np.random.seed(batch_seed)
                train_loss, train_accuracy, train_predicted,train_auc = \
                    Run_Graph_SS(self.train,sess,self,GO,batch_size,random=True,train=True,drop_out_rate=drop_out_rate,multisample_dropout_rate=multisample_dropout_rate)

                train_accuracy_total.append(train_accuracy)
                train_loss_total.append(train_loss)

                valid_loss, valid_accuracy, valid_predicted,valid_auc = \
                    Run_Graph_SS(self.valid,sess,self,GO,batch_size,random=False,train=False)

                val_loss_total.append(valid_loss)

                test_loss, test_accuracy, test_predicted,test_auc = \
                    Run_Graph_SS(self.test,sess,self,GO,batch_size,random=False,train=False)
                self.y_pred = test_predicted
                self.y_test = self.test[-1]


                if suppress_output is False:
                    print("Training_Statistics: \n",
                          "Epoch: {}".format(e + 1),
                          "Training loss: {:.5f}".format(train_loss),
                          "Validation loss: {:.5f}".format(valid_loss),
                          "Testing loss: {:.5f}".format(test_loss),
                          "Training Accuracy: {:.5}".format(train_accuracy),
                          "Validation Accuracy: {:.5}".format(valid_accuracy),
                          "Testing AUC: {:.5}".format(test_auc))

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if e > epochs_min:
                        if accuracy_min is not None:
                            if np.mean(train_accuracy_total[-3:]) >= accuracy_min:
                                break
                        elif train_loss_min is not None:
                            if np.mean(train_loss_total[-3:]) < train_loss_min:
                                break
                        elif convergence == 'validation':
                            if val_loss_total:
                                stop_check_list.append(stop_check(val_loss_total, stop_criterion, stop_criterion_window))
                                if np.sum(stop_check_list[-3:]) >= 3:
                                    break

                        elif convergence == 'training':
                            if train_loss_total:
                                stop_check_list.append(stop_check(train_loss_total, stop_criterion, stop_criterion_window))
                                if np.sum(stop_check_list[-3:]) >= 3:
                                    break

                e += 1

            train_loss, train_accuracy, train_predicted, train_auc = \
                Run_Graph_SS(self.train, sess, self, GO, batch_size, random=False, train=False)

            self.test_pred.train.y_test.append(self.train[-1])
            self.test_pred.train.y_pred.append(train_predicted)

            valid_loss, valid_accuracy, valid_predicted, valid_auc = \
                Run_Graph_SS(self.valid, sess, self, GO, batch_size, random=False, train=False)

            self.test_pred.valid.y_test.append(self.valid[-1])
            self.test_pred.valid.y_pred.append(valid_predicted)

            test_loss, test_accuracy, test_predicted, test_auc = \
                Run_Graph_SS(self.test, sess, self, GO, batch_size, random=False, train=False)

            self.test_pred.test.y_test.append(self.test[-1])
            self.test_pred.test.y_pred.append(test_predicted)

            # Get_Seq_Features_Indices(self,batch_size,GO,sess)
            # self.features = Get_Latent_Features(self,batch_size,GO,sess)

            idx_base = np.asarray(range(len(self.sample_id)))
            self.train_idx = np.isin(idx_base,self.train[self.var_dict['seq_index']])
            self.valid_idx = np.isin(idx_base,self.valid[self.var_dict['seq_index']])
            self.test_idx = np.isin(idx_base,self.test[self.var_dict['seq_index']])

            if hasattr(self,'predicted'):
                self.predicted[self.test[self.var_dict['seq_index']]] += self.y_pred

            #
            # if self.use_alpha is True:
            #     var_save = [self.alpha_features,self.alpha_indices,self.alpha_sequences]
            #     with open(os.path.join(self.Name, 'alpha_features.pkl'), 'wb') as f:
            #         pickle.dump(var_save, f)
            #
            # if self.use_beta is True:
            #     var_save = [self.beta_features,self.beta_indices,self.beta_sequences]
            #     with open(os.path.join(self.Name, 'beta_features.pkl'), 'wb') as f:
            #         pickle.dump(var_save, f)

            # with open(os.path.join(self.Name, 'kernel.pkl'), 'wb') as f:
            #     pickle.dump(self.kernel, f)

            print('Done Training')
            # save model data and information for inference engine
            save_model_data(self, GO.saver, sess, name='SS', get=GO.predicted,iteration=iteration)

    def Train(self,
              units_tcr = [32,64,128],units_epitope = [32,64,128],units_hla = [32,64,128],
              kernel_tcr = [5,5,5], kernel_epitope = [5,5,5],kernel_hla = [30,5,5],
              stride_tcr=[1, 1, 1], stride_epitope=[1, 1, 1], stride_hla=[15, 1, 1],
              padding_tcr='same', padding_epitope='same', padding_hla='same',
              trainable_embedding = True,embedding_dim_aa = 64, embedding_dim_genes = 48, embedding_dim_hla = 12,
               num_fc_layers = 0, units_fc = 12,weight_by_class = False, class_weights = None,
               graph_seed = None,
               drop_out_rate=0.0,multisample_dropout = False, multisample_dropout_rate = 0.50, multisample_dropout_num_masks = 64,
               batch_size = 1000, epochs_min = 10, stop_criterion = 0.001, stop_criterion_window = 10,
               accuracy_min = None, train_loss_min = None, hinge_loss_t = 0.0, convergence = 'validation', learning_rate = 0.001, suppress_output = False,
                batch_seed = None):
        """
        # Train Single-Sequence Classifier

        This method trains the network and saves features values at the end of training for downstream analysis.

        The method also saves the per sequence predictions at the end of training in the variable self.predicted

        The multiesample parameters are used to implement Multi-Sample Dropout at the final layer of the model as described in "Multi-Sample Dropout for Accelerated Training and Better Generalization" https://arxiv.org/abs/1905.09788. This method has been shown to improve generalization of deep neural networks as well as improve convergence.

        Args:

            kernel (int): Size of convolutional kernel for first layer of convolutions.

            trainable_embedding (bool): Toggle to control whether a trainable embedding layer is used or native one-hot representation for convolutional layers.

            embedding_dim_aa (int): Learned latent dimensionality of amino-acids.

            embedding_dim_genes (int): Learned latent dimensionality of VDJ genes

            embedding_dim_hla (int): Learned latent dimensionality of HLA

            num_fc_layers (int): Number of fully connected layers following convolutional layer.

            units_fc (int): Number of nodes per fully-connected layers following convolutional layer.

            weight_by_class (bool): Option to weight loss by the inverse of the class frequency. Useful for unbalanced classes.

            class_weights (dict): In order to specify custom weights for each class during training, one can provide a dictionary with these weights. i.e. {'A':1.0,'B':2.0'}

            use_only_seq (bool): To only use sequence feaures, set to True. This will turn off features learned from gene usage.

            use_only_gene (bool): To only use gene-usage features, set to True. This will turn off features from the sequences.

            use_only_hla (bool): To only use hla feaures, set to True.

            size_of_net (list or str): The convolutional layers of this network have 3 layers for which the use can modify the number of neurons per layer. The user can either specify the size of the network with the following options:

                - small == [12,32,64] neurons for the 3 respective layers
                - medium == [32,64,128] neurons for the 3 respective layers
                - large == [64,128,256] neurons for the 3 respective layers
                - custom, where the user supplies a list with the number of nuerons for the respective layers
                    i.e. [3,3,3] would have 3 neurons for all 3 layers.
                    One can also adjust the number of layers for the convolutional stack by changing the length of
                    this list. [3,3,3] = 3 layers, [3,3,3,3] = 4 layers.

            graph_seed (int): For deterministic initialization of weights of the graph, set this to value of choice.

            drop_out_rate (float): drop out rate for fully connected layers

            multisample_dropout (bool): Set this parameter to True to implement this method.

            multisample_dropout_rate (float): The dropout rate for this multi-sample dropout layer.

            multisample_dropout_num_masks (int): The number of masks to sample from for the Multi-Sample Dropout layer.

            batch_size (int): Size of batch to be used for each training iteration of the net.

            epochs_min (int): Minimum number of epochs for training neural network.

            stop_criterion (float): Minimum percent decrease in determined interval (below) to continue training. Used as early stopping criterion.

            stop_criterion_window (int): The window of data to apply the stopping criterion.

            accuracy_min (loat): Optional parameter to allow alternative training strategy until minimum training accuracy is achieved, at which point, training ceases.

            train_loss_min (float): Optional parameter to allow alternative training strategy until minimum training loss is achieved, at which point, training ceases.

            hinge_loss_t (float): The per sequence loss minimum at which the loss of that sequence is not used to penalize the model anymore. In other words, once a per sequence loss has hit this value, it gets set to 0.0.

            convergence (str): This parameter determines which loss to assess the convergence criteria on. Options are 'validation' or 'training'. This is useful in the case one wants to change the convergence criteria on the training data when the training and validation partitions have been combined and used to training the model.

            learning_rate (float): The learning rate for training the neural network. Making this value larger will increase the rate of convergence but can introduce instability into training. For most, altering this value will not be necessary.

            suppress_output (bool): To suppress command line output with training statisitcs, set to True.

            batch_seed (int): For deterministic batching during training, set this value to an integer of choice.

        """
        self._reset_models()
        self.test_pred = make_test_pred_object()
        self._build(units_tcr,units_epitope,units_hla,
                    kernel_tcr,kernel_epitope,kernel_hla,
                    stride_tcr, stride_epitope, stride_hla,
                    padding_tcr, padding_epitope, padding_hla,
                    trainable_embedding,embedding_dim_aa, embedding_dim_genes, embedding_dim_hla,
               num_fc_layers, units_fc,weight_by_class, class_weights,
               graph_seed,
               drop_out_rate,multisample_dropout, multisample_dropout_rate, multisample_dropout_num_masks,
               batch_size, epochs_min, stop_criterion, stop_criterion_window,
               accuracy_min, train_loss_min, hinge_loss_t, convergence, learning_rate, suppress_output)
        self._train(batch_seed=batch_seed,iteration=0)
        for set in ['train', 'valid', 'test']:
            self.test_pred.__dict__[set].y_test = np.vstack(self.test_pred.__dict__[set].y_test)
            self.test_pred.__dict__[set].y_pred = np.vstack(self.test_pred.__dict__[set].y_pred)

    def Monte_Carlo_CrossVal(self,folds=5,test_size=0.25,LOO=None,split_by_sample=False,combine_train_valid=False,seeds=None,
                             units_tcr=[32, 64, 128], units_epitope=[32, 64, 128], units_hla=[32, 64, 128],
                             kernel_tcr=[5,5,5],kernel_epitope=[5,5,5],kernel_hla=[10,10,10],
                             stride_tcr=[1, 1, 1], stride_epitope=[1, 1, 1], stride_hla=[1, 5, 5],
                             padding_tcr='same', padding_epitope='same', padding_hla='same',
                             trainable_embedding=True, embedding_dim_aa=64, embedding_dim_genes=48, embedding_dim_hla=12,
                             num_fc_layers=0, units_fc=12, weight_by_class=False, class_weights=None,
                             graph_seed=None,
                             drop_out_rate=0.0, multisample_dropout=False, multisample_dropout_rate=0.50, multisample_dropout_num_masks=64,
                             batch_size=1000, epochs_min=10, stop_criterion=0.001, stop_criterion_window=10,
                             accuracy_min=None, train_loss_min=None, hinge_loss_t=0.0, convergence='validation', learning_rate=0.001, suppress_output=False,
                             batch_seed=None):

        '''
        # Monte Carlo Cross-Validation for Single-Sequence Classifier

        If the number of sequences is small but training the single-sequence classifier, one can use Monte Carlo Cross Validation to train a number of iterations before assessing predictive performance.After this method is run, the AUC_Curve method can be run to assess the overall performance.

        The method also saves the per sequence predictions at the end of training in the variable self.predicted. These per sequenes predictions are only assessed when the sequences are in the test set. Ideally, after running the classifier with multiple folds, each sequencce will have multiple predicttions that were collected when they were in the test set.

        The multisample parameters are used to implement Multi-Sample Dropout at the final layer of the model as described in "Multi-Sample Dropout for Accelerated Training and Better Generalization" https://arxiv.org/abs/1905.09788. This method has been shown to improve generalization of deep neural networks as well as inmprove convergence.

        Args:

            folds (int): Number of iterations for Cross-Validation

            test_size (float): Fraction of sample to be used for valid and test set.

            LOO (int): Number of sequences to leave-out in Leave-One-Out Cross-Validation. For example, when set to 20, 20 sequences will be left out for the validation set and 20 samples will be left out for the test set.

            split_by_sample (int): In the case one wants to train the single sequence classifer but not to mix the train/test sets with sequences from different samples, one can set this parameter to True to do the train/test splits by sample.

            combine_train_valid (bool): To combine the training and validation partitions into one which will be used for training and updating the model parameters, set this to True. This will also set the validation partition to the test partition. In other words, new train set becomes (original train + original valid) and then new valid = original test partition, new test = original test partition. Therefore, if setting this parameter to True, change one of the training parameters to set the stop training criterion (i.e. train_loss_min) to stop training based on the train set. If one does not change the stop training criterion, the decision of when to stop training will be based on the test data (which is considered a form of over-fitting).

            seeds (nd.array): In order to set a deterministic train/test split over the Monte-Carlo Simulations, one can provide an array of seeds for each MC simulation. This will result in the same train/test split over the N MC simulations. This parameter, if provided, should have the same size of the value of folds.

            kernel (int): Size of convolutional kernel for first layer of convolutions.

            trainable_embedding (bool): Toggle to control whether a trainable embedding layer is used or native one-hot representation for convolutional layers.

            embedding_dim_aa (int): Learned latent dimensionality of amino-acids.

            embedding_dim_genes (int): Learned latent dimensionality of VDJ genes

            embedding_dim_hla (int): Learned latent dimensionality of HLA

            num_fc_layers (int): Number of fully connected layers following convolutional layer.

            units_fc (int): Number of nodes per fully-connected layers following convolutional layer.

            weight_by_class (bool): Option to weight loss by the inverse of the class frequency. Useful for unbalanced classes.

            class_weights (dict): In order to specify custom weights for each class during training, one can provide a dictionary with these weights. i.e. {'A':1.0,'B':2.0'}

            use_only_seq (bool): To only use sequence feaures, set to True. This will turn off features learned from gene usage.

            use_only_gene (bool): To only use gene-usage features, set to True. This will turn off features from the sequences.

            use_only_hla (bool): To only use hla feaures, set to True.

            size_of_net (list or str): The convolutional layers of this network have 3 layers for which the use can modify the number of neurons per layer. The user can either specify the size of the network with the following options:

                - small == [12,32,64] neurons for the 3 respective layers
                - medium == [32,64,128] neurons for the 3 respective layers
                - large == [64,128,256] neurons for the 3 respective layers
                - custom, where the user supplies a list with the number of nuerons for the respective layers
                    i.e. [3,3,3] would have 3 neurons for all 3 layers.
                    One can also adjust the number of layers for the convolutional stack by changing the length of
                    this list. [3,3,3] = 3 layers, [3,3,3,3] = 4 layers.

            graph_seed (int): For deterministic initialization of weights of the graph, set this to value of choice.

            drop_out_rate (float): drop out rate for fully connected layers

            multisample_dropout (bool): Set this parameter to True to implement this method.

            multisample_dropout_rate (float): The dropout rate for this multi-sample dropout layer.

            multisample_dropout_num_masks (int): The number of masks to sample from for the Multi-Sample Dropout layer.

            batch_size (int): Size of batch to be used for each training iteration of the net.

            epochs_min (int): Minimum number of epochs for training neural network.

            stop_criterion (float): Minimum percent decrease in determined interval (below) to continue training. Used as early stopping criterion.

            stop_criterion_window (int): The window of data to apply the stopping criterion.

            accuracy_min (float): Optional parameter to allow alternative training strategy until minimum training accuracy is achieved, at which point, training ceases.

            train_loss_min (float): Optional parameter to allow alternative training strategy until minimum training loss is achieved, at which point, training ceases.

            hinge_loss_t (float): The per sequence loss minimum at which the loss of that sequence is not used to penalize the model anymore. In other words, once a per sequence loss has hit this value, it gets set to 0.0.

            convergence (str): This parameter determines which loss to assess the convergence criteria on. Options are 'validation' or 'training'. This is useful in the case one wants to change the convergence criteria on the training data when the training and validation partitions have been combined and used to training the model.

            learning_rate (float): The learning rate for training the neural network. Making this value larger will increase the rate of convergence but can introduce instability into training. For most, altering this value will not be necessary.

            suppress_output (bool): To suppress command line output with training statisitcs, set to True.

            batch_seed (int): For deterministic batching during training, set this value to an integer of choice.

        '''

        y_pred = []
        y_test = []
        self.test_pred = make_test_pred_object()
        predicted = np.zeros_like(self.predicted)
        counts = np.zeros_like(self.predicted)
        self._reset_models()
        self._build(units_tcr,units_epitope,units_hla,
                    kernel_tcr,kernel_epitope,kernel_hla,
                    stride_tcr, stride_epitope, stride_hla,
                    padding_tcr, padding_epitope, padding_hla,
                    trainable_embedding,embedding_dim_aa, embedding_dim_genes, embedding_dim_hla,
               num_fc_layers, units_fc,weight_by_class, class_weights,
               graph_seed,
               drop_out_rate,multisample_dropout, multisample_dropout_rate, multisample_dropout_num_masks,
               batch_size, epochs_min, stop_criterion, stop_criterion_window,
               accuracy_min, train_loss_min, hinge_loss_t, convergence, learning_rate, suppress_output)

        for i in tqdm(range(0, folds)):
            if suppress_output is False:
                print('Fold '+str(i))
            if seeds is not None:
                np.random.seed(seeds[i])
            self.Get_Train_Valid_Test(test_size=test_size, LOO=LOO,split_by_sample=split_by_sample,combine_train_valid=combine_train_valid)
            self._train(batch_seed=batch_seed,iteration=i)

            y_test.append(self.y_test)
            y_pred.append(self.y_pred)

            predicted[self.test[self.var_dict['seq_index']]] += self.y_pred
            counts[self.test[self.var_dict['seq_index']]] += 1

            if self.regression is False:
                if suppress_output is False:
                    print_performance_epoch(self)
            print('')

        self.y_test = np.vstack(y_test)
        self.y_pred = np.vstack(y_pred)
        self.predicted = np.divide(predicted,counts, out = np.zeros_like(predicted), where = counts != 0)

        for set in ['train', 'valid', 'test']:
            self.test_pred.__dict__[set].y_test = np.vstack(self.test_pred.__dict__[set].y_test)
            self.test_pred.__dict__[set].y_pred = np.vstack(self.test_pred.__dict__[set].y_pred)

        print('Monte Carlo Simulation Completed')

    def K_Fold_CrossVal(self,folds=None,split_by_sample=False,combine_train_valid=False,seeds=None,
                        units_tcr=[32, 64, 128], units_epitope=[32, 64, 128], units_hla=[32, 64, 128],
                        kernel_tcr=5,kernel_epitope=5,kernel_hla=30,
                        stride_tcr=[1, 1, 1], stride_epitope=[1, 1, 1], stride_hla=[15, 1, 1],
                        padding_tcr='same', padding_epitope='same', padding_hla='same',
                        trainable_embedding=True, embedding_dim_aa=64, embedding_dim_genes=48, embedding_dim_hla=12,
                        num_fc_layers=0, units_fc=12, weight_by_class=False, class_weights=None,
                        graph_seed=None,
                        drop_out_rate=0.0, multisample_dropout=False, multisample_dropout_rate=0.50, multisample_dropout_num_masks=64,
                        batch_size=1000, epochs_min=10, stop_criterion=0.001, stop_criterion_window=10,
                        accuracy_min=None, train_loss_min=None, hinge_loss_t=0.0, convergence='validation', learning_rate=0.001, suppress_output=False,
                        batch_seed=None):

        '''
        # K_Fold Cross-Validation for Single-Sequence Classifier

        If the number of sequences is small but training the single-sequence classifier, one
        can use K_Fold Cross Validation to train on all but one before assessing
        predictive performance.After this method is run, the AUC_Curve method can be run to
        assess the overall performance.

        The method also saves the per sequence predictions at the end of training in the variable self.predicted. These per sequenes predictions are only assessed when the sequences are in the test set.

        The multisample parameters are used to implement Multi-Sample Dropout at the final layer of the model as described in "Multi-Sample Dropout for Accelerated Training and Better Generalization" https://arxiv.org/abs/1905.09788. This method has been shown to improve generalization of deep neural networks as well as inmprove convergence.

        Args:

            folds (int): Number of Folds

            split_by_sample (int): In the case one wants to train the single sequence classifer but not to mix the train/test sets with sequences from different samples, one can set this parameter to True to do the train/test splits by sample.

            combine_train_valid (bool): To combine the training and validation partitions into one which will be used for training and updating the model parameters, set this to True. This will also set the validation partition to the test partition. In other words, new train set becomes (original train + original valid) and then new valid = original test partition, new test = original test partition. Therefore, if setting this parameter to True, change one of the training parameters to set the stop training criterion (i.e. train_loss_min) to stop training based on the train set. If one does not chanage the stop training criterion, the decision of when to stop training will be based on the test data (which is considered a form of over-fitting).

            seeds (nd.array): In order to set a deterministic train/test split over the K-Fold Simulations, one can provide an array of seeds for each K-fold simulation. This will result in the same train/test split over the N Fold simulations. This parameter, if provided, should have the same size of the value of folds.

            kernel (int): Size of convolutional kernel for first layer of convolutions.

            trainable_embedding (bool): Toggle to control whether a trainable embedding layer is used or native one-hot representation for convolutional layers.

            embedding_dim_aa (int): Learned latent dimensionality of amino-acids.

            embedding_dim_genes (int): Learned latent dimensionality of VDJ genes

            embedding_dim_hla (int): Learned latent dimensionality of HLA

            num_fc_layers (int): Number of fully connected layers following convolutional layer.

            units_fc (int): Number of nodes per fully-connected layers following convolutional layer.

            weight_by_class (bool): Option to weight loss by the inverse of the class frequency. Useful for unbalanced classes.

            class_weights (dict): In order to specify custom weights for each class during training, one can provide a dictionary with these weights. i.e. {'A':1.0,'B':2.0'}

            use_only_seq (bool): To only use sequence feaures, set to True. This will turn off features learned from gene usage.

            use_only_gene (bool): To only use gene-usage features, set to True. This will turn off features from the sequences.

            use_only_hla (bool): To only use hla feaures, set to True.

            size_of_net (list or str): The convolutional layers of this network have 3 layers for which the use can modify the number of neurons per layer. The user can either specify the size of the network with the following options:

                - small == [12,32,64] neurons for the 3 respective layers
                - medium == [32,64,128] neurons for the 3 respective layers
                - large == [64,128,256] neurons for the 3 respective layers
                - custom, where the user supplies a list with the number of nuerons for the respective layers
                    i.e. [3,3,3] would have 3 neurons for all 3 layers.
                    One can also adjust the number of layers for the convolutional stack by changing the length of
                    this list. [3,3,3] = 3 layers, [3,3,3,3] = 4 layers.

            graph_seed (int): For deterministic initialization of weights of the graph, set this to value of choice.

            drop_out_rate (float): drop out rate for fully connected layers

            multisample_dropout (bool):
                Set this parameter to True to implement this method.

            multisample_dropout_rate (float): The dropout rate for this multi-sample dropout layer.

            multisample_dropout_num_masks (int): The number of masks to sample from for the Multi-Sample Dropout layer.

            batch_size (int): Size of batch to be used for each training iteration of the net.

            epochs_min (int): Minimum number of epochs for training neural network.

            stop_criterion (float): Minimum percent decrease in determined interval (below) to continue training. Used as early stopping criterion.

            stop_criterion_window (int): The window of data to apply the stopping criterion.

            accuracy_min (float): Optional parameter to allow alternative training strategy until minimum training accuracy is achieved, at which point, training ceases.

            train_loss_min (float): Optional parameter to allow alternative training strategy until minimum training loss is achieved, at which point, training ceases.

            hinge_loss_t (float): The per sequence loss minimum at which the loss of that sequence is not used to penalize the model anymore. In other words, once a per sequence loss has hit this value, it gets set to 0.0.

            convergence (str): This parameter determines which loss to assess the convergence criteria on. Options are 'validation' or 'training'. This is useful in the case one wants to change the convergence criteria on the training data when the training and validation partitions have been combined and used to training the model.

            learning_rate (float): The learning rate for training the neural network. Making this value larger will increase the rate of convergence but can introduce instability into training. For most, altering this value will not be necessary.

            suppress_output (bool): To suppress command line output with training statisitcs, set to True.

            batch_seed (int): For deterministic batching during training, set this value to an integer of choice.

        '''

        #Create Folds
        if split_by_sample is False:
            if folds is None:
                folds = len(self.Y)

            idx = list(range(len(self.Y)))
            idx_left = idx
            file_per_sample = len(self.Y) // folds
            test_idx = []
            for ii in range(folds):
                if seeds is not None:
                    np.random.seed(seeds[ii])
                if ii != folds-1:
                    idx_sel = np.random.choice(idx_left, size=file_per_sample, replace=False)
                else:
                    idx_sel = idx_left

                test_idx.append(idx_sel)
                idx_left = np.setdiff1d(idx_left, idx_sel)
        else:
            if folds is None:
                folds = len(np.unique(self.sample_id))

            idx = np.unique(self.sample_id)
            idx_left = idx
            file_per_sample = len(np.unique(self.sample_id)) // folds
            test_idx = []
            for ii in tqdm(range(folds)):
                if seeds is not None:
                    np.random.seed(seeds[ii])
                if ii != folds-1:
                    idx_sel = np.random.choice(idx_left, size=file_per_sample, replace=False)
                    idx_sel_seq = np.where(np.isin(self.sample_id,idx_sel))[0]
                else:
                    idx_sel = idx_left
                    idx_sel_seq = np.where(np.isin(self.sample_id,idx_sel))[0]

                test_idx.append(idx_sel_seq)
                idx_left = np.setdiff1d(idx_left, idx_sel)

            idx = list(range(len(self.Y)))


        self._reset_models()
        self._build(units_tcr,units_epitope,units_hla,
                    kernel_tcr,kernel_epitope,kernel_hla,
                    stride_tcr, stride_epitope, stride_hla,
                    padding_tcr, padding_epitope, padding_hla,
                    trainable_embedding, embedding_dim_aa, embedding_dim_genes, embedding_dim_hla,
                    num_fc_layers, units_fc, weight_by_class, class_weights,
                    graph_seed,
                    drop_out_rate, multisample_dropout, multisample_dropout_rate, multisample_dropout_num_masks,
                    batch_size, epochs_min, stop_criterion, stop_criterion_window,
                    accuracy_min, train_loss_min, hinge_loss_t, convergence, learning_rate, suppress_output)

        y_test = []
        y_pred = []
        self.test_pred = make_test_pred_object()
        for ii in range(folds):
            if suppress_output is False:
                print('Fold '+str(ii))
            train_idx = np.setdiff1d(idx,test_idx[ii])
            valid_idx = np.random.choice(train_idx,len(train_idx)//(folds-1),replace=False)
            train_idx = np.setdiff1d(train_idx,valid_idx)

            Vars = [self.X_Seq_alpha, self.X_Seq_beta, self.alpha_sequences, self.beta_sequences, self.sample_id,
                    self.class_id, self.seq_index,
                    self.v_beta_num, self.d_beta_num, self.j_beta_num, self.v_alpha_num, self.j_alpha_num,
                    self.v_beta, self.d_beta, self.j_beta, self.v_alpha, self.j_alpha,self.hla_data_seq_num]

            var_names = ['X_Seq_alpha', 'X_Seq_beta', 'alpha_sequences', 'beta_sequences', 'sample_id', 'class_id',
                         'seq_index',
                         'v_beta_num', 'd_beta_num', 'j_beta_num', 'v_alpha_num', 'j_alpha_num', 'v_beta', 'd_beta',
                         'j_beta',
                         'v_alpha', 'j_alpha','hla_data_seq_num']

            self.var_dict = dict(zip(var_names, list(range(len(var_names)))))

            self.train,self.valid, self.test = Get_Train_Valid_Test_KFold(Vars=Vars,
                                                               train_idx=train_idx,
                                                               valid_idx = valid_idx,
                                                               test_idx = test_idx[ii],Y=self.Y)
            if combine_train_valid:
                for i in range(len(self.train)):
                    self.train[i] = np.concatenate((self.train[i], self.valid[i]), axis=0)
                    self.valid[i] = self.test[i]

            self.LOO = None
            self._train(batch_seed=batch_seed,iteration=ii)

            y_test.append(self.y_test)
            y_pred.append(self.y_pred)

            if self.regression is False:
                if suppress_output is False:
                    print_performance_epoch(self)
            print('')

        self.y_test = np.vstack(y_test)
        self.y_pred = np.vstack(y_pred)
        test_idx = np.hstack(test_idx)
        self.predicted = np.zeros_like(self.predicted)
        self.predicted[test_idx] = self.y_pred

        for set in ['train', 'valid', 'test']:
            self.test_pred.__dict__[set].y_test = np.vstack(self.test_pred.__dict__[set].y_test)
            self.test_pred.__dict__[set].y_pred = np.vstack(self.test_pred.__dict__[set].y_pred)

        print('K-fold Cross Validation Completed')

    def AUC_Curve(self,by=None,filename='AUC.tif',title=None,title_font=None,plot=True,diag_line=True,
                  xtick_size = None, ytick_size=None, xlabel_size = None, ylabel_size=None,
                  legend_font_size=None,frameon=True,legend_loc = 'lower right',
                  figsize=None,set='test',color_dict=None):
        """
        # AUC Curve for both Sequence and Repertoire/Sample Classifiers

        Args:

            by (str): To show AUC curve for only one class, set this parameter to the name of the class label one wants to plot.

            filename (str): Filename to save tif file of AUC curve.

            title (str): Optional Title to put on ROC Curve.

            title_font (int): Optional font size for title

            plot (bool): To suppress plotting and just save the data/figure, set to False.

            diag_line (bool): To plot the line/diagonal of y=x defining no predictive power, set to True. To remove from plot, set to False.

            xtick_size (float): Size of xticks

            ytick_size (float): Size of yticks

            xlabel_size (float): Size of xlabel

            ylabel_size (float): Size of ylabel

            legend_font_size (float): Size of legend

            frameon (bool): Whether to show frame around legend.

            figsize (tuple): To change the default size of the figure, set this to size of figure (i.e. - (10,10) )

            set (str): Which partition of the data to look at performance of model. Options are train/valid/test.

            color_dict (dict): An optional dictionary that maps classes to colors in the case user wants to define colors of lines on plot.

        Returns:
            AUC Data

            - self.AUC_DF (Pandas Dataframe):
            AUC scores are returned for each class.

            In addition to plotting the ROC Curve, the AUC's are saved to a csv file in the results directory called 'AUC.csv'

        """
        try:
            y_test = self.test_pred.__dict__[set].y_test
            y_pred = self.test_pred.__dict__[set].y_pred
        except:
            y_test = self.y_test
            y_pred = self.y_pred

        auc_scores = []
        classes = []
        if plot is False:
            plt.ioff()
        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            plt.figure()

        if diag_line:
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')


        if color_dict is None:
            RGB_tuples = distinctipy.get_colors(len(self.lb.classes_),rng=0)
            color_dict = dict(zip(self.lb.classes_, RGB_tuples))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        if by is None:
            for ii, class_name in enumerate(self.lb.classes_, 0):
                try:
                    roc_score = roc_auc_score(y_test[:, ii], y_pred[:,ii])
                    classes.append(class_name)
                    auc_scores.append(roc_score)
                    fpr, tpr, _ = roc_curve(y_test[:, ii], y_pred[:,ii])
                    plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (class_name, roc_score),c=color_dict[class_name])
                except:
                    continue
        else:
            class_name = by
            ii = self.lb.transform([by])[0]
            roc_score = roc_auc_score(y_test[:, ii], y_pred[:, ii])
            auc_scores.append(roc_score)
            classes.append(class_name)
            fpr, tpr, _ = roc_curve(y_test[:, ii], y_pred[:, ii])
            plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (class_name, roc_score),c=color_dict[class_name])

        plt.legend(loc=legend_loc,frameon=frameon)
        if legend_font_size is not None:
            plt.legend(prop={'size': legend_font_size},loc=legend_loc,frameon=frameon)

        if title is not None:
            if title_font is not None:
                plt.title(title,fontsize=title_font)
            else:
                plt.title(title)

        ax = plt.gca()

        if xlabel_size is not None:
            ax.xaxis.label.set_size(xlabel_size)

        if ylabel_size is not None:
            ax.yaxis.label.set_size(ylabel_size)

        if xtick_size is not None:
            plt.xticks(fontsize=xtick_size)

        if ytick_size is not None:
            plt.yticks(fontsize=ytick_size)

        plt.tight_layout()
        plt.savefig(os.path.join(self.directory_results,filename))
        if plot is True:
            plt.show(block=False)
        else:
            plt.close()

        df_out = pd.DataFrame()
        df_out['Class'] = classes
        df_out['AUC'] = auc_scores
        df_out.to_csv(os.path.join(self.directory_results,'AUC.csv'),index=False)
        self.AUC_DF = df_out

    def SRCC(self, s=10, kde=False, title=None):
        """
        # Spearman's Rank Correlation Coefficient Plot

        In the case one is doing a regression-based model for the sequence classiifer, one can plot the predicted vs actual labeled value with this method. The method returns a plot for the regression and a value of the correlation coefficient.

        Args:

            s (int): size of points for scatterplot

            kde (bool): To do a kernel density estimation per point and plot this as a color-scheme, set to True. Warning: this option will take longer to run.

            title (str): Title for the plot.

        Returns:
            SRCC Output

            - corr (float):
            Spearman's Rank Correlation Coefficient

            - ax (matplotlib axis):
                axis on which plot is drawn
        """
        x, y = np.squeeze(self.y_pred, -1), np.squeeze(self.y_test, -1)
        corr, _ = spearmanr(x, y)

        fig, ax = plt.subplots()
        if kde:
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            r = np.argsort(z)
            x, y, z = x[r], y[r], z[r]
            ax.scatter(x, y, s=s, c=z, cmap=plt.cm.jet)
        else:
            ax.scatter(x, y, s=s)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        if title is not None:
            plt.title(title)
        return corr, ax

    def Representative_Sequences(self,top_seq=10,motif_seq=5,make_seq_logos=True,
                                 color_scheme='weblogo_protein',logo_file_format='.eps'):
        """
        # Identify most highly predicted sequences for each class and corresponding motifs.

        This method allows the user to query which sequences were most predicted to belong to a given class along with the motifs that were learned for these representative sequences. Of note, this method only reports sequences that were in the test set so as not to return highly predicted sequences that were over-fit in the training set. To obtain the highest predicted sequences in all the data, run a K-fold cross-validation or Monte-Carlo cross-validation before running this method. In this way, the predicted probability will have been assigned to a sequence only when it was in the independent test set.

        In the case of a regression task, the representative sequences for the 'high' and 'low' values for the regression model are returned in the Rep_Seq Dict.

        This method will also determine motifs the network has learned that are highly associated with the label through multi-nomial linear regression and creates seq logos and fasta files in the results folder. Within a folder for a given class, the motifs are sorted by their linear coefficient. The coefficient is in the file name (i.e. 0_0.125_feature_2.eps reflects the the 0th highest feature with a coefficient of 0.125.

        Args:

            top_seq (int): The number of top sequences to show for each class.

            motif_seq (int): The number of sequences to use to generate each motif. The more sequences, the possibly more noisy the seq_logo will be.

            make_seq_logos (bool): In order to make seq logos for visualization of enriched motifs, set this to True. Whether this is set to True or not, the fast files that define enriched motifs will still be saved.

            color_scheme (str): color scheme to use for LogoMaker.
            ###
            options are:
                - weblogo_protein
                - skylign_protein
                - dmslogo_charge
                - dmslogo_funcgroup
                - hydrophobicity
                - chemistry
                - charge
                - NajafabadiEtAl2017

            logo_file_format (str):
                The type of image file one wants to save the seqlogo as. Default is vector-based format (.eps)

        Returns:
            Outputs

            - self.Rep_Seq (dictionary of dataframes):
            This dictionary of dataframes holds for each class the top sequences and their respective probabiltiies for all classes. These dataframes can also be found in the results folder under Rep_Sequences.

            - self.Rep_Seq_Features_(alpha/beta) (dataframe):
            This dataframe holds information for which features were associated by a multinomial linear model to the predicted probabilities of the neural network. The values in this dataframe are the linear model coefficients. This allows one to see which features were associated with the output of the trained neural network. These are also the same values that are on the motif seqlogo files in the results folder.

        Furthermore, the motifs are written in the results directory underneath the Motifs folder. To find the beta motifs for a given class, look under Motifs/beta/class_name/. These fasta/logo files are labeled by the linear coefficient of that given feature for that given class followed by the number name of the feature. These fasta files can then be visualized via weblogos at the following site: "https://weblogo.berkeley.edu/logo.cgi" or are present in the folder for direct visualization.


        """
        dir = 'Rep_Sequences'
        dir = os.path.join(self.directory_results, dir)
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        file_list = [f for f in os.listdir(dir)]
        [os.remove(os.path.join(dir, f)) for f in file_list]

        Rep_Seq = []
        keep = []
        df_temp = pd.DataFrame()
        df_temp['alpha'] = self.alpha_sequences
        df_temp['beta'] = self.beta_sequences
        df_temp['v_alpha'] = self.v_alpha
        df_temp['j_alpha'] = self.j_alpha
        df_temp['v_beta'] = self.v_beta
        df_temp['d_beta'] = self.d_beta
        df_temp['j_beta'] = self.j_beta
        df_temp['Class'] = self.class_id
        if self.regression is True:
            df_temp['Regressed_Val'] = self.Y
        df_temp['Sample'] = self.sample_id
        df_temp['Freq'] = self.freq
        if hasattr(self, 'counts'):
            df_temp['Counts'] = self.counts
        try:
            df_temp['HLA'] = list(map(list, self.hla_data_seq.tolist()))
        except:
            pass

        if self.regression is False:
            for ii, sample in enumerate(self.lb.classes_, 0):
                df_temp[sample] = self.predicted[:, ii]

            for ii, sample in enumerate(self.lb.classes_, 0):
                df_temp.sort_values(by=sample, ascending=False, inplace=True)
                df_sample = df_temp[df_temp['Class'] == sample][0:top_seq]

                if not df_sample.empty:
                    Rep_Seq.append(df_sample)
                    df_sample.to_csv(os.path.join(dir, sample + '.csv'), index=False)
                    keep.append(ii)

            self.Rep_Seq = dict(zip(self.lb.classes_[keep], Rep_Seq))

            if self.use_alpha:
                self.Req_Seq_Features_alpha = Motif_Features(self, self.alpha_features, self.alpha_indices,
                                                             self.alpha_sequences, self.directory_results,
                                                             'alpha', self.kernel, motif_seq,make_seq_logos,
                                                                 color_scheme,logo_file_format)

            if self.use_beta:
                self.Req_Seq_Features_beta = Motif_Features(self, self.beta_features, self.beta_indices,
                                                            self.beta_sequences, self.directory_results,
                                                            'beta', self.kernel, motif_seq,make_seq_logos,
                                                                color_scheme,logo_file_format)
        else:
            df_temp['Predicted'] = self.predicted
            df_temp.sort_values(by='Predicted',ascending=False,inplace=True)
            df_sample_top = df_temp[0:top_seq]
            df_temp.sort_values(by='Predicted',ascending=True,inplace=True)
            df_sample_bottom = df_temp[0:top_seq]
            labels = ['High','Low']
            Rep_Seq.append(df_sample_top)
            Rep_Seq.append(df_sample_bottom)
            df_sample_top.to_csv(os.path.join(dir,'high.csv'),index=False)
            df_sample_bottom.to_csv(os.path.join(dir,'low.csv'),index=False)

            self.Rep_Seq = dict(zip(labels,Rep_Seq))

            if self.use_alpha:
                self.Req_Seq_Features_alpha = Motif_Features_Reg(self, self.alpha_features, self.alpha_indices,
                                                             self.alpha_sequences, self.directory_results,
                                                             'alpha', self.kernel, motif_seq,make_seq_logos,
                                                                 color_scheme,logo_file_format)

            if self.use_beta:
                self.Req_Seq_Features_beta = Motif_Features_Reg(self, self.beta_features, self.beta_indices,
                                                            self.beta_sequences, self.directory_results,
                                                            'beta', self.kernel, motif_seq,make_seq_logos,
                                                                color_scheme,logo_file_format)

    def Sequence_Inference(self, alpha_sequences=None, beta_sequences=None, v_beta=None, d_beta=None, j_beta=None,
                  v_alpha=None, j_alpha=None, epitope_sequences=None, p=None,hla=None, batch_size=10000,models=None,return_dist=False):
        """
        # Predicting outputs of sequence models on new data

        This method allows a user to take a pre-trained autoencoder/sequence classifier and generate outputs from the model on new data. For the autoencoder, this returns the features from the latent space. For the sequence classifier, it is the probability of belonging to each class.

        In the case that multiple models have been trained via MC or K-fold Cross-Validation strategy for the sequence classifier, this method can use some or all trained models in an ensemble fashion to provide the average prediction per sequence as well as the distribution of predictions from all trained models.

        This method is included in the two sequence DeepTCR objects:

        - DeepTCR_U (unsupervised)
        - DeepTCR_SS (supervised sequence classifier/regressor)

        Args:

            alpha_sequences (ndarray of strings): A 1d array with the sequences for inference for the alpha chain.

            beta_sequences (ndarray of strings): A 1d array with the sequences for inference for the beta chain.

            v_beta (ndarray of strings): A 1d array with the v-beta genes for inference.

            d_beta (ndarray of strings): A 1d array with the d-beta genes for inference.

            j_beta (ndarray of strings): A 1d array with the j-beta genes for inference.

            v_alpha (ndarray of strings): A 1d array with the v-alpha genes for inference.

            j_alpha (ndarray of strings): A 1d array with the j-alpha genes for inference.

            hla (ndarray of tuples/arrays): To input the hla context for each sequence fed into DeepTCR, this will need to formatted as an ndarray that is (N,) for each sequence where each entry is a tuple/array of strings referring to the alleles seen for that sequence. ('A*01:01', 'A*11:01', 'B*35:01', 'B*35:02', 'C*04:01')

                - If the model used for inference was trained to use HLA-supertypes, one should still enter the HLA in the format it was provided to the original model (i.e. A0101). This mehthod will then convert those HLA alleles into the appropriaet supertype designation. The HLA alleles DO NOT need to be provided to this method in the supertype designation.

            p (multiprocessing pool object): a pre-formed pool object can be passed to method for multiprocessing tasks.

            batch_size (int): Batch size for inference.

            models (list): In the case of the supervised sequence classifier, if several models were trained (via MC or Kfold crossvals), one can specify which ones to use for inference. Otherwise, thie method uses all trained models found in Name/models/ in an ensemble fashion. The method will output of the average of all models as well as the distribution of outputs for the user.

            return_dist (bool): If the user wants to also return teh distribution of sequence predicionts over all models use dfor inference, one should set this value to True.

        Returns:
            features, features_dist

            - features (array), shape = [N, latent_dim]: An array that contains n x latent_dim containing features for all sequences. For the VAE, this represents the features from the latent space. For the sequence classifier, this represents the probabilities for every class or the regressed value from the sequence regressor. In the case of multiple models being used for inference in ensemble, this becomes the average prediction over all models.

            - features_dist (array), shape = [n_models,N,latent_dim]: An array that contains the output of all models separately for each input sequence. This output is useful if using multiple models in ensemble to predict on a new sequence. This output describes the distribution of the predictions over all models.

        """
        model_type,get = load_model_data(self)
        out, out_dist = inference_method_ss(get,alpha_sequences,beta_sequences,
                               v_beta,d_beta,j_beta,v_alpha,j_alpha,hla,
                                p,batch_size,self,models)

        if return_dist:
            return out, out_dist
        else:
            return out

    def _residue(self, alpha_sequence, beta_sequence, v_beta, d_beta, j_beta, v_alpha, j_alpha, hla,
                 p, batch_size, models, chain):

        if self.model_type == 'SS':
            inf_func = self.Sequence_Inference
        elif self.model_type == 'WF':
            inf_func = self.Sample_Inference

        df_alpha = pd.DataFrame()
        df_beta = pd.DataFrame()
        if chain == 'alpha':
            if alpha_sequence is not None:
                alpha_list, pos, ref, alt = make_seq_list(alpha_sequence, ref=list(self.aa_idx.keys()))
                len_list = len(alpha_list)

                if beta_sequence is None:
                    beta_sequences = None
                else:
                    beta_sequences = np.array([beta_sequence] * len_list)

                if v_beta is None:
                    v_beta = None
                else:
                    v_beta = np.array([v_beta] * len_list)

                if d_beta is None:
                    d_beta = None
                else:
                    d_beta = np.array([d_beta] * len_list)

                if j_beta is None:
                    j_beta = None
                else:
                    j_beta = np.array([j_beta] * len_list)

                if v_alpha is None:
                    v_alpha = None
                else:
                    v_alpha = np.array([v_alpha] * len_list)

                if j_alpha is None:
                    j_alpha = None
                else:
                    j_alpha = np.array([j_alpha] * len_list)

                if hla is None:
                    hla = None
                else:
                    hla = np.array([hla] * len_list)

                out = inf_func(beta_sequences=beta_sequences,
                               alpha_sequences=np.array(alpha_list),
                               v_beta=v_beta,
                               d_beta=d_beta,
                               j_beta=j_beta,
                               v_alpha=v_alpha,
                               j_alpha=j_alpha,
                               p=p,
                               hla=hla,
                               batch_size=batch_size,
                               models=models)

                df_alpha['alpha'] = alpha_list
                df_alpha['pos'] = pos
                df_alpha['ref'] = ref
                df_alpha['alt'] = alt
                if self.regression:
                    df_alpha['high'] = out[:, 0]
                else:
                    for ii in range(out.shape[1]):
                        df_alpha[self.lb.inverse_transform([ii])[0]] = out[:, ii]

        if chain == 'beta':
            if beta_sequence is not None:
                beta_list, pos, ref, alt = make_seq_list(beta_sequence, ref=list(self.aa_idx.keys()))
                len_list = len(beta_list)
                if alpha_sequence is None:
                    alpha_sequences = None
                else:
                    alpha_sequences = np.array([alpha_sequence] * len_list)

                if v_beta is None:
                    v_beta = None
                else:
                    v_beta = np.array([v_beta] * len_list)

                if d_beta is None:
                    d_beta = None
                else:
                    d_beta = np.array([d_beta] * len_list)

                if j_beta is None:
                    j_beta = None
                else:
                    j_beta = np.array([j_beta] * len_list)

                if v_alpha is None:
                    v_alpha = None
                else:
                    v_alpha = np.array([v_alpha] * len_list)

                if j_alpha is None:
                    j_alpha = None
                else:
                    j_alpha = np.array([j_alpha] * len_list)

                if hla is None:
                    hla = None
                else:
                    hla = np.array([hla] * len_list)

                out = inf_func(beta_sequences=np.array(beta_list),
                               alpha_sequences=alpha_sequences,
                               v_beta=v_beta,
                               d_beta=d_beta,
                               j_beta=j_beta,
                               v_alpha=v_alpha,
                               j_alpha=j_alpha,
                               p=p,
                               hla=hla,
                               batch_size=batch_size,
                               models=models)

                df_beta['beta'] = beta_list
                df_beta['pos'] = pos
                df_beta['ref'] = ref
                df_beta['alt'] = alt
                if self.regression:
                    df_beta['high'] = out[:, 0]
                else:
                    for ii in range(out.shape[1]):
                        df_beta[self.lb.inverse_transform([ii])[0]] = out[:, ii]

        if chain == 'alpha':
            return df_alpha
        elif chain == 'beta':
            return df_beta

    def Residue_Sensitivity_Logo(self,alpha_sequences=None, beta_sequences=None, v_beta=None, d_beta=None, j_beta=None,
                                v_alpha=None, j_alpha=None, hla=None,p=None, batch_size=10000,models=None,
                                 figsize=(4,8),low_color='red',medium_color='white',high_color='blue',
                                    font_name='serif',class_sel=None,
                                 cmap=None,min_size=0.0,edgecolor='black',edgewidth=0.25,background_color='black',
                                 Load_Prev_Data=False,norm_to_seq=True):
        """
        # Create Residue Sensitivity Logos

        This method allows the user to create Residue Sensitivity Logos where a set of provided sequences is perturbed to assess for position of the CDR3 sequence that if altered, would change the predicted specificity or affinity of the sequence (depending on whether training classification or regression task).

        Residue Sensitivity Logos can be created from any supervised model (including sequence and repertoire classifiers). Following the training of one of these models, one can feed into this method an cdr3 sequence defined by all/any of alpha/beta cdr3 sequence, V/D/J gene usage, and HLA context within which the TCR was seen.

        The output is a logo created by LogoMaker where the size of the character denotes how sensitive this position is to perturbation and color denotes the consequences of changes at this site. As default, red coloration means changes at this site would generally decrease the predicted value and blue coloration means changes at this site would increase the predicted value.

        Args:

            alpha_sequences (ndarray of strings): A 1d array with the sequences for inference for the alpha chain.

            beta_sequences (ndarray of strings): A 1d array with the sequences for inference for the beta chain.

            v_beta (ndarray of strings): A 1d array with the v-beta genes for inference.

            d_beta (ndarray of strings): A 1d array with the d-beta genes for inference.

            j_beta (ndarray of strings): A 1d array with the j-beta genes for inference.

            v_alpha (ndarray of strings): A 1d array with the v-alpha genes for inference.

            j_alpha (ndarray of strings): A 1d array with the j-alpha genes for inference.

            hla (ndarray of tuples/arrays): To input the hla context for each sequence fed into DeepTCR, this will need to formatted as an ndarray that is (N,) for each sequence where each entry is a tuple/array of strings referring to the alleles seen for that sequence. ('A*01:01', 'A*11:01', 'B*35:01', 'B*35:02', 'C*04:01')

            p (multiprocessing pool object): a pre-formed pool object can be passed to method for multiprocessing tasks.

            batch_size (int): Batch size for inference.

            models (list): In the case of the supervised sequence classifier, if several models were trained (via MC or Kfold crossvals), one can specify which ones to use for inference. Otherwise, thie method uses all trained models found in Name/models/ in an ensemble fashion. The method will output of the average of all models as well as the distribution of outputs for the user.

            figsize (tuple): This specifies the dimensions of the logo.

            low_color (str): The color to use when changes at this site would largely result in decreased prediction values.

            medium_color (str): The color to use when changes at this site would result in either decreased or inreased prediction values.

            high_color (str): The color to use when changes at this site would result in increased prediction values.

            font_name (str): The font to use for LogoMaker.

            class_sel (str): In the case of a model being trained in a multi-class fashion, one must select which class to make the logo for.

            cmap (matplotlib cmap): One can alsp provide custom cmap for logomaker that will be used to denote changes at sites that result in increased of decreased prediction values.

            min_size (float (0.0 - 1.0)):
            Some residues may have such little change with any perturbation that the character would be difficult to read. To set a minimum size for a residue, one can set this parameter to a value between 0 and 1.

            edgecolor (str): The color of the edge of the characters of the logo.

            edgewidth (float): The thickness of the edge of the characters.

            background_color (str): The background color of the logo.

            norm_to_seq (bool): When determining the color intensity of the logo, one can choose to normalize the value to just characters in that sequence (True) or one can choose to normalize to all characters in the sequences provdied (False).

            Load_Prev_Data (bool): Since the first part of the method runs a time-intensive step to get all the predictions for all perturbations at all residue sites, we've incorporated a paramter which can be set to True following running the method once in order to adjust the visual aspects of the plot. Therefore, one should run this method first setting this parameter to False (it's default setting) but then switch to True and run again with different visualization parameters (i.e. figsize, etc).

        Returns:
            Residue Sensitivity Logo

            - (fig,ax) - the matplotlib figure and axis/axes.

        """

        self.model_type, get = load_model_data(self)
        if Load_Prev_Data is False:
            if p is None:
                p_ = Pool(40)
            else:
                p_ = p

            inputs = [alpha_sequences,beta_sequences,v_beta,d_beta,j_beta,v_alpha,j_alpha, hla]

            for i in inputs:
                if i is not None:
                    assert isinstance(i,np.ndarray),'Inputs into DeepTCR must come in as numpy arrays!'

            inputs = [alpha_sequences,beta_sequences,v_beta,d_beta,j_beta,v_alpha,j_alpha,hla]
            for i in inputs:
                if i is not None:
                    len_input = len(i)
                    break

            if alpha_sequences is None:
                alpha_sequences = np.array([None]*len_input)

            if beta_sequences is None:
                beta_sequences = np.array([None]*len_input)

            if v_beta is None:
                v_beta = np.array([None]*len_input)

            if d_beta is None:
                d_beta = np.array([None] * len_input)

            if j_beta is None:
                j_beta = np.array([None]*len_input)

            if v_alpha is None:
                v_alpha = np.array([None]*len_input)

            if j_alpha is None:
                j_alpha = np.array([None]*len_input)

            if hla is None:
                hla = np.array([None]*len_input)

            alpha_matrices = []
            alpha_masks = []
            beta_matrices = []
            beta_masks = []
            df_alpha_list = []
            df_beta_list = []
            for i in range(len_input):
                df_alpha = self._residue(alpha_sequences[i], beta_sequences[i],
                                         v_beta[i], d_beta[i], j_beta[i],
                                         v_alpha[i], j_alpha[i], hla[i],
                                         p_, batch_size, models, 'alpha')
                df_beta = self._residue(alpha_sequences[i], beta_sequences[i],
                                        v_beta[i], d_beta[i], j_beta[i],
                                        v_alpha[i], j_alpha[i], hla[i],
                                        p_, batch_size, models, 'beta')
                df_alpha_list.append(df_alpha)
                df_beta_list.append(df_beta)
                if not df_alpha.empty:
                    if self.regression:
                        temp = np.zeros(shape=[len(alpha_sequences[i]),len(self.aa_idx.keys())])
                        temp_mask = np.zeros(shape=[len(alpha_sequences[i]),len(self.aa_idx.keys())])
                        for _ in df_alpha.iterrows():
                            temp[_[1]['pos'],self.aa_idx[_[1]['alt']]-1] =_[1]['high']
                            if _[1]['ref'] == _[1]['alt']:
                                temp_mask[_[1]['pos'], self.aa_idx[_[1]['alt']] - 1] = 1
                        alpha_matrices.append(temp)
                        alpha_masks.append(temp_mask)
                    else:
                        temp = []
                        temp_mask = []
                        for ii, cl in enumerate(self.lb.classes_, 0):
                            temp_i = np.zeros(shape=[len(alpha_sequences[i]), len(self.aa_idx.keys())])
                            temp_mask_i = np.zeros(shape=[len(alpha_sequences[i]), len(self.aa_idx.keys())])
                            for _ in df_alpha.iterrows():
                                temp_i[_[1]['pos'], self.aa_idx[_[1]['alt']] - 1] = _[1][cl]
                                if _[1]['ref'] == _[1]['alt']:
                                    temp_mask_i[_[1]['pos'], self.aa_idx[_[1]['alt']] - 1] = 1
                            temp.append(temp_i)
                            temp_mask.append(temp_mask_i)
                        temp = np.stack(temp, 0)
                        temp_mask = np.stack(temp_mask, 0)
                        temp = temp[self.lb.transform([class_sel])[0]]
                        temp_mask = temp_mask[self.lb.transform([class_sel])[0]]
                        alpha_matrices.append(temp)
                        alpha_masks.append(temp_mask)

                if not df_beta.empty:
                    if self.regression:
                        temp = np.zeros(shape=[len(beta_sequences[i]), len(self.aa_idx.keys())])
                        temp_mask = np.zeros(shape=[len(beta_sequences[i]),len(self.aa_idx.keys())])
                        for _ in df_beta.iterrows():
                            temp[_[1]['pos'], self.aa_idx[_[1]['alt']] - 1] = _[1]['high']
                            if _[1]['ref'] == _[1]['alt']:
                                temp_mask[_[1]['pos'], self.aa_idx[_[1]['alt']] - 1] = 1
                        beta_matrices.append(temp)
                        beta_masks.append(temp_mask)
                    else:
                        temp = []
                        temp_mask = []
                        for ii,cl in enumerate(self.lb.classes_,0):
                            temp_i = np.zeros(shape=[len(beta_sequences[i]), len(self.aa_idx.keys())])
                            temp_mask_i = np.zeros(shape=[len(beta_sequences[i]), len(self.aa_idx.keys())])
                            for _ in df_beta.iterrows():
                                temp_i[_[1]['pos'], self.aa_idx[_[1]['alt']] - 1] = _[1][cl]
                                if _[1]['ref'] == _[1]['alt']:
                                    temp_mask_i[_[1]['pos'], self.aa_idx[_[1]['alt']] - 1] = 1
                            temp.append(temp_i)
                            temp_mask.append(temp_mask_i)
                        temp = np.stack(temp,0)
                        temp_mask = np.stack(temp_mask,0)
                        temp = temp[self.lb.transform([class_sel])[0]]
                        temp_mask = temp_mask[self.lb.transform([class_sel])[0]]
                        beta_matrices.append(temp)
                        beta_masks.append(temp_mask)


            if p is None:
                p_.close()
                p_.join()

            with open(os.path.join(self.Name,'sens_data.pkl'),'wb') as f:
                pickle.dump([alpha_sequences,alpha_matrices,alpha_masks,df_alpha_list,
                             beta_sequences,beta_matrices,beta_masks,df_beta_list],f,protocol=4)

        else:
            with open(os.path.join(self.Name,'sens_data.pkl'),'rb') as f:
                alpha_sequences, alpha_matrices, alpha_masks,df_alpha_list,\
                beta_sequences, beta_matrices, beta_masks,df_beta_list = pickle.load(f)

        max_max_diff = []
        max_mean_diff = []
        if self.use_alpha:
            max_max_diff_alpha,max_mean_diff_alpha = get_max_val(alpha_matrices,alpha_masks)
            max_max_diff.append(max_max_diff_alpha)
            max_mean_diff.append(max_mean_diff_alpha)

        if self.use_beta:
            max_max_diff_beta, max_mean_diff_beta = get_max_val(beta_matrices, beta_masks)
            max_max_diff.append(max_max_diff_beta)
            max_mean_diff.append(max_mean_diff_beta)

        if norm_to_seq:
            max_max_diff = np.max(np.vstack(max_max_diff),0)
            max_mean_diff = np.max(np.vstack(max_mean_diff),0)
        else:
            max_max_diff = np.max(max_max_diff)
            max_mean_diff = np.max(max_mean_diff)
            max_max_diff = np.array([max_max_diff]*len(alpha_sequences))
            max_mean_diff = np.array([max_mean_diff]*len(alpha_sequences))

        if self.use_alpha & self.use_beta:
            fig, ax = plt.subplots(1, 2, figsize=figsize,facecolor=background_color)
            dir_alpha, mag_alpha = sensitivity_logo(alpha_sequences,alpha_matrices,alpha_masks,ax=ax[0],
                             low_color=low_color,medium_color=medium_color,high_color=high_color,font_name=font_name,
                             cmap=cmap,max_max_diff=max_max_diff,max_mean_diff=max_mean_diff,
                             min_size=min_size,edgecolor=edgecolor,edgewidth=edgewidth,background_color=background_color)
            dir_beta, mag_beta = sensitivity_logo(beta_sequences,beta_matrices,beta_masks,ax=ax[1],
                             low_color=low_color,medium_color=medium_color,high_color=high_color,font_name=font_name,
                             cmap=cmap,max_max_diff=max_max_diff,max_mean_diff=max_mean_diff,
                             min_size=min_size,edgecolor=edgecolor,edgewidth=edgewidth,background_color=background_color)
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=figsize,facecolor=background_color)
            if self.use_alpha:
                dir_alpha, mag_alpha = sensitivity_logo(alpha_sequences, alpha_matrices, alpha_masks, ax=ax,
                                 low_color=low_color,medium_color=medium_color,high_color=high_color,font_name=font_name,
                                 cmap=cmap, max_max_diff=max_max_diff, max_mean_diff=max_mean_diff,
                                 min_size=min_size, edgecolor=edgecolor, edgewidth=edgewidth,background_color=background_color)
            if self.use_beta:
                dir_beta, mag_beta = sensitivity_logo(beta_sequences, beta_matrices, beta_masks, ax=ax,
                                 low_color=low_color,medium_color=medium_color,high_color=high_color,font_name=font_name,
                                 cmap=cmap, max_max_diff=max_max_diff, max_mean_diff=max_mean_diff,
                                 min_size=min_size, edgecolor=edgecolor, edgewidth=edgewidth,background_color=background_color)
            plt.tight_layout()

        self.df_alpha_list = df_alpha_list
        self.df_beta_list = df_beta_list

        if self.use_alpha:
            self.dir_alpha, self.mag_alpha = dir_alpha, mag_alpha
        if self.use_beta:
            self.dir_beta,self.mag_beta = dir_beta, mag_beta

        return fig,ax


