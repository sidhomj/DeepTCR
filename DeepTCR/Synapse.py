import sys
sys.path.append('../')
from DeepTCR.functions.Layers import *
from DeepTCR.functions.utils_u import *
from DeepTCR.functions.utils_s import *
from DeepTCR.functions.act_fun import *
from DeepTCR.functions.plot_func import *
from DeepTCR.functions_syn.data_processing import *
import seaborn as sns
import colorsys
from scipy.cluster.hierarchy import linkage,fcluster,dendrogram, leaves_list
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import pdist, squareform
import umap
from sklearn.cluster import DBSCAN,KMeans
import sklearn
import DeepTCR.phenograph as phenograph
from scipy.spatial import distance
import glob
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import shutil
import warnings
from scipy.stats import spearmanr,gaussian_kde
from distinctipy import distinctipy
from tqdm import tqdm

class Synapse(object):

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

        inputs = [alpha_sequences,beta_sequences,epitope_sequences,v_beta,d_beta,j_beta,v_alpha,j_alpha]
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

        if p is None:
            p_.close()
            p_.join()

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

        if hla is not None:
            self.use_hla = True
            if use_hla_supertype:
                hla = supertype_conv_op(hla,keep_non_supertype_alleles)
                self.use_hla_sup = True
                self.keep_non_supertype_alleles = keep_non_supertype_alleles

            elif use_hla_seq:
                df_hla = load_hla_seq()
                hla = hla_seq_conv_op(hla,df_hla)
                self.use_hla_seq = True

            else:
                self.lb_hla = MultiLabelBinarizer()
                self.hla_data_seq_num = self.lb_hla.fit_transform(hla)
                self.hla_data_seq = hla

        else:
            self.lb_hla = MultiLabelBinarizer()
            self.hla_data_seq_num = np.zeros([len_input,1])
            self.hla_data_seq = np.zeros(len_input)

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
