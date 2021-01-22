import sys
sys.path.append('../')
from DeepTCR.functions.Layers import *
from DeepTCR.functions.utils_u import *
from DeepTCR.functions.utils_s import *
from DeepTCR.functions.act_fun import *
from DeepTCR.functions.plot_func import *
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

class DeepTCR_base(object):

    def __init__(self,Name,max_length=40,device=0):
        """
        Initialize Training Object.

        Initializes object and sets initial parameters.

        Inputs
        ---------------------------------------
        Name: str
            Name of the object.

        max_length: int
            maximum length of CDR3 sequence

        device: int
            In the case user is using tensorflow-gpu, one can
            specify the particular device to build the graphs on.
            This selects which GPU the user wants to put the graph
            and train on.

        Returns
        ---------------------------------------


        """
        #Assign parameters
        self.Name = Name
        self.max_length = max_length
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
        self.regression = False
        self.use_w = False
        self.ind = None

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

        tf.compat.v1.disable_eager_execution()

    def Get_Data(self,directory,Load_Prev_Data=False,classes=None,type_of_data_cut='Fraction_Response',data_cut=1.0,n_jobs=40,
                    aa_column_alpha = None,aa_column_beta = None, count_column = None,sep='\t',aggregate_by_aa=True,
                    v_alpha_column=None,j_alpha_column=None,
                    v_beta_column=None,j_beta_column=None,d_beta_column=None,
                 p=None,hla=None,use_hla_supertype=False,keep_non_supertype_alleles=False):
        """
        Get Data for DeepTCR

        Parse Data into appropriate inputs for neural network from directories where data is stored.

        Inputs
        ---------------------------------------
        directory: str
            Path to directory with folders with tsv files are present
            for analysis. Folders names become labels for files within them. If the directory contains
            the TCRSeq files not organized into classes/labels, DeepTCR will load all files within that directory.

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

        p: multiprocessing pool object
            For parellelized operations, one can pass a multiprocessing pool object
            to this method.

        hla: str
            In order to use HLA information as part of the TCR-seq representation, one can provide
            a csv file where the first column is the file name and the remaining columns hold HLA alleles
            for each file. By including HLA information for each repertoire being analyzed, one is able to
            find a representation of TCR-Seq that is more meaningful across repertoires with different HLA
            backgrounds.

        use_hla_supertype: bool
            Given the diversity of the HLA-loci, training with a full allele may cause over-fitting. And while individuals
            may have different HLA alleles, these different allelees may bind peptide in a functionality similar way.
            This idea of supertypes of HLA is a method by which assignments of HLA genes can be aggregated to 6 HLA-A and
            6 HLA-B supertypes. In roder to convert input of HLA-allele genes to supertypes, a more biologically functional
            representation, one can se this parameter to True and if the alleles provided are of one of 945 alleles found in
            the reference below, it will be assigned to a known supertype.

            For this method to work, alleles must be provided in the following format: A0101 where the first letter of the
            designation is the HLA loci (A or B) and then the 4 digit gene designation. HLA supertypes only exist for
            HLA-A and HLA-B. All other alleles will be dropped from the analysis.

            Sidney, J., Peters, B., Frahm, N., Brander, C., & Sette, A. (2008).
            HLA class I supertypes: a revised and updated classification. BMC immunology, 9(1), 1.

        keep_non_supertype_alleles: bool
            If assigning supertypes to HLA alleles, one can choose to keep HLA-alleles that do not have a known supertype
            (i.e. HLA-C alleles or certain HLA-A or HLA-B alleles) or discard them for the analysis. In order to keep these alleles,
            one should set this parameter to True. Default is False and non HLA-A or B alleles will be discarded.

        Returns

        self.alpha_sequences: ndarray
            array with alpha sequences (if provided)

        self.beta_sequences: ndarray
            array with beta sequences (if provided)

        self.class_id: ndarray
            array with sequence class labels

        self.sample_id: ndarray
            array with sequence file labels

        self.freq: ndarray
            array with sequence frequencies from samples

        self.counts: ndarray
            array with sequence counts from samples

        self.(v/d/j)_(alpha/beta): ndarray
            array with sequence (v/d/j)-(alpha/beta) usage

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
            data_in_dirs = True
            if classes is None:
                classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory,d))]
                classes = [f for f in classes if not f.startswith('.')]
                if not classes:
                    classes = ['None']
                    data_in_dirs = False


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
            seq_index = []
            print('Loading Data...')
            for type in self.classes:
                if data_in_dirs:
                    files_read = glob.glob(os.path.join(directory, type, ext))
                else:
                    files_read = glob.glob(os.path.join(directory,ext))
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

                DF_temp = []
                files_read_temp = []
                for df,file in zip(DF,files_read):
                    if df.empty is False:
                        DF_temp.append(df)
                        files_read_temp.append(file)

                DF = DF_temp
                files_read = files_read_temp

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
                    seq_index += df.index.tolist()

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
            seq_index = np.asarray(seq_index)

            Y = self.lb.transform(label_id)
            OH = OneHotEncoder(sparse=False,categories='auto')
            Y = OH.fit_transform(Y.reshape(-1,1))

            print('Embedding Sequences...')
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

            if hla is not None:
                self.use_hla = True
                hla_df = pd.read_csv(hla)
                if use_hla_supertype:
                    hla_df = supertype_conv(hla_df,keep_non_supertype_alleles)
                    self.use_hla_sup = True
                    self.keep_non_supertype_alleles = keep_non_supertype_alleles
                hla_df = hla_df.set_index(hla_df.columns[0])
                hla_id = []
                hla_data = []
                for i in hla_df.iterrows():
                    hla_id.append(i[0])
                    temp = np.asarray(i[1].dropna().tolist())
                    hla_data.append(temp)

                hla_id = np.asarray(hla_id)
                hla_data = np.asarray(hla_data)

                keep,idx_1,idx_2 = np.intersect1d(file_list,hla_id,return_indices=True)
                file_list = keep
                hla_data = hla_data[idx_2]

                self.lb_hla = MultiLabelBinarizer()
                hla_data_num = self.lb_hla.fit_transform(hla_data)

                hla_data_seq_num = np.zeros(shape=[file_id.shape[0],hla_data_num.shape[1]])
                for file,h in zip(file_list,hla_data_num):
                    hla_data_seq_num[file_id==file] = h
                hla_data_seq_num = hla_data_seq_num.astype(int)
                hla_data_seq = np.asarray(self.lb_hla.inverse_transform(hla_data_seq_num))

                #remove sequences with no hla information
                idx_keep = np.sum(hla_data_seq_num,-1)>0
                X_Seq_alpha = X_Seq_alpha[idx_keep]
                X_Seq_beta = X_Seq_beta[idx_keep]
                Y = Y[idx_keep]
                alpha_sequences = alpha_sequences[idx_keep]
                beta_sequences = beta_sequences[idx_keep]
                label_id = label_id[idx_keep]
                file_id = file_id[idx_keep]
                freq = freq[idx_keep]
                counts = counts[idx_keep]
                seq_index = seq_index[idx_keep]
                v_beta = v_beta[idx_keep]
                d_beta = d_beta[idx_keep]
                j_beta = j_beta[idx_keep]
                v_alpha = v_alpha[idx_keep]
                j_alpha = j_alpha[idx_keep]
                v_beta_num = v_beta_num[idx_keep]
                d_beta_num = d_beta_num[idx_keep]
                j_beta_num = j_beta_num[idx_keep]
                v_alpha_num = v_alpha_num[idx_keep]
                j_alpha_num = j_alpha_num[idx_keep]
                hla_data_seq = hla_data_seq[idx_keep]
                hla_data_seq_num = hla_data_seq_num[idx_keep]

            else:
                self.lb_hla = MultiLabelBinarizer()
                file_list = np.asarray(file_list)
                hla_data = np.asarray(['None']*len(file_list))
                hla_data_num = np.asarray(['None']*len(file_list))
                hla_data_seq = np.asarray(['None']*len(file_id))
                hla_data_seq_num = np.asarray(['None']*len(file_id))

            with open(os.path.join(self.Name,self.Name) + '_Data.pkl', 'wb') as f:
                pickle.dump([X_Seq_alpha,X_Seq_beta,Y, alpha_sequences,beta_sequences, label_id, file_id, freq,counts,seq_index,
                             self.lb,file_list,self.use_alpha,self.use_beta,
                             self.lb_v_beta, self.lb_d_beta, self.lb_j_beta,self.lb_v_alpha,self.lb_j_alpha,
                             v_beta, d_beta,j_beta,v_alpha,j_alpha,
                             v_beta_num, d_beta_num, j_beta_num,v_alpha_num,j_alpha_num,
                             self.use_v_beta,self.use_d_beta,self.use_j_beta,self.use_v_alpha,self.use_j_alpha,
                             self.lb_hla, hla_data, hla_data_num,hla_data_seq,hla_data_seq_num,
                             self.use_hla,self.use_hla_sup,self.keep_non_supertype_alleles],f,protocol=4)

        else:
            with open(os.path.join(self.Name,self.Name) + '_Data.pkl', 'rb') as f:
                X_Seq_alpha,X_Seq_beta,Y, alpha_sequences,beta_sequences, label_id, file_id, freq,counts,seq_index,\
                self.lb,file_list,self.use_alpha,self.use_beta,\
                    self.lb_v_beta, self.lb_d_beta, self.lb_j_beta,self.lb_v_alpha,self.lb_j_alpha,\
                    v_beta, d_beta,j_beta,v_alpha,j_alpha,\
                    v_beta_num, d_beta_num, j_beta_num,v_alpha_num,j_alpha_num,\
                    self.use_v_beta,self.use_d_beta,self.use_j_beta,self.use_v_alpha,self.use_j_alpha,\
                    self.lb_hla, hla_data,hla_data_num,hla_data_seq,hla_data_seq_num,\
                self.use_hla,self.use_hla_sup,self.keep_non_supertype_alleles = pickle.load(f)

        self.X_Seq_alpha = X_Seq_alpha
        self.X_Seq_beta = X_Seq_beta
        self.Y = Y
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
        self.seq_index = np.asarray(list(range(len(self.Y))))
        self.predicted = np.zeros((len(self.Y),len(self.lb.classes_)))
        self.hla_data_seq = hla_data_seq
        self.hla_data_seq_num = hla_data_seq_num
        self.w = np.ones(len(self.seq_index))
        #self.seq_index_j = seq_index
        print('Data Loaded')

    def Load_Data(self,alpha_sequences=None,beta_sequences=None,v_beta=None,d_beta=None,j_beta=None,
                  v_alpha=None,j_alpha=None,class_labels=None,sample_labels=None,freq=None,counts=None,Y=None,
                  p=None,hla=None,use_hla_supertype=False,keep_non_supertype_alleles=False,w=None):
        """
        Load Data programatically into DeepTCR.

        DeepTCR allows direct user input of sequence data for DeepTCR analysis. By using this method,
        a user can load numpy arrays with relevant TCRSeq data for analysis.

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

        class_labels: ndarray of strings
            A 1d array with class labels for the sequence (i.e. antigen-specificities)

        sample_labels: ndarray of strings
            A 1d array with sample labels for the sequence. (i.e. when loading data from different samples)

        counts: ndarray of ints
            A 1d array with the counts for each sequence, in the case they come from samples.

        freq: ndarray of float values
            A 1d array with the frequencies for each sequence, in the case they come from samples.

        Y: ndarray of float values
            In the case one wants to regress TCR sequences or repertoires against a numerical label, one can provide
            these numerical values for this input. For the TCR sequence regressor, each sequence will be regressed to
            the value denoted for each sequence. For the TCR repertoire regressor, the average of all instance level values
            will be used to regress the sample. Therefore, if there is one sample level value for regression, one would just
            repeat that same value for all the instances/sequences of the sample.

        hla: ndarray of tuples/arrays
            To input the hla context for each sequence fed into DeepTCR, this will need to formatted
            as an ndarray that is (N,) for each sequence where each entry is a tuple or array of strings referring
            to the alleles seen for that sequence.
                ('A*01:01', 'A*11:01', 'B*35:01', 'B*35:02', 'C*04:01')

        use_hla_supertype: bool
            Given the diversity of the HLA-loci, training with a full allele may cause over-fitting. And while individuals
            may have different HLA alleles, these different allelees may bind peptide in a functionality similar way.
            This idea of supertypes of HLA is a method by which assignments of HLA genes can be aggregated to 6 HLA-A and
            6 HLA-B supertypes. In roder to convert input of HLA-allele genes to supertypes, a more biologically functional
            representation, one can se this parameter to True and if the alleles provided are of one of 945 alleles found in
            the reference below, it will be assigned to a known supertype.

            For this method to work, alleles must be provided in the following format: A0101 where the first letter of the
            designation is the HLA loci (A or B) and then the 4 digit gene designation. HLA supertypes only exist for
            HLA-A and HLA-B. All other alleles will be dropped from the analysis.

            Sidney, J., Peters, B., Frahm, N., Brander, C., & Sette, A. (2008).
            HLA class I supertypes: a revised and updated classification. BMC immunology, 9(1), 1.

        keep_non_supertype_alleles: bool
            If assigning supertypes to HLA alleles, one can choose to keep HLA-alleles that do not have a known supertype
            (i.e. HLA-C alleles or certain HLA-A or HLA-B alleles) or discard them for the analysis. In order to keep these alleles,
            one should set this parameter to True. Default is False and non HLA-A or B alleles will be discarded.

        p: multiprocessing pool object
            a pre-formed pool object can be passed to method for multiprocessing tasks.

        w: ndarray
            optional set of weights for training of autoencoder

        Returns

        self.alpha_sequences: ndarray
            array with alpha sequences (if provided)

        self.beta_sequences: ndarray
            array with beta sequences (if provided)

        self.label_id: ndarray
            array with sequence class labels

        self.file_id: ndarray
            array with sequence file labels

        self.freq: ndarray
            array with sequence frequencies from samples

        self.counts: ndarray
            array with sequence counts from samples

        self.(v/d/j)_(alpha/beta): ndarray
            array with sequence (v/d/j)-(alpha/beta) usage

        ---------------------------------------

        """

        inputs = [alpha_sequences,beta_sequences,v_beta,d_beta,j_beta,v_alpha,j_alpha,
                  class_labels,sample_labels,counts,freq,Y,hla,w]

        for i in inputs:
            if i is not None:
                assert isinstance(i,np.ndarray),'Inputs into DeepTCR must come in as numpy arrays!'

        inputs = [alpha_sequences,beta_sequences,v_beta,d_beta,j_beta,v_alpha,j_alpha]
        for i in inputs:
            if i is not None:
                len_input = len(i)
                break

        if p is None:
            p = Pool(40)

        if alpha_sequences is not None:
            self.alpha_sequences = alpha_sequences
            args = list(zip(alpha_sequences, [self.aa_idx] * len(alpha_sequences), [self.max_length] * len(alpha_sequences)))
            result = p.starmap(Embed_Seq_Num, args)
            sequences_num = np.vstack(result)
            self.X_Seq_alpha = np.expand_dims(sequences_num, 1)
            self.use_alpha = True
        else:
            self.X_Seq_alpha = np.zeros(shape=[len_input])
            self.alpha_sequences = np.asarray([None] * len_input)

        if beta_sequences is not None:
            self.beta_sequences = beta_sequences
            args = list(zip(beta_sequences, [self.aa_idx] * len(beta_sequences), [self.max_length] * len(beta_sequences)))
            result = p.starmap(Embed_Seq_Num, args)
            sequences_num = np.vstack(result)
            self.X_Seq_beta = np.expand_dims(sequences_num, 1)
            self.use_beta = True
        else:
            self.X_Seq_beta = np.zeros(shape=[len_input])
            self.beta_sequences = np.asarray([None] * len_input)

        if v_beta is not None:
            self.v_beta = v_beta
            self.lb_v_beta = LabelEncoder()
            self.v_beta_num = self.lb_v_beta.fit_transform(v_beta)
            self.use_v_beta = True
        else:
            self.lb_v_beta = LabelEncoder()
            self.v_beta_num = np.zeros(shape=[len_input])
            self.v_beta = np.asarray([None] * len_input)

        if d_beta is not None:
            self.d_beta = d_beta
            self.lb_d_beta = LabelEncoder()
            self.d_beta_num = self.lb_d_beta.fit_transform(d_beta)
            self.use_d_beta = True
        else:
            self.lb_d_beta = LabelEncoder()
            self.d_beta_num = np.zeros(shape=[len_input])
            self.d_beta = np.asarray([None] * len_input)

        if j_beta is not None:
            self.j_beta = j_beta
            self.lb_j_beta = LabelEncoder()
            self.j_beta_num = self.lb_j_beta.fit_transform(j_beta)
            self.use_j_beta = True
        else:
            self.lb_j_beta = LabelEncoder()
            self.j_beta_num = np.zeros(shape=[len_input])
            self.j_beta = np.asarray([None] * len_input)

        if v_alpha is not None:
            self.v_alpha = v_alpha
            self.lb_v_alpha = LabelEncoder()
            self.v_alpha_num = self.lb_v_alpha.fit_transform(v_alpha)
            self.use_v_alpha = True
        else:
            self.lb_v_alpha = LabelEncoder()
            self.v_alpha_num = np.zeros(shape=[len_input])
            self.v_alpha = np.asarray([None] * len_input)

        if j_alpha is not None:
            self.j_alpha = j_alpha
            self.lb_j_alpha = LabelEncoder()
            self.j_alpha_num = self.lb_j_alpha.fit_transform(j_alpha)
            self.use_j_alpha = True
        else:
            self.lb_j_alpha = LabelEncoder()
            self.j_alpha_num = np.zeros(shape=[len_input])
            self.j_alpha = np.asarray([None] * len_input)

        if p is None:
            p.close()
            p.join()

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
            if use_hla_supertype:
                hla = supertype_conv_op(hla,keep_non_supertype_alleles)
                self.use_hla_sup = True
                self.keep_non_supertype_alleles = keep_non_supertype_alleles

            self.lb_hla = MultiLabelBinarizer()
            self.hla_data_seq_num = self.lb_hla.fit_transform(hla)
            self.hla_data_seq = hla
            self.use_hla = True
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

    def Sequence_Inference(self, alpha_sequences=None, beta_sequences=None, v_beta=None, d_beta=None, j_beta=None,
                  v_alpha=None, j_alpha=None, p=None,hla=None, batch_size=10000,models=None,return_dist=False):
        """
        Predicting outputs of sequence models on new data

        This method allows a user to take a pre-trained autoencoder/sequence classifier
        and generate outputs from the model on new data. For the autoencoder, this returns
        the features from the latent space. For the sequence classifier, it is the probability
        of belonging to each class.

        In the case that multiple models have been trained via MC or K-fold Cross-Validation strategy for the
        sequence classifier, this method can use some or all trained models in an ensemble fashion to provide the
        average prediction per sequence as well as the distribution of predictions from all trained models.

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

        hla: ndarray of tuples/arrays
            To input the hla context for each sequence fed into DeepTCR, this will need to formatted
            as an ndarray that is (N,) for each sequence where each entry is a tuple/array of strings referring
            to the alleles seen for that sequence.
                ('A*01:01', 'A*11:01', 'B*35:01', 'B*35:02', 'C*04:01')

            If the model used for inference was trained to use HLA-supertypes, one should still enter the HLA
            in the format it was provided to the original model (i.e. A0101). This mehthod will then convert those
            HLA alleles into the appropriaet supertype designation. The HLA alleles DO NOT need to be provided to
            this method in the supertype designation.

        p: multiprocessing pool object
            a pre-formed pool object can be passed to method for multiprocessing tasks.

        batch_size: int
            Batch size for inference.

        models: list
            In the case of the supervised sequence classifier, if several models were trained (via MC or Kfold crossvals),
            one can specify which ones to use for inference. Otherwise, thie method uses all trained models found in
            Name/models/ in an ensemble fashion. The method will output of the average of all models as well as the
            distribution of outputs for the user.

        return_dist: bool
            If the user wants to also return teh distribution of sequence predicionts over all models use dfor inference,
            one should set this value to True.

        Returns
        [features, features_dist]

        features: array
            shape = [N, latent_dim]

            An array that contains n x latent_dim containing features for all sequences. For the VAE, this represents
            the features from the latent space. For the sequence classifier, this represents the probabilities for every
            class or the regressed value from the sequence regressor. In the case of multiple models being used for infernece
            in ensemble, this becomes the average prediction over all models.

        features_dist: array
            shape = [n_models,N,latent_dim]

            An array that contains the output of all models separately for each input sequence. This output is useful
            if using multiple models in ensemble to predict on a new sequence. This output describes the distribution
            of the predictions over all models.

        ---------------------------------------

        """
        model_type,get = load_model_data(self)
        out, out_dist = inference_method_ss(get,alpha_sequences,beta_sequences,
                               v_beta,d_beta,j_beta,v_alpha,j_alpha,hla,
                                p,batch_size,self,models)

        if return_dist:
            return out, out_dist
        else:
            return out

class feature_analytics_class(object):
    def Structural_Diversity(self, sample=None, n_jobs=1):
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
            IDX, _, _ = phenograph.cluster(features_sel, n_jobs=n_jobs)
            knn_class = sklearn.neighbors.KNeighborsClassifier(n_neighbors=30, n_jobs=n_jobs).fit(features_sel, IDX)
            IDX = knn_class.predict(self.features)
        else:
            IDX, _, _ = phenograph.cluster(self.features, k=30, n_jobs=n_jobs)

        DFs = []
        DF_Sum = pd.DataFrame()
        DF_Sum['Sample'] = self.sample_list
        DF_Sum.set_index('Sample', inplace=True)
        for i in np.unique(IDX):
            if i != -1:
                sel = IDX == i
                seq_alpha = self.alpha_sequences[sel]
                seq_beta = self.beta_sequences[sel]
                label = self.class_id[sel]
                file = self.sample_id[sel]
                freq = self.freq[sel]

                df = pd.DataFrame()
                df['Alpha_Sequences'] = seq_alpha
                df['Beta_Sequences'] = seq_beta
                df['Labels'] = label
                df['Sample'] = file
                df['Frequency'] = freq
                df['V_alpha'] = self.v_alpha[sel]
                df['J_alpha'] = self.j_alpha[sel]
                df['V_beta'] = self.v_beta[sel]
                df['D_beta'] = self.d_beta[sel]
                df['J_beta'] = self.j_beta[sel]

                df_sum = df.groupby(by='Sample', sort=False).agg({'Frequency': 'sum'})

                DF_Sum['Cluster_' + str(i)] = df_sum

                DFs.append(df)

        DF_Sum.fillna(0.0, inplace=True)

        labels = []
        num_clusters = []
        entropy_list = []
        for file in self.sample_list:
            v = np.array(DF_Sum.loc[file].tolist())
            v = v[v > 0.0]
            entropy_list.append(entropy(v))
            num_clusters.append(len(v))
            labels.append(self.class_id[self.sample_id == file][0])

        df_out = pd.DataFrame()
        df_out['Sample'] = self.sample_list
        df_out['Class'] = labels
        df_out['Entropy'] = entropy_list
        df_out['Num of Clusters'] = num_clusters

        self.Structural_Diversity_DF = df_out

    def Cluster(self,set='all', clustering_method='phenograph', t=None, criterion='distance',
                linkage_method='ward', write_to_sheets=False, sample=None, n_jobs=1,order_by_linkage=False):

        """
        Clustering Sequences by Latent Features

        This method clusters all sequences by learned latent features from
        either the variational autoencoder Several clustering algorithms are included including
        Phenograph, DBSCAN, or hierarchical clustering. DBSCAN is implemented from the
        sklearn package. Hierarchical clustering is implemented from the scipy package.

        Inputs
        ---------------------------------------

        set: str
            To choose which set of sequences to analye, enter either
            'all','train', 'valid',or 'test'. Since the sequences in the train set
            may be overfit, it preferable to generally examine the test set on its own.

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

        order_by_linkage: bool
            To list sequences in the cluster dataframes by how they are related via ward's linakge,
            set this value to True. Otherwise, each cluster dataframe will list the sequences by the order they
            were loaded into DeepTCR.

        Returns

        self.Cluster_DFs: list of Pandas dataframes
            Clusters by sequences/label

        self.var: list
            Variance of lengths in each cluster

        self.Cluster_Frequencies: Pandas dataframe
            A dataframe containing the frequency contribution of each cluster to each sample.

        self.Cluster_Assignments: ndarray
            Array with cluster assignments by number.

        ---------------------------------------

        """
        if set == 'all':
            features = self.features
            class_id = self.class_id
            sample_id = self.sample_id
            freq = self.freq
            alpha_sequences = self.alpha_sequences
            beta_sequences = self.beta_sequences
            v_alpha = self.v_alpha
            j_alpha = self.j_alpha
            v_beta = self.v_beta
            d_beta = self.d_beta
            j_beta = self.j_beta
            hla_data_seq = self.hla_data_seq

        elif set == 'train':
            features = self.features[self.train_idx]
            class_id = self.class_id[self.train_idx]
            sample_id = self.sample_id[self.train_idx]
            freq = self.freq[self.train_idx]
            alpha_sequences = self.alpha_sequences[self.train_idx]
            beta_sequences = self.beta_sequences[self.train_idx]
            v_alpha = self.v_alpha[self.train_idx]
            j_alpha = self.j_alpha[self.train_idx]
            v_beta = self.v_beta[self.train_idx]
            d_beta = self.d_beta[self.train_idx]
            j_beta = self.j_beta[self.train_idx]
            hla_data_seq = self.hla_data_seq[self.train_idx]

        elif set == 'valid':
            features = self.features[self.valid_idx]
            class_id = self.class_id[self.valid_idx]
            sample_id = self.sample_id[self.valid_idx]
            freq = self.freq[self.valid_idx]
            alpha_sequences = self.alpha_sequences[self.valid_idx]
            beta_sequences = self.beta_sequences[self.valid_idx]
            v_alpha = self.v_alpha[self.valid_idx]
            j_alpha = self.j_alpha[self.valid_idx]
            v_beta = self.v_beta[self.valid_idx]
            d_beta = self.d_beta[self.valid_idx]
            j_beta = self.j_beta[self.valid_idx]
            hla_data_seq = self.hla_data_seq[self.valid_idx]


        elif set == 'test':
            features = self.features[self.test_idx]
            class_id = self.class_id[self.test_idx]
            sample_id = self.sample_id[self.test_idx]
            freq = self.freq[self.test_idx]
            alpha_sequences = self.alpha_sequences[self.test_idx]
            beta_sequences = self.beta_sequences[self.test_idx]
            v_alpha = self.v_alpha[self.test_idx]
            j_alpha = self.j_alpha[self.test_idx]
            v_beta = self.v_beta[self.test_idx]
            d_beta = self.d_beta[self.test_idx]
            j_beta = self.j_beta[self.test_idx]
            hla_data_seq = self.hla_data_seq[self.test_idx]


        if sample is not None:
            idx_sel = np.random.choice(range(len(features)), sample, replace=False)
            features_sel = features[idx_sel]
            distances = squareform(pdist(features_sel))

            if clustering_method == 'hierarchical':
                if t is None:
                    IDX = hierarchical_optimization(distances, features_sel, method=linkage_method, criterion=criterion)
                else:
                    Z = linkage(squareform(distances), method=linkage_method)
                    IDX = fcluster(Z, t, criterion=criterion)

            elif clustering_method == 'dbscan':
                if t is None:
                    IDX = dbscan_optimization(distances, features_sel)
                else:
                    IDX = DBSCAN(eps=t, metric='precomputed').fit_predict(distances)
                    IDX[IDX == -1] = np.max(IDX + 1)

            elif clustering_method == 'phenograph':
                IDX, _, _ = phenograph.cluster(features_sel, k=30, n_jobs=n_jobs)

            elif clustering_method == 'kmeans':
                IDX = KMeans(n_clusters=t).fit_predict(features_sel)

            knn_class = sklearn.neighbors.KNeighborsClassifier(n_neighbors=30, n_jobs=n_jobs).fit(features_sel, IDX)
            IDX = knn_class.predict(features)
        else:
            distances = squareform(pdist(features))
            if clustering_method == 'hierarchical':
                if t is None:
                    IDX = hierarchical_optimization(distances, features, method=linkage_method, criterion=criterion)
                else:
                    Z = linkage(squareform(distances), method=linkage_method)
                    IDX = fcluster(Z, t, criterion=criterion)

            elif clustering_method == 'dbscan':
                if t is None:
                    IDX = dbscan_optimization(distances, features)
                else:
                    IDX = DBSCAN(eps=t, metric='precomputed').fit_predict(distances)
                    IDX[IDX == -1] = np.max(IDX + 1)

            elif clustering_method == 'phenograph':
                IDX, _, _ = phenograph.cluster(features, k=30, n_jobs=n_jobs)

            elif clustering_method == 'kmeans':
                IDX = KMeans(n_clusters=t).fit_predict(features)

        DFs = []
        DF_Sum = pd.DataFrame()
        DF_Sum['Sample'] = np.unique(sample_id)
        DF_Sum.set_index('Sample', inplace=True)
        var_list_alpha = []
        var_list_beta = []
        for i in np.unique(IDX):
            if i != -1:
                sel = IDX == i
                seq_alpha = alpha_sequences[sel]
                seq_beta = beta_sequences[sel]
                label = class_id[sel]
                file = sample_id[sel]
                freq_sel = freq[sel]

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
                df['index'] = np.where(sel)[0]
                df['Alpha_Sequences'] = seq_alpha
                df['Beta_Sequences'] = seq_beta
                df['V_alpha'] = v_alpha[sel]
                df['J_alpha'] = j_alpha[sel]
                df['V_beta'] = v_beta[sel]
                df['D_beta'] = d_beta[sel]
                df['J_beta'] = j_beta[sel]
                df['Frequency'] = freq_sel
                df['Labels'] = label
                df['Sample'] = file

                if np.unique(hla_data_seq[sel])[0]==0:
                    df['HLA'] = None
                else:
                    df['HLA'] = list(map(list,hla_data_seq[sel].tolist()))

                if order_by_linkage:
                    df = df.iloc[leaves_list(linkage(features[sel],'ward'))]

                df_sum = df.groupby(by='Sample', sort=False).agg({'Frequency': 'sum'})

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

        self.Cluster_DFs = DFs
        self.Cluster_Frequencies = DF_Sum
        self.var_alpha = var_list_alpha
        self.var_beta = var_list_beta
        self.Cluster_Assignments = IDX
        print('Clustering Done')

    def Motif_Identification(self,group,p_val_threshold=0.05,by_samples=False,top_seq=10):
        """
        Motif Identification Supervised Classifiers

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

        by_samples: bool
            To run a motif identification that looks for enriched motifs at the sample
            instead of the seuence level, set this parameter to True. Otherwise, the enrichment
            analysis will be done at the sequence level.

        top_seq: int
            The number of sequences from which to derive the learned motifs. The larger the number,
            the more noisy the motif logo may be.

        Returns
        ---------------------------------------

        self.(alpha/beta)_group_features: Pandas Dataframe
            Sequences used to determine motifs in fasta files
            are stored in this dataframe where column names represent
            the feature number.

        """
        #Get Saved Features, Indices, and Sequences
        with open(os.path.join(self.Name,self.Name) + '_kernel.pkl', 'rb') as f:
            self.kernel = pickle.load(f)

        if self.use_alpha is True:
            with open(os.path.join(self.Name, self.Name) + '_alpha_features.pkl', 'rb') as f:
                self.alpha_features, self.alpha_indices, self.alpha_sequences = pickle.load(f)

        if self.use_beta is True:
            with open(os.path.join(self.Name, self.Name) + '_beta_features.pkl', 'rb') as f:
                self.beta_features, self.beta_indices, self.beta_sequences = pickle.load(f)

        group_num = np.where(self.lb.classes_ == group)[0][0]

        # Find diff expressed features
        idx_pos = self.Y[:, group_num] == 1
        idx_neg = self.Y[:, group_num] == 0

        if self.use_alpha is True:
            self.alpha_group_features = Diff_Features(self.alpha_features, self.alpha_indices, self.alpha_sequences,
                                                         'alpha', self.sample_id,p_val_threshold, idx_pos, idx_neg,
                                                        self.directory_results, group, self.kernel,by_samples,top_seq)

        if self.use_beta is True:
            self.beta_group_features = Diff_Features(self.beta_features, self.beta_indices, self.beta_sequences,
                                                        'beta',self.sample_id,p_val_threshold, idx_pos, idx_neg,
                                                        self.directory_results, group, self.kernel,by_samples,top_seq)


        print('Motif Identification Completed')

    def Sample_Features(self, set='all',Weight_by_Freq=True):
        """
        Sample-Level Feature Values

        This method returns a dataframe with the aggregate sample level features.

        Inputs
        ---------------------------------------

        set: str
            To choose which set of sequences to analye, enter either
            'all','train', 'valid',or 'test'. Since the sequences in the train set
            may be overfit, it preferable to generally examine the test set on its own.

        Weight_by_Freq: bool
            Option to weight each sequence used in aggregate measure
            of feature across sample by its frequency.

        Returns
        self.sample_featres: pandas dataframe
            This function returns the average feature vector for each sample analyzed. This can be used to make further
            downstream comparisons such as inter-repertoire distances.
        ---------------------------------------

        """

        if set == 'all':
            features = self.features
            class_id = self.class_id
            sample_id = self.sample_id
            freq = self.freq
        elif set == 'train':
            features = self.features[self.train_idx]
            class_id = self.class_id[self.train_idx]
            sample_id = self.sample_id[self.train_idx]
            freq = self.freq[self.train_idx]
        elif set == 'valid':
            features = self.features[self.valid_idx]
            class_id = self.class_id[self.valid_idx]
            sample_id = self.sample_id[self.valid_idx]
            freq = self.freq[self.valid_idx]
        elif set == 'test':
            features = self.features[self.test_idx]
            class_id = self.class_id[self.test_idx]
            sample_id = self.sample_id[self.test_idx]
            freq = self.freq[self.test_idx]

        sample_list = np.unique(sample_id)

        vector = []
        file_label = []
        for id in sample_list:
            sel = sample_id == id
            sel_idx = features[sel]
            sel_freq = np.expand_dims(freq[sel], 1)
            if Weight_by_Freq is True:
                dist = np.expand_dims(np.sum(sel_idx * sel_freq, 0), 0)
            else:
                dist = np.expand_dims(np.mean(sel_idx, 0), 0)
            file_label.append(np.unique(class_id[sel])[0])
            vector.append(dist)

        vector = np.vstack(vector)
        dfs = pd.DataFrame(vector)
        dfs.set_index(sample_list, inplace=True)
        self.sample_features = dfs

class vis_class(object):

    def HeatMap_Sequences(self,set='all', filename='Heatmap_Sequences.tif', sample_num=None,
                          sample_num_per_class=None,color_dict=None):

        """
        HeatMap of Sequences

        This method creates a heatmap/clustermap for sequences by latent features
        for the unsupervised deep lerning methods.

        Inputs
        ---------------------------------------

        set: str
            To choose which set of sequences to analye, enter either
            'all','train', 'valid',or 'test'. Since the sequences in the train set
            may be overfit, it preferable to generally examine the test set on its own.

        filename: str
            Name of file to save heatmap.

        sample_num: int
            Number of events to randomly sample for heatmap.

        sample_num_per_class: int
            Number of events to randomly sample per class for heatmap.

        color_dict: dict
            Optional dictionary to provide specified colors for classes.

        Returns
        ---------------------------------------

        """

        if set == 'all':
            features = self.features
            class_id = self.class_id
            sample_id = self.sample_id
        elif set == 'train':
            features = self.features[self.train_idx]
            class_id = self.class_id[self.train_idx]
            sample_id = self.sample_id[self.train_idx]
        elif set == 'valid':
            features = self.features[self.valid_idx]
            class_id = self.class_id[self.valid_idx]
            sample_id = self.sample_id[self.valid_idx]
        elif set == 'test':
            features = self.features[self.test_idx]
            class_id = self.class_id[self.test_idx]
            sample_id = self.sample_id[self.test_idx]

        keep=[]
        for i,column in enumerate(features.T,0):
            if len(np.unique(column)) > 1:
                keep.append(i)
        keep = np.asarray(keep)
        features = features[:,keep]

        if sample_num_per_class is not None and sample_num is not None:
            print("sample_num_per_class and sample_num cannot be assigned simultaneously")
            return

        if sample_num is not None:
            sel = np.random.choice(range(len(features)), sample_num, replace=False)
            features = features[sel]
            class_id = class_id[sel]
            sample_id = sample_id[sel]

        if sample_num_per_class is not None:
            features_temp = []
            label_temp = []
            file_temp = []
            for i in self.lb.classes_:
                sel = np.where(self.class_id == i)[0]
                sel = np.random.choice(sel, sample_num_per_class, replace=False)
                features_temp.append(features[sel])
                label_temp.append(class_id[sel])
                file_temp.append(sample_id[sel])


            features = np.vstack(features_temp)
            class_id = np.hstack(label_temp)
            sample_id = np.hstack(file_temp)


        if color_dict is None:
            N = len(np.unique(class_id))
            HSV_tuples = [(x * 1.0 / N, 1.0, 0.5) for x in range(N)]
            np.random.shuffle(HSV_tuples)
            RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
            color_dict = dict(zip(np.unique(class_id), RGB_tuples))

        row_colors = [color_dict[x] for x in class_id]
        sns.set(font_scale=0.5)
        CM = sns.clustermap(features, standard_scale=1, row_colors=row_colors, cmap='bwr')
        ax = CM.ax_heatmap
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        plt.show()
        plt.savefig(os.path.join(self.directory_results, filename))

    def HeatMap_Samples(self, set='all',filename='Heatmap_Samples.tif', Weight_by_Freq=True, color_dict=None, labels=True,
                        font_scale=1.0):
        """
        HeatMap of Samples

        This method creates a heatmap/clustermap for samples by latent features
        for the unsupervised deep learning methods.

        Inputs
        ---------------------------------------

        set: str
            To choose which set of sequences to analye, enter either
            'all','train', 'valid',or 'test'. Since the sequences in the train set
            may be overfit, it preferable to generally examine the test set on its own.

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
        self.sample_featres: pandas dataframe
            This function returns the average feature vector for each sample analyzed. This can be used to make further
            downstream comparisons such as inter-repertoire distances.
        ---------------------------------------

        """

        if set == 'all':
            features = self.features
            class_id = self.class_id
            sample_id = self.sample_id
            freq = self.freq
        elif set == 'train':
            features = self.features[self.train_idx]
            class_id = self.class_id[self.train_idx]
            sample_id = self.sample_id[self.train_idx]
            freq = self.freq[self.train_idx]
        elif set == 'valid':
            features = self.features[self.valid_idx]
            class_id = self.class_id[self.valid_idx]
            sample_id = self.sample_id[self.valid_idx]
            freq = self.freq[self.valid_idx]
        elif set == 'test':
            features = self.features[self.test_idx]
            class_id = self.class_id[self.test_idx]
            sample_id = self.sample_id[self.test_idx]
            freq = self.freq[self.test_idx]


        keep = []
        for i, column in enumerate(features.T, 0):
            if len(np.unique(column)) > 1:
                keep.append(i)
        keep = np.asarray(keep)
        features = features[:, keep]

        sample_list = np.unique(sample_id)

        vector = []
        file_label = []
        for id in sample_list:
            sel = sample_id == id
            sel_idx = features[sel]
            sel_freq = np.expand_dims(freq[sel], 1)
            if Weight_by_Freq is True:
                dist = np.expand_dims(np.sum(sel_idx * sel_freq, 0), 0)
            else:
                dist = np.expand_dims(np.mean(sel_idx, 0), 0)
            file_label.append(np.unique(class_id[sel])[0])
            vector.append(dist)

        vector = np.vstack(vector)

        if color_dict is None:
            N = len(np.unique(class_id))
            HSV_tuples = [(x * 1.0 / N, 1.0, 0.5) for x in range(N)]
            np.random.shuffle(HSV_tuples)
            RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
            color_dict = dict(zip(np.unique(class_id), RGB_tuples))

        row_colors = [color_dict[x] for x in file_label]

        dfs = pd.DataFrame(vector)
        dfs.set_index(sample_list, inplace=True)
        sns.set(font_scale=font_scale)
        CM = sns.clustermap(dfs, standard_scale=1, cmap='bwr', figsize=(12, 10), row_colors=row_colors)
        ax = CM.ax_heatmap
        ax.set_xticklabels('')
        if labels is False:
            ax.set_yticklabels('')
        plt.subplots_adjust(right=0.8)
        plt.show()
        plt.savefig(os.path.join(self.directory_results, filename))
        self.sample_features = dfs

    def Repertoire_Dendrogram(self,set='all', distance_metric='KL', sample=None, n_jobs=1, color_dict=None,
                             dendrogram_radius=0.32, repertoire_radius=0.4, linkage_method='ward',
                             gridsize=24, Load_Prev_Data=False,filename=None,sample_labels=False,
                              gaussian_sigma=0.5, vmax=0.01,n_pad=5,lw=None,log_scale=False):
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

        set: str
            To choose which set of sequences to analye, enter either
            'all','train', 'valid',or 'test'. Since the sequences in the train set
            may be overfit, it preferable to generally examine the test set on its own.

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

        filename: str
            To save dendrogram plot to results folder, enter a name for the file and the dendrogram
            will be saved to the results directory.
            i.e. dendrogram.png

        sample_labels: bool
            To show the sample labels on the dendrogram, set to True.

        gaussian_sigma: float
            The amount of blur to introduce in the plots.

        vmax: float
            Highest color density value. Color scales from 0 to vmax (i.e. larger vmax == dimmer plot)

        lw: float
            The width of the circle edge around each sample.

        log_scale: bool
            To plot the log of the counts for the UMAP density plot, set this value to True. This can be
            particularly helpful for visualization if the populations are very clonal.

        Returns

        self.pairwise_distances: Pandas dataframe
            Pairwise distances of all samples
        ---------------------------------------

        """
        if set == 'all':
            features = self.features
            class_id = self.class_id
            sample_id = self.sample_id
            counts = self.counts
        elif set == 'train':
            features = self.features[self.train_idx]
            class_id = self.class_id[self.train_idx]
            sample_id = self.sample_id[self.train_idx]
            counts = self.counts[self.train_idx]
        elif set == 'valid':
            features = self.features[self.valid_idx]
            class_id = self.class_id[self.valid_idx]
            sample_id = self.sample_id[self.valid_idx]
            counts = self.counts[self.valid_idx]
        elif set == 'test':
            features = self.features[self.test_idx]
            class_id = self.class_id[self.test_idx]
            sample_id = self.sample_id[self.test_idx]
            counts = self.counts[self.test_idx]

        if Load_Prev_Data is False:
            print('UMAP transformation...')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if sample is not None:
                    s_idx = np.random.choice(range(len(features)),sample,replace=False)
                    u_obj = umap.UMAP()
                    u_obj.fit(features[s_idx])
                    X_2 = u_obj.transform(features)
                else:
                    X_2 = umap.UMAP().fit_transform(features)

            print('PhenoGraph Clustering...')
            self.Cluster(sample=sample, n_jobs=n_jobs,set=set)
            prop = self.Cluster_Frequencies
            with open(os.path.join(self.Name, 'dendro.pkl'), 'wb') as f:
                pickle.dump([X_2, prop], f)
        else:
            with open(os.path.join(self.Name, 'dendro.pkl'), 'rb') as f:
                X_2, prop = pickle.load(f)

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
            labels.append(class_id[np.where(sample_id == i)[0][0]])

        samples = prop.index.tolist()

        if color_dict is None:
            N = len(np.unique(class_id))
            HSV_tuples = [(x * 1.0 / N, 1.0, 0.5) for x in range(N)]
            np.random.shuffle(HSV_tuples)
            RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
            color_dict = dict(zip(np.unique(class_id), RGB_tuples))

        temp_x = []
        temp_s = []
        for ii,(x,s) in enumerate(zip(X_2,sample_id),0):
            temp_x.append(counts[ii]*[x])
            temp_s.append(counts[ii]*[s])

        X_2 = np.vstack(temp_x)
        sample_id = np.hstack(temp_s)

        H = rad_plot(X_2, sample_id, samples, labels, sample_labels=sample_labels, pairwise_distances=squareform(pairwise_distances),
                     linkage_method=linkage_method, color_dict=color_dict, gridsize=gridsize, dg_radius=dendrogram_radius, axes_radius=repertoire_radius,
                     gaussian_sigma=gaussian_sigma, vmax=vmax, n_pad=n_pad, lw=lw, log_scale=log_scale, figsize=8, filename=filename)

    def UMAP_Plot(self, set='all',by_class=False, by_cluster=False,
                  by_sample=False, freq_weight=False, show_legend=True,
                  scale=100,Load_Prev_Data=False, alpha=1.0,sample=None,sample_per_class=None,filename=None,
                  prob_plot=None,plot_by_class=False):

        """
        UMAP vizualisation of TCR Sequences

        This method displays the sequences in a 2-dimensional UMAP where the user can color code points by
        class label, sample label, or prior computing clustering solution. Size of points can also be made to be proportional
        to frequency of sequence within sample.

        Inputs
        ---------------------------------------

        set: str
            To choose which set of sequences to analye, enter either
            'all','train', 'valid',or 'test'. Since the sequences in the train set
            may be overfit, it preferable to generally examine the test set on its own.

        by_class: bool
            To color the points by their class label, set to True.

        by_sample: bool
            To color the points by their sample lebel, set to True.

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

        sample: int
            Number of events to sub-sample for visualization.

        sample_per_class: int
             Number of events to randomly sample per class for UMAP.

        filename: str
            To save umap plot to results folder, enter a name for the file and the umap
            will be saved to the results directory.
            i.e. umap.png

        prob_plot: str
            To plot the predicted probabilities for the sequences as an additional heatmap, specify
            the class probability one wants to visualize (i.e. if the class of interest is class A, input
            'A' as a string). Of note, only probabilities determined from the sequences in the test set are
            displayed as a means of not showing over-fit probabilities. Therefore, it is best to use this parameter
            when the set parameter is turned to 'test'.


        Returns

        ---------------------------------------

        """
        idx = None
        features = self.features
        class_id = self.class_id
        sample_id = self.sample_id
        freq = self.freq
        predicted = self.predicted
        if hasattr(self, 'Cluster_Assignments'):
            IDX = self.Cluster_Assignments
        else:
            IDX = None

        if sample_per_class is not None and sample is not None:
            print("sample_per_class and sample cannot be assigned simultaneously")
            return

        if sample is not None:
            idx = np.random.choice(range(len(features)), sample, replace=False)
            features = features[idx]
            class_id = class_id[idx]
            sample_id = sample_id[idx]
            freq = freq[idx]
            predicted = predicted[idx]
            if hasattr(self, 'Cluster_Assignments'):
                IDX = IDX[idx]
            else:
                IDX = None

        if sample_per_class is not None:
            features_temp = []
            class_temp = []
            sample_temp = []
            freq_temp = []
            predicted_temp = []
            cluster_temp = []

            for i in self.lb.classes_:
                sel = np.where(class_id == i)[0]
                sel = np.random.choice(sel, sample_per_class, replace=False)
                features_temp.append(features[sel])
                class_temp.append(class_id[sel])
                sample_temp.append(sample_id[sel])
                freq_temp.append(freq[sel])
                predicted_temp.append(predicted[sel])
                if hasattr(self, 'Cluster_Assignments'):
                    cluster_temp.append(IDX[sel])

            features = np.vstack(features_temp)
            class_id = np.hstack(class_temp)
            sample_id = np.hstack(sample_temp)
            freq = np.hstack(freq_temp)
            predicted = np.hstack(predicted_temp)
            if hasattr(self, 'Cluster_Assignments'):
                IDX = np.hstack(cluster_temp)

        if Load_Prev_Data is False:
            umap_obj = umap.UMAP()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                X_2 = umap_obj.fit_transform(features)
            with open(os.path.join(self.Name, 'umap.pkl'), 'wb') as f:
                pickle.dump([X_2,features,class_id,sample_id,freq,IDX,idx], f, protocol=4)
        else:
            with open(os.path.join(self.Name, 'umap.pkl'), 'rb') as f:
                X_2,features,class_id,sample_id,freq,IDX,idx = pickle.load(f)

        df_plot = pd.DataFrame()
        df_plot['x'] = X_2[:, 0]
        df_plot['y'] = X_2[:, 1]
        df_plot['Class'] = class_id
        df_plot['Sample'] = sample_id

        if prob_plot is not None:
            df_plot['Predicted'] = predicted[:,self.lb.transform([prob_plot])[0]]

        if set != 'all':
            df_plot['Set'] = None
            with pd.option_context('mode.chained_assignment',None):
                df_plot['Set'].iloc[np.where(self.train_idx)[0]] = 'train'
                df_plot['Set'].iloc[np.where(self.valid_idx)[0]] = 'valid'
                df_plot['Set'].iloc[np.where(self.test_idx)[0]] = 'test'

        if IDX is not None:
            IDX[IDX == -1] = np.max(IDX) + 1
            IDX = ['Cluster_' + str(I) for I in IDX]
            df_plot['Cluster'] = IDX

        if freq_weight is True:
            s = freq * scale
        else:
            s = scale

        df_plot['s']=s

        if show_legend is True:
            legend = 'full'
        else:
            legend = False

        if by_class is True:
            hue = 'Class'
        elif by_cluster is True:
            hue = 'Cluster'
        elif by_sample is True:
            hue = 'Sample'
        else:
            hue = None

        if set == 'all':
            df_plot_sel = df_plot
        elif set == 'train':
            df_plot_sel = df_plot[df_plot['Set']=='train']
        elif set == 'valid':
            df_plot_sel = df_plot[df_plot['Set']=='valid']
        elif set == 'test':
            df_plot_sel = df_plot[df_plot['Set']=='test']

        df_plot_sel = df_plot_sel.sample(frac=1)
        plt.figure()
        sns.scatterplot(data=df_plot_sel, x='x', y='y', s=df_plot_sel['s'], hue=hue, legend=legend, alpha=alpha, linewidth=0.0)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')

        if filename is not None:
            plt.savefig(os.path.join(self.directory_results, filename))

        if prob_plot is not None:
            plt.figure()
            plt.scatter(df_plot_sel['x'],df_plot_sel['y'],c=df_plot_sel['Predicted'],s=df_plot_sel['s'],
                    alpha=alpha,cmap='bwr')
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('')
            plt.ylabel('')

            if filename is not None:
                plt.savefig(os.path.join(self.directory_results, 'prob_'+filename))

        if plot_by_class is True:
            for i in self.lb.classes_:
                sel = df_plot_sel['Class']==i
                plt.figure()
                sns.scatterplot(data=df_plot_sel[sel], x='x', y='y', s=df_plot_sel['s'][sel], hue=hue, legend=legend, alpha=alpha,
                                linewidth=0.0)
                plt.xticks([])
                plt.yticks([])
                plt.xlabel('')
                plt.ylabel('')
    
    def UMAP_Plot_Samples(self,set='all',filename='UMAP_Samples.tif', Weight_by_Freq=True,scale=5,alpha=1.0):
        """
        UMAP vizualisation of TCR Samples

        This method displays the samples in a 2-dimensional UMAP

        Inputs
        ---------------------------------------

        set: str
            To choose which set of sequences to analye, enter either
            'all','train', 'valid',or 'test'. Since the sequences in the train set
            may be overfit, it preferable to generally examine the test set on its own.

        Weight_by_Freq: bool
            Option to weight each sequence used in aggregate measure
            of feature across sample by its frequency.

        scale: float
            To change size of points, change scale parameter.

        alpha: float
            Value between 0-1 that controls transparency of points.

        filename: str
            To save umap plot to results folder, enter a name for the file and the umap
            will be saved to the results directory.
            i.e. umap.png

        Returns

        ---------------------------------------

        """

        if set == 'all':
            features = self.features
            class_id = self.class_id
            sample_id = self.sample_id
            freq = self.freq
        elif set == 'train':
            features = self.features[self.train_idx]
            class_id = self.class_id[self.train_idx]
            sample_id = self.sample_id[self.train_idx]
            freq = self.freq[self.train_idx]
        elif set == 'valid':
            features = self.features[self.valid_idx]
            class_id = self.class_id[self.valid_idx]
            sample_id = self.sample_id[self.valid_idx]
            freq = self.freq[self.valid_idx]
        elif set == 'test':
            features = self.features[self.test_idx]
            class_id = self.class_id[self.test_idx]
            sample_id = self.sample_id[self.test_idx]
            freq = self.freq[self.test_idx]


        keep = []
        for i, column in enumerate(features.T, 0):
            if len(np.unique(column)) > 1:
                keep.append(i)
        keep = np.asarray(keep)
        features = features[:, keep]

        sample_list = np.unique(sample_id)

        vector = []
        file_label = []
        for id in sample_list:
            sel = sample_id == id
            sel_idx = features[sel]
            sel_freq = np.expand_dims(freq[sel], 1)
            if Weight_by_Freq is True:
                dist = np.expand_dims(np.sum(sel_idx * sel_freq, 0), 0)
            else:
                dist = np.expand_dims(np.mean(sel_idx, 0), 0)
            file_label.append(np.unique(class_id[sel])[0])
            vector.append(dist)

        vector = np.vstack(vector)

        X_2 = umap.UMAP().fit_transform(vector)
        df_plot = pd.DataFrame()
        df_plot['x'] = X_2[:,0]
        df_plot['y'] = X_2[:,1]
        df_plot['class'] = file_label
        df_plot['s'] = scale

        plt.figure()
        legend = 'full'
        sns.scatterplot(data=df_plot, x='x', y='y', s=df_plot['s'], hue='class', legend=legend, alpha=alpha,
                        linewidth=0.0)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig(os.path.join(self.directory_results, filename))

class DeepTCR_U(DeepTCR_base,feature_analytics_class,vis_class):

    def _reset_models(self):
        self.models_dir = os.path.join(self.Name,'models')
        if os.path.exists(self.models_dir):
            shutil.rmtree(self.models_dir)
        os.makedirs(self.models_dir)

    def Train_VAE(self,latent_dim=256, kernel = 5, trainable_embedding=True, embedding_dim_aa = 64,embedding_dim_genes = 48,embedding_dim_hla=12,
                  use_only_seq=False,use_only_gene=False,use_only_hla=False,size_of_net='medium',latent_alpha=1e-3,sparsity_alpha=None,var_explained=None,graph_seed=None,
                  batch_size=10000, epochs_min=0,stop_criterion=0.01,stop_criterion_window=30, accuracy_min=None,
                  suppress_output = False,learning_rate=0.001,split_seed=None,Load_Prev_Data=False):

        """
        Train Variational Autoencoder (VAE)

        This method trains the network and saves features values for sequences
        to create heatmaps.

        Inputs
        ---------------------------------------

        Model Parameters

        latent_dim: int
            Number of latent dimensions for VAE.

        kernel: int
            The motif k-mer of the first convolutional layer of the graph.

        trainable_embedding: bool
            Toggle to control whether a trainable embedding layer is used or native
            one-hot representation for convolutional layers.

        embedding_dim_aa: int
            Learned latent dimensionality of amino-acids.

        embedding_dim_genes: int
            Learned latent dimensionality of VDJ genes

        embedding_dim_hla: int
            Learned latent dimensionality of HLA

        use_only_seq: bool
            To only use sequence feaures, set to True.

        use_only_gene: bool
            To only use gene-usage features, set to True.

        use_only_hla: bool
            To only use hla feaures, set to True.

        size_of_net: list or str
            The convolutional layers of this network have 3 layers for which the use can
            modify the number of neurons per layer. The user can either specify the size of the network
            with the following options:
                - small == [12,32,64] neurons for the 3 respective layers
                - medium == [32,64,128] neurons for the 3 respective layers
                - large == [64,128,256] neurons for the 3 respective layers
                - custom, where the user supplies a list with the number of nuerons for the respective layers
                    i.e. [3,3,3] would have 3 neurons for all 3 layers.
                    One can also adjust the number of layers for the convolutional stack by changing the length of
                    this list. [3,3,3] = 3 layers, [3,3,3,3] = 4 layers.

        latent_alpha: float
            Penalty coefficient for latent loss. This value changes the degree of latent regularization on
            the VAE.

        sparsity_alpha: float
            When training an autoencoder, the number of latent nodes required to model the underlying distribution
            of the data is often arrived to by trial and error and tuning this hyperparameter. In many cases, by using
            too many latent nodes, one my fit the distribution but downstream analysis tasks may be computationally
            burdensome with i.e. 256 latent features. Additionally, there can be a high level of colinearlity among
            these latent features. In our implemnetation of VAE, we introduce this concept of a sparsity constraint which
            turns off latent nodes in a soft fashion throughout straining and acts as another mode of regularization to
            find the minimal number of latent features to model the underlying distribution. Following training, one can
            set the var_explained parameter to select the number of latent nodes required  to cover X percent variation
            explained akin to PCA analysis. This results in a lower dimensional space and more linearly indepeendent
            latent space. Good starting value is 1.0.

        var_explained: float (0-1.0)
            Following training, one can select the number of latent features that explain N% of the variance in the
            data. The output of the method will be the features in order of the explained variance.

        graph_seed: int
            For deterministic initialization of weights of the graph, set this to value of choice.


        Training Parameters

        batch_size: int
            Size of batch to be used for each training iteration of the net.

        epochs_min: int
            The minimum number of epochs to train the autoencoder.

        stop_criterion: float
            Minimum percent decrease in determined interval (below) to continue
            training. Used as early stopping criterion.

        stop_criterion_window: int
            The window of data to apply the stopping criterion.

        accuracy_min: float
            Minimum reconstruction accuracy before terminating training.

        suppress_output: bool
            To suppress command line output with training statisitcs, set to True.

        split_seed: int
            For deterministic batching of data during training, one can set this parameter to value of choice.

        Load_Prev_Data: bool
            Load previous feature data from prior training.


        Returns

        self.features: array
            An array that contains n x latent_dim containing features for all sequences

        self.explained_variance_ : array
            The explained variance for the N number of latent features in order of descending value.

        self.explained_variance_ratio_ : array
            The explained variance ratio for the N number of latent features in order of descending value.

        ---------------------------------------

        """

        if Load_Prev_Data is False:
            GO = graph_object()
            GO.size_of_net = size_of_net
            GO.embedding_dim_genes = embedding_dim_genes
            GO.embedding_dim_aa = embedding_dim_aa
            GO.embedding_dim_hla = embedding_dim_hla
            GO.l2_reg = 0.0

            graph_model_AE = tf.Graph()
            with graph_model_AE.device(self.device):
                with graph_model_AE.as_default():
                    if graph_seed is not None:
                        tf.compat.v1.set_random_seed(graph_seed)

                    GO.net = 'ae'
                    if self.use_w:
                        GO.w = tf.compat.v1.placeholder(tf.float32, shape=[None])
                    GO.Features = Conv_Model(GO, self, trainable_embedding, kernel, use_only_seq, use_only_gene,use_only_hla)
                    fc = tf.compat.v1.layers.dense(GO.Features, 256)
                    fc = tf.compat.v1.layers.dense(fc, latent_dim)
                    z_w = tf.compat.v1.get_variable(name='z_w',shape=[latent_dim,latent_dim])
                    z_mean = tf.matmul(fc,z_w)
                    z_mean = tf.identity(z_mean,'z_mean')
                    z_log_var = tf.compat.v1.layers.dense(fc, latent_dim, activation=tf.nn.softplus, name='z_log_var')
                    latent_cost = Latent_Loss(z_log_var,z_mean,alpha=latent_alpha)

                    z = z_mean + tf.exp(z_log_var / 2) * tf.random.normal(tf.shape(input=z_mean), 0.0, 1.0, dtype=tf.float32)
                    z = tf.identity(z, name='z')

                    fc_up = tf.compat.v1.layers.dense(z, 128)
                    fc_up = tf.compat.v1.layers.dense(fc_up, 256)
                    fc_up_flat = fc_up
                    fc_up = tf.reshape(fc_up, shape=[-1, 1, 4, 64])

                    seq_losses = []
                    seq_accuracies = []
                    if self.use_beta:
                        upsample1_beta = tf.compat.v1.layers.conv2d_transpose(fc_up, 128, (1, 3), (1, 2), activation=tf.nn.relu)
                        upsample2_beta = tf.compat.v1.layers.conv2d_transpose(upsample1_beta, 64, (1, 3), (1, 2), activation=tf.nn.relu)
                        kr, str = determine_kr_str(upsample2_beta, GO, self)

                        if trainable_embedding is True:
                            #upsample3_beta = tf.layers.conv2d_transpose(upsample2_beta, GO.embedding_dim_aa, (1, 4),(1, 2), activation=tf.nn.relu)
                            upsample3_beta = tf.compat.v1.layers.conv2d_transpose(upsample2_beta, GO.embedding_dim_aa, (1, kr),(1, str), activation=tf.nn.relu)
                            upsample3_beta = upsample3_beta[:,:,0:self.max_length,:]

                            embedding_layer_seq_back = tf.transpose(a=GO.embedding_layer_seq, perm=(0, 1, 3, 2))
                            logits_AE_beta = tf.squeeze(tf.tensordot(upsample3_beta, embedding_layer_seq_back, axes=(3, 2)),axis=(3, 4), name='logits')
                        else:
                            logits_AE_beta = tf.compat.v1.layers.conv2d_transpose(upsample2_beta, 21, (1, kr),(1, str), activation=tf.nn.relu)
                            logits_AE_beta = logits_AE_beta[:,:,0:self.max_length,:]

                        recon_cost_beta = Recon_Loss(GO.X_Seq_beta, logits_AE_beta)
                        seq_losses.append(recon_cost_beta)

                        predicted_beta = tf.squeeze(tf.argmax(input=logits_AE_beta, axis=3), axis=1)
                        actual_ae_beta = tf.squeeze(GO.X_Seq_beta, axis=1)
                        w = tf.cast(tf.squeeze(tf.greater(GO.X_Seq_beta, 0), 1), tf.float32)
                        correct_ae_beta = tf.reduce_sum(input_tensor=w * tf.cast(tf.equal(predicted_beta, actual_ae_beta), tf.float32),axis=1) / tf.reduce_sum(input_tensor=w, axis=1)

                        accuracy_beta = tf.reduce_mean(input_tensor=correct_ae_beta, axis=0)
                        seq_accuracies.append(accuracy_beta)

                    if self.use_alpha:
                        upsample1_alpha = tf.compat.v1.layers.conv2d_transpose(fc_up, 128, (1, 3), (1, 2), activation=tf.nn.relu)
                        upsample2_alpha = tf.compat.v1.layers.conv2d_transpose(upsample1_alpha, 64, (1, 3), (1, 2),activation=tf.nn.relu)
                        kr, str = determine_kr_str(upsample2_alpha, GO, self)

                        if trainable_embedding is True:
                            # upsample3_alpha = tf.layers.conv2d_transpose(upsample2_alpha, GO.embedding_dim_aa, (1, 4), (1, 2),activation=tf.nn.relu)
                            upsample3_alpha = tf.compat.v1.layers.conv2d_transpose(upsample2_alpha, GO.embedding_dim_aa, (1, kr), (1, str),activation=tf.nn.relu)
                            upsample3_alpha = upsample3_alpha[:,:,0:self.max_length,:]

                            embedding_layer_seq_back = tf.transpose(a=GO.embedding_layer_seq, perm=(0, 1, 3, 2))
                            logits_AE_alpha = tf.squeeze(tf.tensordot(upsample3_alpha, embedding_layer_seq_back, axes=(3, 2)),axis=(3, 4), name='logits')
                        else:
                            logits_AE_alpha = tf.compat.v1.layers.conv2d_transpose(upsample2_alpha, 21, (1, kr), (1, str),activation=tf.nn.relu)
                            logits_AE_alpha = logits_AE_alpha[:,:,0:self.max_length,:]

                        recon_cost_alpha = Recon_Loss(GO.X_Seq_alpha, logits_AE_alpha)
                        seq_losses.append(recon_cost_alpha)

                        predicted_alpha = tf.squeeze(tf.argmax(input=logits_AE_alpha, axis=3), axis=1)
                        actual_ae_alpha = tf.squeeze(GO.X_Seq_alpha, axis=1)
                        w = tf.cast(tf.squeeze(tf.greater(GO.X_Seq_alpha, 0), 1), tf.float32)
                        correct_ae_alpha = tf.reduce_sum(input_tensor=w * tf.cast(tf.equal(predicted_alpha, actual_ae_alpha), tf.float32), axis=1) / tf.reduce_sum(input_tensor=w, axis=1)
                        accuracy_alpha = tf.reduce_mean(input_tensor=correct_ae_alpha, axis=0)
                        seq_accuracies.append(accuracy_alpha)

                    hla_accuracies = []
                    hla_losses = []
                    if self.use_hla:
                        hla_loss, hla_acc = Get_HLA_Loss(fc_up_flat,GO.embedding_layer_hla,GO.X_hla)
                        hla_losses.append(hla_loss)
                        hla_accuracies.append(hla_acc)

                    gene_loss = []
                    gene_accuracies = []
                    if self.use_v_beta is True:
                        v_beta_loss,v_beta_acc = Get_Gene_Loss(fc_up_flat,GO.embedding_layer_v_beta,GO.X_v_beta_OH)
                        gene_loss.append(v_beta_loss)
                        gene_accuracies.append(v_beta_acc)

                    if self.use_d_beta is True:
                        d_beta_loss, d_beta_acc = Get_Gene_Loss(fc_up_flat,GO.embedding_layer_d_beta,GO.X_d_beta_OH)
                        gene_loss.append(d_beta_loss)
                        gene_accuracies.append(d_beta_acc)

                    if self.use_j_beta is True:
                        j_beta_loss,j_beta_acc = Get_Gene_Loss(fc_up_flat,GO.embedding_layer_j_beta,GO.X_j_beta_OH)
                        gene_loss.append(j_beta_loss)
                        gene_accuracies.append(j_beta_acc)

                    if self.use_v_alpha is True:
                        v_alpha_loss,v_alpha_acc = Get_Gene_Loss(fc_up_flat,GO.embedding_layer_v_alpha,GO.X_v_alpha_OH)
                        gene_loss.append(v_alpha_loss)
                        gene_accuracies.append(v_alpha_acc)

                    if self.use_j_alpha is True:
                        j_alpha_loss,j_alpha_acc = Get_Gene_Loss(fc_up_flat,GO.embedding_layer_j_alpha,GO.X_j_alpha_OH)
                        gene_loss.append(j_alpha_loss)
                        gene_accuracies.append(j_alpha_acc)

                    recon_losses = seq_losses + gene_loss + hla_losses
                    accuracies = seq_accuracies + gene_accuracies + hla_accuracies

                    if use_only_gene:
                        recon_losses = gene_loss
                        accuracies = gene_accuracies
                    if use_only_seq:
                        recon_losses = seq_losses
                        accuracies = seq_accuracies
                    if use_only_hla:
                        recon_losses = hla_losses
                        accuracies = hla_accuracies

                    temp = []
                    for l in recon_losses:
                        l = l[:,tf.newaxis]
                        temp.append(l)
                    recon_losses = temp
                    recon_losses = tf.concat(recon_losses,1)
                    if self.use_w:
                        recon_losses = GO.w[:,tf.newaxis]*recon_losses

                    recon_cost = tf.reduce_sum(input_tensor=recon_losses,axis=1)
                    recon_cost = tf.reduce_mean(input_tensor=recon_cost)

                    if self.use_w:
                        latent_cost = GO.w*latent_cost

                    total_cost = [recon_losses,latent_cost[:,tf.newaxis]]
                    total_cost = tf.concat(total_cost,1)
                    total_cost = tf.reduce_sum(input_tensor=total_cost,axis=1)
                    total_cost = tf.reduce_mean(input_tensor=total_cost)

                    num_acc = len(accuracies)
                    accuracy = 0
                    for a in accuracies:
                        accuracy += a
                    accuracy = accuracy/num_acc
                    latent_cost = tf.reduce_mean(input_tensor=latent_cost)

                    opt_ae = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_cost)

                    if sparsity_alpha is not None:
                        sparsity_cost = sparsity_loss(z_w,sparsity_alpha)
                        total_cost += sparsity_cost
                        opt_sparse = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(sparsity_cost,var_list=z_w)
                        opt_ae = tf.group(opt_ae,opt_sparse)
                        self.use_sparsity = True
                    else:
                        sparsity_cost = tf.Variable(0.0)

                    GO.saver = tf.compat.v1.train.Saver(max_to_keep=None)

            self._reset_models()
            tf.compat.v1.reset_default_graph()
            config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True

            with tf.compat.v1.Session(graph=graph_model_AE,config=config) as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                stop_check_list = []
                accuracy_list = []
                recon_loss = []
                train_loss = []
                latent_loss = []
                training = True
                e = 0
                while training:
                    iteration = 0
                    Vars = [self.X_Seq_alpha,self.X_Seq_beta,self.v_beta_num,self.d_beta_num,self.j_beta_num,
                            self.v_alpha_num,self.j_alpha_num,self.hla_data_seq_num,self.w]

                    if split_seed is not None:
                        np.random.seed(split_seed)

                    for vars in get_batches(Vars, batch_size=batch_size,random=True):
                        feed_dict = {}
                        if self.use_alpha is True:
                            feed_dict[GO.X_Seq_alpha] = vars[0]
                        if self.use_beta is True:
                            feed_dict[GO.X_Seq_beta] = vars[1]

                        if self.use_v_beta is True:
                            feed_dict[GO.X_v_beta] = vars[2]

                        if self.use_d_beta is True:
                            feed_dict[GO.X_d_beta] = vars[3]

                        if self.use_j_beta is True:
                            feed_dict[GO.X_j_beta] = vars[4]

                        if self.use_v_alpha is True:
                            feed_dict[GO.X_v_alpha] = vars[5]

                        if self.use_j_alpha is True:
                            feed_dict[GO.X_j_alpha] = vars[6]

                        if self.use_hla:
                            feed_dict[GO.X_hla] = vars[7]

                        if self.use_w:
                            feed_dict[GO.w] = vars[8]

                        train_loss_i, recon_loss_i, latent_loss_i, sparsity_loss_i, accuracy_i, _ = sess.run([total_cost, recon_cost, latent_cost, sparsity_cost, accuracy, opt_ae], feed_dict=feed_dict)
                        accuracy_list.append(accuracy_i)
                        recon_loss.append(recon_loss_i)
                        latent_loss.append(latent_loss_i)
                        train_loss.append(train_loss_i)

                        if suppress_output is False:
                            print("Epoch = {}, Iteration = {}".format(e,iteration),
                                  "Total Loss: {:.5f}:".format(train_loss_i),
                                  "Recon Loss: {:.5f}:".format(recon_loss_i),
                                  "Latent Loss: {:.5f}:".format(latent_loss_i),
                                  "Sparsity Loss: {:.5f}:".format(sparsity_loss_i),
                                  "Recon Accuracy: {:.5f}".format(accuracy_i))

                        if e >= epochs_min:
                            if accuracy_min is not None:
                                if np.mean(accuracy_list[-10:]) > accuracy_min:
                                    training = False
                                    break
                            else:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    stop_check_list.append(stop_check(train_loss,stop_criterion,stop_criterion_window))
                                    if np.sum(stop_check_list[-3:]) >= 3:
                                        training = False
                                        break
                        iteration += 1
                    e += 1

                features_list = []
                accuracy_list = []
                alpha_features_list = []
                alpha_indices_list = []
                beta_features_list = []
                beta_indices_list = []
                Vars = [self.X_Seq_alpha, self.X_Seq_beta, self.v_beta_num, self.d_beta_num, self.j_beta_num,
                        self.v_alpha_num, self.j_alpha_num,self.hla_data_seq_num,self.w]

                for vars in get_batches(Vars, batch_size=batch_size, random=False):
                    feed_dict = {}
                    if self.use_alpha is True:
                        feed_dict[GO.X_Seq_alpha] = vars[0]
                    if self.use_beta is True:
                        feed_dict[GO.X_Seq_beta] = vars[1]

                    if self.use_v_beta is True:
                        feed_dict[GO.X_v_beta] = vars[2]

                    if self.use_d_beta is True:
                        feed_dict[GO.X_d_beta] = vars[3]

                    if self.use_j_beta is True:
                        feed_dict[GO.X_j_beta] = vars[4]

                    if self.use_v_alpha is True:
                        feed_dict[GO.X_v_alpha] = vars[5]

                    if self.use_j_alpha is True:
                        feed_dict[GO.X_j_alpha] = vars[6]

                    if self.use_hla:
                        feed_dict[GO.X_hla] = vars[7]

                    if self.use_w:
                        feed_dict[GO.w] = vars[8]

                    get = z_mean
                    features_ind, accuracy_check = sess.run([get, accuracy], feed_dict=feed_dict)
                    features_list.append(features_ind)
                    accuracy_list.append(accuracy_check)

                    if self.use_alpha is True:
                        alpha_ft, alpha_i = sess.run([GO.alpha_out,GO.indices_alpha],feed_dict=feed_dict)
                        alpha_features_list.append(alpha_ft)
                        alpha_indices_list.append(alpha_i)

                    if self.use_beta is True:
                        beta_ft, beta_i = sess.run([GO.beta_out,GO.indices_beta],feed_dict=feed_dict)
                        beta_features_list.append(beta_ft)
                        beta_indices_list.append(beta_i)

                features = np.vstack(features_list)
                accuracy_list = np.hstack(accuracy_list)
                if self.use_alpha is True:
                    self.alpha_features = np.vstack(alpha_features_list)
                    self.alpha_indices = np.vstack(alpha_indices_list)

                if self.use_beta is True:
                    self.beta_features = np.vstack(beta_features_list)
                    self.beta_indices = np.vstack(beta_indices_list)

                self.kernel = kernel
                #
                if self.use_alpha is True:
                    var_save = [self.alpha_features, self.alpha_indices, self.alpha_sequences]
                    with open(os.path.join(self.Name, self.Name) + '_alpha_features.pkl', 'wb') as f:
                        pickle.dump(var_save, f)

                if self.use_beta is True:
                    var_save = [self.beta_features, self.beta_indices, self.beta_sequences]
                    with open(os.path.join(self.Name, self.Name) + '_beta_features.pkl', 'wb') as f:
                        pickle.dump(var_save, f)

                with open(os.path.join(self.Name, self.Name) + '_kernel.pkl', 'wb') as f:
                    pickle.dump(self.kernel, f)


                print('Reconstruction Accuracy: {:.5f}'.format(np.nanmean(accuracy_list)))

                embedding_layers = [GO.embedding_layer_v_alpha,GO.embedding_layer_j_alpha,
                                    GO.embedding_layer_v_beta,GO.embedding_layer_d_beta,
                                    GO.embedding_layer_j_beta]
                embedding_names = ['v_alpha','j_alpha','v_beta','d_beta','j_beta']
                name_keep = []
                embedding_keep = []
                for n,layer in zip(embedding_names,embedding_layers):
                    if layer is not None:
                        embedding_keep.append(layer.eval())
                        name_keep.append(n)

                embed_dict = dict(zip(name_keep,embedding_keep))

                # sort features by variance explained
                cov = np.cov(features.T)
                explained_variance = np.diag(cov)
                ind = np.flip(np.argsort(explained_variance))
                explained_variance = explained_variance[ind]
                explained_variance_ratio = explained_variance / np.sum(explained_variance)
                features = features[:, ind]

                if var_explained is not None:
                    features = features[:, 0:np.where(np.cumsum(explained_variance_ratio) > var_explained)[0][0] + 1]

                self.ind = ind[:features.shape[1]]
                #save model data and information for inference engine
                save_model_data(self,GO.saver,sess,name='VAE',get=z_mean)

            with open(os.path.join(self.Name,self.Name) + '_VAE_features.pkl', 'wb') as f:
                pickle.dump([features,embed_dict,explained_variance,explained_variance_ratio], f,protocol=4)

        else:
            with open(os.path.join(self.Name,self.Name) + '_VAE_features.pkl', 'rb') as f:
                features,embed_dict,explained_variance,explained_variance_ratio = pickle.load(f)

        self.features = features
        self.embed_dict = embed_dict
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance_ratio
        print('Training Done')

    def KNN_Sequence_Classifier(self,folds=5, k_values=list(range(1, 500, 25)), rep=5, plot_metrics=False, by_class=False,
                                plot_type='violin', metrics=['Recall', 'Precision', 'F1_Score', 'AUC'],
                                n_jobs=1,Load_Prev_Data=False):
        """
        K-Nearest Neighbor Sequence Classifier

        This method uses a K-Nearest Neighbor Classifier to assess the ability to predict a sequence
        label given its sequence features.The method returns AUC,Precision,Recall, and
        F1 Scores for all classes.

        Inputs
        ---------------------------------------

        folds: int
            Number of folds to train/test K-Nearest Classifier.

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

        n_jobs: int
            Number of workers to set for KNeighborsClassifier.

        Load_Prev_Data: bool
            To make new figures from old previously run analysis, set this value to True
            after running the method for the first time. This will load previous performance
            metrics from previous run.

        Returns

        self.KNN_Sequence_DF: Pandas dataframe
            Dataframe with all metrics of performance organized by the class label,
            metric (i.e. AUC), k-value (from k-nearest neighbors), and the value of the
            performance metric.
        ---------------------------------------

        """

        if Load_Prev_Data is False:
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
                try:
                    classes, metric, value, k_l = KNN(distances, self.class_id, k=k, metrics=metrics,
                                                      folds=folds,n_jobs=n_jobs)
                    metric_list.extend(metric)
                    val_list.extend(value)
                    class_list.extend(classes)
                    k_list.extend(k_l)
                except:
                    continue

            df_out = pd.DataFrame()
            df_out['Classes'] = class_list
            df_out['Metric'] = metric_list
            df_out['Value'] = val_list
            df_out['k'] = k_list

            with open(os.path.join(self.Name,'knn_seq.pkl'),'wb') as f:
                pickle.dump(df_out,f,protocol=4)
        else:
            with open(os.path.join(self.Name,'knn_seq.pkl'),'rb') as f:
                df_out = pickle.load(f)

        self.KNN_Sequence_DF = df_out

        if plot_metrics is True:
            if by_class is True:
                sns.catplot(data=df_out, x='Metric', y='Value', hue='Classes', kind=plot_type)
            else:
                sns.catplot(data=df_out, x='Metric', y='Value', kind=plot_type)

    def KNN_Repertoire_Classifier(self,folds=5, distance_metric='KL', sample=None, n_jobs=1, plot_metrics=False,
                                  plot_type='violin', by_class=False, Load_Prev_Data=False,
                                  metrics=['Recall', 'Precision', 'F1_Score', 'AUC']):
        """
        K-Nearest Neighbor Repertoire Classifier

        This method uses a K-Nearest Neighbor Classifier to assess the ability to predict a repertoire
        label given the structural distribution of the repertoire.The method returns AUC,Precision,Recall, and
        F1 Scores for all classes.

        Inputs
        ---------------------------------------

        folds: int
            Number of folds to train/test K-Nearest Classifier.

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
            self.Cluster(sample=sample, n_jobs=n_jobs)
            prop = self.Cluster_Frequencies
            with open(os.path.join(self.Name, 'KNN_sample.pkl'), 'wb') as f:
                pickle.dump(prop, f, protocol=4)
        else:
            with open(os.path.join(self.Name, 'KNN_sample.pkl'), 'rb') as f:
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
            labels.append(self.class_id[np.where(self.sample_id == i)[0][0]])

        class_list = []
        k_list = []
        metric_list = []
        val_list = []
        for k in k_values:
            try:
                classes, metric, value, k_l = KNN_samples(pairwise_distances, labels, k=k, metrics=metrics,
                                                          folds=folds,n_jobs=n_jobs)
                metric_list.extend(metric)
                val_list.extend(value)
                class_list.extend(classes)
                k_list.extend(k_l)
            except:
                continue

        df_out = pd.DataFrame()
        df_out['Classes'] = class_list
        df_out['Metric'] = metric_list
        df_out['Value'] = val_list
        df_out['k'] = k_list

        self.KNN_Repertoire_DF = df_out

        if plot_metrics is True:
            if by_class is True:
                sns.catplot(data=df_out, x='Metric', y='Value', hue='Classes', kind=plot_type)
            else:
                sns.catplot(data=df_out, x='Metric', y='Value', kind=plot_type)

class DeepTCR_S_base(DeepTCR_base,feature_analytics_class,vis_class):
    def AUC_Curve(self,by=None,filename='AUC.tif',title=None,title_font=None,plot=True,diag_line=True,
                  xtick_size = None, ytick_size=None, xlabel_size = None, ylabel_size=None,
                  legend_font_size=None,frameon=True,legend_loc = 'lower right',
                  figsize=None):
        """
        AUC Curve for both Sequence and Repertoire/Sample Classifiers

        Inputs
        ---------------------------------------
        by: str
            To show AUC curve for only one class, set this parameter
            to the name of the class label one wants to plot.

        filename: str
            Filename to save tif file of AUC curve.

        title: str
            Optional Title to put on ROC Curve.

        title_font: int
            Optional font size for title

        plot: bool
            To suppress plotting and just save the data/figure, set to False.

        diag_line: bool
            To plot the line/diagonal of y=x defining no predictive power, set to True.
            To remove from plot, set to False.

        xtick_size: float
            Size of xticks

        ytick_size: float
            Size of yticks

        xlabel_size: float
            Size of xlabel

        ylabel_size: float
            Size of ylabel

        legend_font_size: float
            Size of legend

        frameon: bool
            Whether to show frame around legend.

        figsize: tuple
            To change the default size of the figure, set this to size of figure (i.e. - (10,10) )

        Returns

        self.AUC_DF: Pandas Dataframe
            AUC scores are returned for each class.

        In addition to plotting the ROC Curve, the AUC's are saved
        to a csv file in the results directory called 'AUC.csv'

        ---------------------------------------

        """
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
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        if by is None:
            for ii, class_name in enumerate(self.lb.classes_, 0):
                roc_score = roc_auc_score(y_test[:, ii], y_pred[:,ii])
                classes.append(class_name)
                auc_scores.append(roc_score)
                fpr, tpr, _ = roc_curve(y_test[:, ii], y_pred[:,ii])
                plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (class_name, roc_score))
        else:
            class_name = by
            ii = self.lb.transform([by])[0]
            roc_score = roc_auc_score(y_test[:, ii], y_pred[:, ii])
            auc_scores.append(roc_score)
            classes.append(class_name)
            fpr, tpr, _ = roc_curve(y_test[:, ii], y_pred[:, ii])
            plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (class_name, roc_score))

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
        Spearman's Rank Correlation Coefficient Plot

        In the case one is doing a regression-based model for the sequence classiifer,
        one can plot the predicted vs actual labeled value with this method. The method
        returns a plot for the regression and a value of the correlation coefficient.

        Inputs
        ---------------------------------------
        s: int
            size of points for scatterplot

        kde: bool
            To do a kernel density estimation per point and plot this as a color-scheme,
            set to True. Warning: this option will take longer to run.

        title: str
            Title for the plot.

        Returns
        ---------------------------------------
        corr: float
            Spearman's Rank Correlation Coefficient

        ax: matplotlib axis
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
        Identify most highly predicted sequences for each class and corresponding motifs.

        This method allows the user to query which sequences were most predicted to belong to a given class along
        with the motifs that were learned for these representative sequences.
        Of note, this method only reports sequences that were in the test set so as not to return highly predicted
        sequences that were over-fit in the training set. To obtain the highest predicted sequences in all the data,
        run a K-fold cross-validation or Monte-Carlo cross-validation before running this method. In this way,
        the predicted probability will have been assigned to a sequence only when it was in the independent test set.

        In the case of a regression task, the representative sequences for the 'high' and 'low' values for the regression
        model are returned in the Rep_Seq Dict.

        This method will also determine motifs the network has learned that are highly associated with the label through
        multi-nomial linear regression and creates seq logos and fasta files in the results folder. Within a folder
        for a given class, the motifs are sorted by their linear coefficient. The coefficient is in the file name
        (i.e. 0_0.125_feature_2.eps reflects the the 0th highest feature with a coefficient of 0.125.


        Inputs
        ---------------------------------------

        top_seq: int
            The number of top sequences to show for each class.

        motif_seq: int
            The number of sequences to use to generate each motif. The more sequences, the possibly more noisy
            the seq_logo will be.

        make_seq_logos: bool
            In order to make seq logos for visualization of enriched motifs, set this to True. Whether this is set to
            True or not, the fast files that define enriched motifs will still be saved.

        color_scheme: str
            color scheme to use for LogoMaker.
            options are:
                weblogo_protein
                skylign_protein
                dmslogo_charge
                dmslogo_funcgroup
                hydrophobicity
                chemistry
                charge
                NajafabadiEtAl2017

        logo_file_format: str
            The type of image file one wants to save the seqlogo as. Default is vector-based format (.eps)

        Returns

        self.Rep_Seq: dictionary of dataframes
            This dictionary of dataframes holds for each class the top sequences and their respective
            probabiltiies for all classes. These dataframes can also be found in the results folder under Rep_Sequences.

        self.Rep_Seq_Features_(alpha/beta): dataframe
            This dataframe holds information for which features were associated by a multinomial linear model
            to the predicted probabilities of the neural network. The values in this dataframe are the linear model
            coefficients. This allows one to see which features were associated with the output of the trained
            neural network. These are also the same values that are on the motif seqlogo files in the results folder.

        Furthermore, the motifs are written in the results directory underneath the Motifs folder. To find the beta
        motifs for a given class, look under Motifs/beta/class_name/. These fasta/logo files are labeled by the linear
        coefficient of that given feature for that given class followed by the number name of the feature. These fasta files
        can then be visualized via weblogos at the following site: "https://weblogo.berkeley.edu/logo.cgi" or are present
        in the folder for direct visualization.

        ---------------------------------------


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

    def _residue(self,alpha_sequence,beta_sequence,v_beta,d_beta,j_beta,v_alpha,j_alpha,hla,
                 p,batch_size,models):
        if self.model_type == 'SS':
            inf_func = self.Sequence_Inference
        elif self.model_type == 'WF':
            inf_func = self.Sample_Inference

        df_alpha = pd.DataFrame()
        df_beta = pd.DataFrame()
        if alpha_sequence is not None:
            alpha_list,pos,ref,alt = make_seq_list(alpha_sequence,ref= list(self.aa_idx.keys()))
            len_list = len(alpha_list)

            if beta_sequence is None:
                beta_sequences = None
            else:
                beta_sequences = np.array([beta_sequence] * len_list)

            if v_beta is None:
                v_beta = None
            else:
                v_beta = np.array([v_beta]*len_list)

            if d_beta is None:
                d_beta = None
            else:
                d_beta = np.array([d_beta]*len_list)

            if j_beta is None:
                j_beta = None
            else:
                j_beta =  np.array([j_beta]*len_list)

            if v_alpha is None:
                v_alpha = None
            else:
                v_alpha =  np.array([v_alpha]*len_list)

            if j_alpha is None:
                j_alpha = None
            else:
                j_alpha = np.array([j_alpha]*len_list)

            if hla is None:
                hla = None
            else:
                hla = np.array([hla]*len_list)

            out = inf_func(beta_sequences = beta_sequences,
                                 alpha_sequences = np.array(alpha_list),
                                 v_beta = v_beta,
                                 d_beta = d_beta,
                                 j_beta = j_beta,
                                 v_alpha = v_alpha,
                                 j_alpha = j_alpha,
                                 p = p,
                                 hla = hla,
                                 batch_size = batch_size,
                                 models=models)

            df_alpha['alpha'] = alpha_list
            df_alpha['pos'] = pos
            df_alpha['ref'] = ref
            df_alpha['alt'] = alt
            if self.regression:
                df_alpha['high'] = out[:,0]
            else:
                for ii in range(out.shape[1]):
                    df_alpha[self.lb.inverse_transform([ii])[0]] = out[:,ii]

        if beta_sequence is not None:
            beta_list,pos,ref,alt = make_seq_list(beta_sequence,ref= list(self.aa_idx.keys()))
            len_list = len(beta_list)
            if alpha_sequence is None:
                alpha_sequences = None
            else:
                alpha_sequences = np.array([alpha_sequence] * len_list)

            if v_beta is None:
                v_beta = None
            else:
                v_beta = np.array([v_beta]*len_list)

            if d_beta is None:
                d_beta = None
            else:
                d_beta = np.array([d_beta]*len_list)

            if j_beta is None:
                j_beta = None
            else:
                j_beta =  np.array([j_beta]*len_list)

            if v_alpha is None:
                v_alpha = None
            else:
                v_alpha =  np.array([v_alpha]*len_list)

            if j_alpha is None:
                j_alpha = None
            else:
                j_alpha = np.array([j_alpha]*len_list)

            if hla is None:
                hla = None
            else:
                hla = np.array([hla]*len_list)

            out = inf_func(beta_sequences = np.array(beta_list),
                                 alpha_sequences = alpha_sequences,
                                 v_beta = v_beta,
                                 d_beta = d_beta,
                                 j_beta = j_beta,
                                 v_alpha = v_alpha,
                                 j_alpha = j_alpha,
                                 p = p,
                                 hla = hla,
                                 batch_size = batch_size,
                                 models=models)

            df_beta['beta'] = beta_list
            df_beta['pos'] = pos
            df_beta['ref'] = ref
            df_beta['alt'] = alt
            if self.regression:
                df_beta['high'] = out[:,0]
            else:
                for ii in range(out.shape[1]):
                    df_beta[self.lb.inverse_transform([ii])[0]] = out[:,ii]

            return df_alpha,df_beta

    def Residue_Sensitivity_Logo(self,alpha_sequences=None, beta_sequences=None, v_beta=None, d_beta=None, j_beta=None,
                                v_alpha=None, j_alpha=None, hla=None,p=None, batch_size=10000,models=None,
                                 figsize=(10,8),low_color='red',medium_color='white',high_color='blue',
                                    font_name='serif',class_sel=None,
                                 cmap=None,min_size=0.0,edgecolor='black',edgewidth=0.25,background_color='white',
                                 Load_Prev_Data=False,norm_to_seq=True):
        """
        Create Residue Sensitivity Logos

        This method allows the user to create Residue Sensitivity Logos where a set of provided sequences is perturbed
        to assess for position of the CDR3 sequence that if altered, would change the predicted specificity or affinity
        of the sequence (depending on whether training classification or regression task).

        Residue Sensitivity Logos can be created from any supervised model (including sequence and repertoire classifiers).
        Following the training of one of these models, one can feed into this method an cdr3 sequence defined by all/any
        of alpha/beta cdr3 sequence, V/D/J gene usage, and HLA context within which the TCR was seen.

        The output is a logo created by LogoMaker where the size of the character denotes how sensitive this position
        is to perturbation and color denotes the consequences of changes at this site. As default, red coloration means
        changes at this site would generally decrease the predicted value and blue coloration means changes at this site
        would increase the predicted value.

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

        hla: ndarray of tuples/arrays
            To input the hla context for each sequence fed into DeepTCR, this will need to formatted
            as an ndarray that is (N,) for each sequence where each entry is a tuple/array of strings referring
            to the alleles seen for that sequence.
                ('A*01:01', 'A*11:01', 'B*35:01', 'B*35:02', 'C*04:01')

        p: multiprocessing pool object
            a pre-formed pool object can be passed to method for multiprocessing tasks.

        batch_size: int
            Batch size for inference.

        models: list
            In the case of the supervised sequence classifier, if several models were trained (via MC or Kfold crossvals),
            one can specify which ones to use for inference. Otherwise, thie method uses all trained models found in
            Name/models/ in an ensemble fashion. The method will output of the average of all models as well as the
            distribution of outputs for the user.

        figsize: tuple
            This specifies the dimensions of the logo.

        low_color: str
            The color to use when changes at this site would largely result in decreased prediction values.

        medium_color: str
            The color to use when changes at this site would result in either decreased or inreased prediction values.

        high_color: str
            The color to use when changes at this site would result in increased prediction values.

        font_name: str
            The font to use for LogoMaker.

        class_sel: str
            In the case of a model being trained in a multi-class fashion, one must select which class to make the
            logo for.

        cmap: matplotlib cmap
            One can alsp provide custom cmap for logomaker that will be used to denote changes at sites that result
            in increased of decreased prediction values.

        min_size: float (0.0 - 1.0)
            Some residues may have such little change with any perturbation that the character would be difficult to
            read. To set a minimum size for a residue, one can set this parameter to a value between 0 and 1.

        edgecolor: str
            The color of the edge of the characters of the logo.

        edgewidth: float
            The thickness of the edge of the characters.

        background_color: str
            The background color of the logo.

        norm_to_seq: bool
            When determining the color intensity of the logo, one can choose to normalize the value to just characters
            in that sequence (True) or one can choose to normalize to all characters in the sequences provdied (False).

        Load_Prev_Data: bool
            Since the first part of the method runs a time-intensive step to get all the predictions for all perturbations
            at all residue sites, we've incorporated a paramter which can be set to True following running the method once
            in order to adjust the visual aspects of the plot. Therefore, one should run this method first setting this parameter
            to False (it's default setting) but then switch to True and run again with different visualization parameters
            (i.e. figsize, etc).

        Returns
        ---------------------------------------
        (fig,ax) - the matplotlib figure and axis/axes.
        """

        self.model_type, get = load_model_data(self)
        if Load_Prev_Data is False:
            if p is None:
                p = Pool(40)

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
                df_alpha,df_beta = self._residue(alpha_sequences[i],beta_sequences[i],
                              v_beta[i],d_beta[i],j_beta[i],
                              v_alpha[i],j_alpha[i],hla[i],
                              p,batch_size,models)
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
                p.close()
                p.join()

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

class DeepTCR_SS(DeepTCR_S_base):
    def Get_Train_Valid_Test(self,test_size=0.25,LOO=None,split_by_sample=False,combine_train_valid=False):
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
            Number of sequences to leave-out in Leave-One-Out Cross-Validation. For example,
            when set to 20, 20 sequences will be left out for the validation set and 20 samples will be left
            out for the test set.

        split_by_sample: int
            In the case one wants to train the single sequence classifer but not to mix the train/test
            sets with sequences from different samples, one can set this parameter to True to do the train/test
            splits by sample.

        combine_train_valid: bool
            To combine the training and validation partitions into one which will be used for training
            and updating the model parameters, set this to True. This will also set the validation partition
            to the test partition. In other words, new train set becomes (original train + original valid) and then
            new valid = original test partition, new test = original test partition. Therefore, if setting this parameter
            to True, change one of the training parameters to set the stop training criterion (i.e. train_loss_min)
            to stop training based on the train set. If one does not chanage the stop training criterion, the decision of
            when to stop training will be based on the test data (which is considered a form of over-fitting).

        """
        Vars = [self.X_Seq_alpha,self.X_Seq_beta,self.alpha_sequences,self.beta_sequences,self.sample_id,self.class_id,self.seq_index,
                self.v_beta_num,self.d_beta_num,self.j_beta_num,self.v_alpha_num,self.j_alpha_num,
                self.v_beta,self.d_beta,self.j_beta,self.v_alpha,self.j_alpha,self.hla_data_seq_num]

        var_names = ['X_Seq_alpha','X_Seq_beta','alpha_sequences','beta_sequences','sample_id','class_id','seq_index',
                     'v_beta_num','d_beta_num','j_beta_num','v_alpha_num','j_alpha_num','v_beta','d_beta','j_beta',
                     'v_alpha','j_alpha','hla_data_seq_num']

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

    def _build(self,kernel = 5,trainable_embedding = True,embedding_dim_aa = 64, embedding_dim_genes = 48, embedding_dim_hla = 12,
               num_fc_layers = 0, units_fc = 12,weight_by_class = False, class_weights = None,
               use_only_seq = False, use_only_gene = False, use_only_hla = False, size_of_net = 'medium',graph_seed = None,
               drop_out_rate=0.0,multisample_dropout = False, multisample_dropout_rate = 0.50, multisample_dropout_num_masks = 64,
               batch_size = 1000, epochs_min = 10, stop_criterion = 0.001, stop_criterion_window = 10,
               accuracy_min = None, train_loss_min = None, hinge_loss_t = 0.0, convergence = 'validation', learning_rate = 0.001, suppress_output = False):


        graph_model = tf.Graph()
        GO = graph_object()
        GO.on_graph_clustering=False
        GO.size_of_net = size_of_net
        GO.embedding_dim_genes = embedding_dim_genes
        GO.embedding_dim_aa = embedding_dim_aa
        GO.embedding_dim_hla = embedding_dim_hla
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
                GO.Features = Conv_Model(GO,self,trainable_embedding,kernel,use_only_seq,use_only_gene,use_only_hla,
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
                self.kernel = kernel

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

            Get_Seq_Features_Indices(self,batch_size,GO,sess)
            self.features = Get_Latent_Features(self,batch_size,GO,sess)

            idx_base = np.asarray(range(len(self.sample_id)))
            self.train_idx = np.isin(idx_base,self.train[self.var_dict['seq_index']])
            self.valid_idx = np.isin(idx_base,self.valid[self.var_dict['seq_index']])
            self.test_idx = np.isin(idx_base,self.test[self.var_dict['seq_index']])

            if hasattr(self,'predicted'):
                self.predicted[self.test[self.var_dict['seq_index']]] += self.y_pred

            #
            if self.use_alpha is True:
                var_save = [self.alpha_features,self.alpha_indices,self.alpha_sequences]
                with open(os.path.join(self.Name, self.Name) + '_alpha_features.pkl', 'wb') as f:
                    pickle.dump(var_save, f)

            if self.use_beta is True:
                var_save = [self.beta_features,self.beta_indices,self.beta_sequences]
                with open(os.path.join(self.Name, self.Name) + '_beta_features.pkl', 'wb') as f:
                    pickle.dump(var_save, f)

            with open(os.path.join(self.Name, self.Name) + '_kernel.pkl', 'wb') as f:
                pickle.dump(self.kernel, f)

            print('Done Training')
            # save model data and information for inference engine
            save_model_data(self, GO.saver, sess, name='SS', get=GO.predicted,iteration=iteration)

    def Train(self,kernel = 5,trainable_embedding = True,embedding_dim_aa = 64, embedding_dim_genes = 48, embedding_dim_hla = 12,
               num_fc_layers = 0, units_fc = 12,weight_by_class = False, class_weights = None,
               use_only_seq = False, use_only_gene = False, use_only_hla = False, size_of_net = 'medium',graph_seed = None,
               drop_out_rate=0.0,multisample_dropout = False, multisample_dropout_rate = 0.50, multisample_dropout_num_masks = 64,
               batch_size = 1000, epochs_min = 10, stop_criterion = 0.001, stop_criterion_window = 10,
               accuracy_min = None, train_loss_min = None, hinge_loss_t = 0.0, convergence = 'validation', learning_rate = 0.001, suppress_output = False,
                batch_seed = None):
        """
        Train Single-Sequence Classifier

        This method trains the network and saves features values at the
        end of training for motif analysis.

        Inputs
        ---------------------------------------
        Model Parameters

        kernel: int
            Size of convolutional kernel for first layer of convolutions.

        trainable_embedding; bool
            Toggle to control whether a trainable embedding layer is used or native
            one-hot representation for convolutional layers.

        embedding_dim_aa: int
            Learned latent dimensionality of amino-acids.

        embedding_dim_genes: int
            Learned latent dimensionality of VDJ genes

        embedding_dim_hla: int
            Learned latent dimensionality of HLA

        num_fc_layers: int
            Number of fully connected layers following convolutional layer.

        units_fc: int
            Number of nodes per fully-connected layers following convolutional layer.

        weight_by_class: bool
            Option to weight loss by the inverse of the class frequency. Useful for
            unbalanced classes.

        class_weights: dict
            In order to specify custom weights for each class during training, one
            can provide a dictionary with these weights.
                i.e. {'A':1.0,'B':2.0'}

        use_only_seq: bool
            To only use sequence feaures, set to True. This will turn off features learned
            from gene usage.

        use_only_gene: bool
            To only use gene-usage features, set to True. This will turn off features from
            the sequences.

        use_only_hla: bool
            To only use hla feaures, set to True.

        size_of_net: list or str
            The convolutional layers of this network have 3 layers for which the use can
            modify the number of neurons per layer. The user can either specify the size of the network
            with the following options:
                - small == [12,32,64] neurons for the 3 respective layers
                - medium == [32,64,128] neurons for the 3 respective layers
                - large == [64,128,256] neurons for the 3 respective layers
                - custom, where the user supplies a list with the number of nuerons for the respective layers
                    i.e. [3,3,3] would have 3 neurons for all 3 layers.
                    One can also adjust the number of layers for the convolutional stack by changing the length of
                    this list. [3,3,3] = 3 layers, [3,3,3,3] = 4 layers.

        graph_seed: int
            For deterministic initialization of weights of the graph, set this to value of choice.

        drop_out_rate: float
            drop out rate for fully connected layers

        The following parameters are used to implement Multi-Sample Dropout at the final layer of the model as described in
        "Multi-Sample Dropout for Accelerated Training and Better Generalization"
        https://arxiv.org/abs/1905.09788
        This method has been shown to improve generalization of deep neural networks as well as inmprove convergence.

        multisample_dropout: bool
            Set this parameter to True to implement this method.

         multisample_dropout_rate: float
            The dropout rate for this multi-sample dropout layer.

         multisample_dropout_num_masks: int
            The number of masks to sample from for the Multi-Sample Dropout layer.


        Training Parameters

        batch_size: int
            Size of batch to be used for each training iteration of the net.

        epochs_min: int
            Minimum number of epochs for training neural network.

        stop_criterion: float
            Minimum percent decrease in determined interval (below) to continue
            training. Used as early stopping criterion.

        stop_criterion_window: int
            The window of data to apply the stopping criterion.

        accuracy_min: float
            Optional parameter to allow alternative training strategy until minimum
            training accuracy is achieved, at which point, training ceases.

        train_loss_min: float
            Optional parameter to allow alternative training strategy until minimum
            training loss is achieved, at which point, training ceases.

        hinge_loss_t: float
            The per sequence loss minimum at which the loss of that sequence is not used
            to penalize the model anymore. In other words, once a per sequence loss has hit
            this value, it gets set to 0.0.

        convergence: str
            This parameter determines which loss to assess the convergence criteria on.
            Options are 'validation' or 'training'. This is useful in the case one wants
            to change the convergence criteria on the training data when the training and validation
            partitions have been combined and used to training the model.

        learning_rate: float
            The learning rate for training the neural network. Making this value larger will
            increase the rate of convergence but can introduce instability into training. For most,
            altering this value will not be necessary.

        suppress_output: bool
            To suppress command line output with training statisitcs, set to True.

        batch_seed: int
            For deterministic batching during training, set this value to an integer of choice.

        Returns
        ---------------------------------------

        """
        self._reset_models()
        self._build(kernel,trainable_embedding,embedding_dim_aa, embedding_dim_genes, embedding_dim_hla,
               num_fc_layers, units_fc,weight_by_class, class_weights,
               use_only_seq, use_only_gene, use_only_hla, size_of_net,graph_seed,
               drop_out_rate,multisample_dropout, multisample_dropout_rate, multisample_dropout_num_masks,
               batch_size, epochs_min, stop_criterion, stop_criterion_window,
               accuracy_min, train_loss_min, hinge_loss_t, convergence, learning_rate, suppress_output)
        self._train(batch_seed=batch_seed,iteration=0)

    def Monte_Carlo_CrossVal(self,folds=5,test_size=0.25,LOO=None,split_by_sample=False,combine_train_valid=False,seeds=None,
                             kernel=5, trainable_embedding=True, embedding_dim_aa=64, embedding_dim_genes=48, embedding_dim_hla=12,
                             num_fc_layers=0, units_fc=12, weight_by_class=False, class_weights=None,
                             use_only_seq=False, use_only_gene=False, use_only_hla=False, size_of_net='medium', graph_seed=None,
                             drop_out_rate=0.0, multisample_dropout=False, multisample_dropout_rate=0.50, multisample_dropout_num_masks=64,
                             batch_size=1000, epochs_min=10, stop_criterion=0.001, stop_criterion_window=10,
                             accuracy_min=None, train_loss_min=None, hinge_loss_t=0.0, convergence='validation', learning_rate=0.001, suppress_output=False,
                             batch_seed=None):

        '''
        Monte Carlo Cross-Validation for Single-Sequence Classifier

        If the number of sequences is small but training the single-sequence classifier, one
        can use Monte Carlo Cross Validation to train a number of iterations before assessing
        predictive performance.After this method is run, the AUC_Curve method can be run to
        assess the overall performance.

        Inputs
        ---------------------------------------
        Monte-Carlo Parameters

        folds: int
            Number of iterations for Cross-Validation

        test_size: float
            Fraction of sample to be used for valid and test set.

        LOO: int
            Number of sequences to leave-out in Leave-One-Out Cross-Validation. For example,
            when set to 20, 20 sequences will be left out for the validation set and 20 samples will be left
            out for the test set.

        split_by_sample: int
            In the case one wants to train the single sequence classifer but not to mix the train/test
            sets with sequences from different samples, one can set this parameter to True to do the train/test
            splits by sample.

        combine_train_valid: bool
            To combine the training and validation partitions into one which will be used for training
            and updating the model parameters, set this to True. This will also set the validation partition
            to the test partition. In other words, new train set becomes (original train + original valid) and then
            new valid = original test partition, new test = original test partition. Therefore, if setting this parameter
            to True, change one of the training parameters to set the stop training criterion (i.e. train_loss_min)
            to stop training based on the train set. If one does not chanage the stop training criterion, the decision of
            when to stop training will be based on the test data (which is considered a form of over-fitting).

        seeds: nd.array
            In order to set a deterministic train/test split over the Monte-Carlo Simulations, one can provide an array
            of seeds for each MC simulation. This will result in the same train/test split over the N MC simulations.
            This parameter, if provided, should have the same size of the value of folds.


        Model Parameters

        kernel: int
            Size of convolutional kernel for first layer of convolutions.

        trainable_embedding; bool
            Toggle to control whether a trainable embedding layer is used or native
            one-hot representation for convolutional layers.

        embedding_dim_aa: int
            Learned latent dimensionality of amino-acids.

        embedding_dim_genes: int
            Learned latent dimensionality of VDJ genes

        embedding_dim_hla: int
            Learned latent dimensionality of HLA

        num_fc_layers: int
            Number of fully connected layers following convolutional layer.

        units_fc: int
            Number of nodes per fully-connected layers following convolutional layer.

        weight_by_class: bool
            Option to weight loss by the inverse of the class frequency. Useful for
            unbalanced classes.

        class_weights: dict
            In order to specify custom weights for each class during training, one
            can provide a dictionary with these weights.
                i.e. {'A':1.0,'B':2.0'}

        use_only_seq: bool
            To only use sequence feaures, set to True. This will turn off features learned
            from gene usage.

        use_only_gene: bool
            To only use gene-usage features, set to True. This will turn off features from
            the sequences.

        use_only_hla: bool
            To only use hla feaures, set to True.

        size_of_net: list or str
            The convolutional layers of this network have 3 layers for which the use can
            modify the number of neurons per layer. The user can either specify the size of the network
            with the following options:
                - small == [12,32,64] neurons for the 3 respective layers
                - medium == [32,64,128] neurons for the 3 respective layers
                - large == [64,128,256] neurons for the 3 respective layers
                - custom, where the user supplies a list with the number of nuerons for the respective layers
                    i.e. [3,3,3] would have 3 neurons for all 3 layers.
                    One can also adjust the number of layers for the convolutional stack by changing the length of
                    this list. [3,3,3] = 3 layers, [3,3,3,3] = 4 layers.

        graph_seed: int
            For deterministic initialization of weights of the graph, set this to value of choice.

        drop_out_rate: float
            drop out rate for fully connected layers

        The following parameters are used to implement Multi-Sample Dropout at the final layer of the model as described in
        "Multi-Sample Dropout for Accelerated Training and Better Generalization"
        https://arxiv.org/abs/1905.09788
        This method has been shown to improve generalization of deep neural networks as well as inmprove convergence.

        multisample_dropout: bool
            Set this parameter to True to implement this method.

         multisample_dropout_rate: float
            The dropout rate for this multi-sample dropout layer.

         multisample_dropout_num_masks: int
            The number of masks to sample from for the Multi-Sample Dropout layer.


        Training Parameters

        batch_size: int
            Size of batch to be used for each training iteration of the net.

        epochs_min: int
            Minimum number of epochs for training neural network.

        stop_criterion: float
            Minimum percent decrease in determined interval (below) to continue
            training. Used as early stopping criterion.

        stop_criterion_window: int
            The window of data to apply the stopping criterion.

        accuracy_min: float
            Optional parameter to allow alternative training strategy until minimum
            training accuracy is achieved, at which point, training ceases.

        train_loss_min: float
            Optional parameter to allow alternative training strategy until minimum
            training loss is achieved, at which point, training ceases.

        hinge_loss_t: float
            The per sequence loss minimum at which the loss of that sequence is not used
            to penalize the model anymore. In other words, once a per sequence loss has hit
            this value, it gets set to 0.0.

        convergence: str
            This parameter determines which loss to assess the convergence criteria on.
            Options are 'validation' or 'training'. This is useful in the case one wants
            to change the convergence criteria on the training data when the training and validation
            partitions have been combined and used to training the model.

        learning_rate: float
            The learning rate for training the neural network. Making this value larger will
            increase the rate of convergence but can introduce instability into training. For most,
            altering this value will not be necessary.

        suppress_output: bool
            To suppress command line output with training statisitcs, set to True.

        batch_seed: int
            For deterministic batching during training, set this value to an integer of choice.

        Returns
        ---------------------------------------


        '''

        y_pred = []
        y_test = []
        predicted = np.zeros_like(self.predicted)
        counts = np.zeros_like(self.predicted)
        self._reset_models()
        self._build(kernel,trainable_embedding,embedding_dim_aa, embedding_dim_genes, embedding_dim_hla,
               num_fc_layers, units_fc,weight_by_class, class_weights,
               use_only_seq, use_only_gene, use_only_hla, size_of_net,graph_seed,
               drop_out_rate,multisample_dropout, multisample_dropout_rate, multisample_dropout_num_masks,
               batch_size, epochs_min, stop_criterion, stop_criterion_window,
               accuracy_min, train_loss_min, hinge_loss_t, convergence, learning_rate, suppress_output)

        for i in range(0, folds):
            if suppress_output is False:
                print(i)
            if seeds is not None:
                np.random.seed(seeds[i])
            self.Get_Train_Valid_Test(test_size=test_size, LOO=LOO,split_by_sample=split_by_sample,combine_train_valid=combine_train_valid)
            self._train(batch_seed=batch_seed,iteration=i)

            y_test.append(self.y_test)
            y_pred.append(self.y_pred)

            predicted[self.test[self.var_dict['seq_index']]] += self.y_pred
            counts[self.test[self.var_dict['seq_index']]] += 1

            if self.regression is False:
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

    def K_Fold_CrossVal(self,folds=None,split_by_sample=False,combine_train_valid=False,seeds=None,
                        kernel=5, trainable_embedding=True, embedding_dim_aa=64, embedding_dim_genes=48, embedding_dim_hla=12,
                        num_fc_layers=0, units_fc=12, weight_by_class=False, class_weights=None,
                        use_only_seq=False, use_only_gene=False, use_only_hla=False, size_of_net='medium', graph_seed=None,
                        drop_out_rate=0.0, multisample_dropout=False, multisample_dropout_rate=0.50, multisample_dropout_num_masks=64,
                        batch_size=1000, epochs_min=10, stop_criterion=0.001, stop_criterion_window=10,
                        accuracy_min=None, train_loss_min=None, hinge_loss_t=0.0, convergence='validation', learning_rate=0.001, suppress_output=False,
                        batch_seed=None):

        '''
        K_Fold Cross-Validation for Single-Sequence Classifier

        If the number of sequences is small but training the single-sequence classifier, one
        can use K_Fold Cross Validation to train on all but one before assessing
        predictive performance.After this method is run, the AUC_Curve method can be run to
        assess the overall performance.

        Inputs
        ---------------------------------------
        K-Fold Parameters

        folds: int
            Number of Folds

        split_by_sample: int
            In the case one wants to train the single sequence classifer but not to mix the train/test
            sets with sequences from different samples, one can set this parameter to True to do the train/test
            splits by sample.

        combine_train_valid: bool
            To combine the training and validation partitions into one which will be used for training
            and updating the model parameters, set this to True. This will also set the validation partition
            to the test partition. In other words, new train set becomes (original train + original valid) and then
            new valid = original test partition, new test = original test partition. Therefore, if setting this parameter
            to True, change one of the training parameters to set the stop training criterion (i.e. train_loss_min)
            to stop training based on the train set. If one does not chanage the stop training criterion, the decision of
            when to stop training will be based on the test data (which is considered a form of over-fitting).

        seeds: nd.array
            In order to set a deterministic train/test split over the K-Fold Simulations, one can provide an array
            of seeds for each K-fold simulation. This will result in the same train/test split over the N Fold simulations.
            This parameter, if provided, should have the same size of the value of folds.

        Model Parameters

        kernel: int
            Size of convolutional kernel for first layer of convolutions.

        trainable_embedding; bool
            Toggle to control whether a trainable embedding layer is used or native
            one-hot representation for convolutional layers.

        embedding_dim_aa: int
            Learned latent dimensionality of amino-acids.

        embedding_dim_genes: int
            Learned latent dimensionality of VDJ genes

        embedding_dim_hla: int
            Learned latent dimensionality of HLA

        num_fc_layers: int
            Number of fully connected layers following convolutional layer.

        units_fc: int
            Number of nodes per fully-connected layers following convolutional layer.

        weight_by_class: bool
            Option to weight loss by the inverse of the class frequency. Useful for
            unbalanced classes.

        class_weights: dict
            In order to specify custom weights for each class during training, one
            can provide a dictionary with these weights.
                i.e. {'A':1.0,'B':2.0'}

        use_only_seq: bool
            To only use sequence feaures, set to True. This will turn off features learned
            from gene usage.

        use_only_gene: bool
            To only use gene-usage features, set to True. This will turn off features from
            the sequences.

        use_only_hla: bool
            To only use hla feaures, set to True.

        size_of_net: list or str
            The convolutional layers of this network have 3 layers for which the use can
            modify the number of neurons per layer. The user can either specify the size of the network
            with the following options:
                - small == [12,32,64] neurons for the 3 respective layers
                - medium == [32,64,128] neurons for the 3 respective layers
                - large == [64,128,256] neurons for the 3 respective layers
                - custom, where the user supplies a list with the number of nuerons for the respective layers
                    i.e. [3,3,3] would have 3 neurons for all 3 layers.
                    One can also adjust the number of layers for the convolutional stack by changing the length of
                    this list. [3,3,3] = 3 layers, [3,3,3,3] = 4 layers.

        graph_seed: int
            For deterministic initialization of weights of the graph, set this to value of choice.

        drop_out_rate: float
            drop out rate for fully connected layers

        The following parameters are used to implement Multi-Sample Dropout at the final layer of the model as described in
        "Multi-Sample Dropout for Accelerated Training and Better Generalization"
        https://arxiv.org/abs/1905.09788
        This method has been shown to improve generalization of deep neural networks as well as inmprove convergence.

        multisample_dropout: bool
            Set this parameter to True to implement this method.

         multisample_dropout_rate: float
            The dropout rate for this multi-sample dropout layer.

         multisample_dropout_num_masks: int
            The number of masks to sample from for the Multi-Sample Dropout layer.


        Training Parameters

        batch_size: int
            Size of batch to be used for each training iteration of the net.

        epochs_min: int
            Minimum number of epochs for training neural network.

        stop_criterion: float
            Minimum percent decrease in determined interval (below) to continue
            training. Used as early stopping criterion.

        stop_criterion_window: int
            The window of data to apply the stopping criterion.

        accuracy_min: float
            Optional parameter to allow alternative training strategy until minimum
            training accuracy is achieved, at which point, training ceases.

        train_loss_min: float
            Optional parameter to allow alternative training strategy until minimum
            training loss is achieved, at which point, training ceases.

        hinge_loss_t: float
            The per sequence loss minimum at which the loss of that sequence is not used
            to penalize the model anymore. In other words, once a per sequence loss has hit
            this value, it gets set to 0.0.

        convergence: str
            This parameter determines which loss to assess the convergence criteria on.
            Options are 'validation' or 'training'. This is useful in the case one wants
            to change the convergence criteria on the training data when the training and validation
            partitions have been combined and used to training the model.

        learning_rate: float
            The learning rate for training the neural network. Making this value larger will
            increase the rate of convergence but can introduce instability into training. For most,
            altering this value will not be necessary.

        suppress_output: bool
            To suppress command line output with training statisitcs, set to True.

        batch_seed: int
            For deterministic batching during training, set this value to an integer of choice.

        Returns
        ---------------------------------------

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
            for ii in range(folds):
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
        self._build(kernel, trainable_embedding, embedding_dim_aa, embedding_dim_genes, embedding_dim_hla,
                    num_fc_layers, units_fc, weight_by_class, class_weights,
                    use_only_seq, use_only_gene, use_only_hla, size_of_net, graph_seed,
                    drop_out_rate, multisample_dropout, multisample_dropout_rate, multisample_dropout_num_masks,
                    batch_size, epochs_min, stop_criterion, stop_criterion_window,
                    accuracy_min, train_loss_min, hinge_loss_t, convergence, learning_rate, suppress_output)

        y_test = []
        y_pred = []
        for ii in range(folds):
            if suppress_output is False:
                print(ii)
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
                y_test2 = np.vstack(y_test)
                y_pred2 = np.vstack(y_pred)

                if suppress_output is False:
                    print("Accuracy = {}".format(np.average(np.equal(np.argmax(y_pred2, 1), np.argmax(y_test2, 1)))))

                    if self.y_test.shape[1] == 2:
                        if ii > 0:
                            if (np.sum(y_test2[:, 0]) != len(y_test2)) and (np.sum(y_test2[:, 0]) != 0):
                                print("AUC = {}".format(roc_auc_score(np.vstack(y_test), np.vstack(y_pred))))

        self.y_test = np.vstack(y_test)
        self.y_pred = np.vstack(y_pred)
        test_idx = np.hstack(test_idx)
        self.predicted = np.zeros_like(self.predicted)
        self.predicted[test_idx] = self.y_pred

        print('K-fold Cross Validation Completed')


class DeepTCR_WF(DeepTCR_S_base):
    def Get_Train_Valid_Test(self,test_size=0.25,LOO=None,combine_train_valid=False,random_perm=False):
        """
        Train/Valid/Test Splits.

        Divide data for train, valid, test set. In general, training is used to train model parameters, validation is
        used to set early stopping, and test acts as blackbox independent test set.

        Inputs
        ---------------------------------------
        test_size: float
            Fraction of sample to be used for valid and test set. For example, if set to 0.25, 25% of the data will
            be set aside for validation and testing sets. In other words, 75% of the data is used for training.

        LOO: int
            Number of samples to leave-out in Leave-One-Out Cross-Validation. For example,
            when set to 2, 2 samples will be left out for the validation set and 2 samples will be left
            out for the test set.

        combine_train_valid: bool
            To combine the training and validation partitions into one which will be used for training
            and updating the model parameters, set this to True. This will also set the validation partition
            to the test partition. In other words, new train set becomes (original train + original valid) and then
            new valid = original test partition, new test = original test partition. Therefore, if setting this parameter
            to True, change one of the training parameters to set the stop training criterion (i.e. train_loss_min)
            to stop training based on the train set. If one does not chanage the stop training criterion, the decision of
            when to stop training will be based on the test data (which is considered a form of over-fitting).

        random_perm: bool
            To do random permutation testing, one can set this parameter to True and this will shuffle the labels.

        Returns
        ---------------------------------------

        """
        Y = []
        for s in self.sample_list:
            if self.regression:
                Y.append(np.array([np.mean(self.Y[np.where(self.sample_id == s)[0]])]))
            else:
                Y.append(np.array([np.mean(self.Y[np.where(self.sample_id == s)[0]],axis=0)]))

        Y = np.vstack(Y)
        if random_perm:
            np.random.shuffle(Y)

        Vars = [np.asarray(self.sample_list)]
        self.train, self.valid, self.test = Get_Train_Valid_Test(Vars=Vars, Y=Y, test_size=test_size, regression=self.regression,LOO=LOO)
        self.LOO = LOO
        Vars.append(Y)
        self.all = Vars

        if (self.valid[0].size==0) or (self.test[0].size==0):
            raise Exception('Choose different train/valid/test parameters!')

        if combine_train_valid:
            for i in range(len(self.train)):
                self.train[i] = np.concatenate((self.train[i],self.valid[i]),axis=0)
                self.valid[i] = self.test[i]

    def _reset_models(self):
        self.models_dir = os.path.join(self.Name,'models')
        if os.path.exists(self.models_dir):
            shutil.rmtree(self.models_dir)
        os.makedirs(self.models_dir)

    def _build(self,kernel=5,num_concepts=12,trainable_embedding = True,embedding_dim_aa=64, embedding_dim_genes=48, embedding_dim_hla=12,
               num_fc_layers=0, units_fc=12,weight_by_class=False, class_weights=None,
               use_only_seq=False, use_only_gene=False, use_only_hla=False, size_of_net='medium',graph_seed = None,
               qualitative_agg=True, quantitative_agg=False, num_agg_layers=0, units_agg=12,
               drop_out_rate=0.0,multisample_dropout=False, multisample_dropout_rate = 0.50,multisample_dropout_num_masks = 64,
               batch_size = 25,batch_size_update = None, epochs_min = 25,stop_criterion=0.25,stop_criterion_window=10,
              accuracy_min = None,train_loss_min=None,hinge_loss_t=0.0,convergence='validation',learning_rate=0.001, suppress_output=False,
               loss_criteria='mean',l2_reg=0.0):

        graph_model = tf.Graph()
        GO = graph_object()
        train_params = graph_object()
        train_params.batch_size = batch_size
        train_params.batch_size_update = batch_size_update
        train_params.epochs_min = epochs_min
        train_params.stop_criterion = stop_criterion
        train_params.stop_criterion_window  = stop_criterion_window
        train_params.accuracy_min = accuracy_min
        train_params.train_loss_min = train_loss_min
        train_params.convergence = convergence
        train_params.suppress_output = suppress_output
        train_params.drop_out_rate = drop_out_rate
        train_params.multisample_dropout_rate = multisample_dropout_rate
        GO.size_of_net = size_of_net
        GO.embedding_dim_genes = embedding_dim_genes
        GO.embedding_dim_aa = embedding_dim_aa
        GO.embedding_dim_hla = embedding_dim_hla
        GO.l2_reg = l2_reg
        with graph_model.device(self.device):
            with graph_model.as_default():
                if graph_seed is not None:
                    tf.compat.v1.set_random_seed(graph_seed)

                GO.net = 'sup'
                GO.Features = Conv_Model(GO,self,trainable_embedding,kernel,
                                         use_only_seq,use_only_gene,use_only_hla,
                                         num_fc_layers,units_fc)
                if self.regression is False:
                    GO.Y = tf.compat.v1.placeholder(tf.float64, shape=[None, self.Y.shape[1]])
                else:
                    GO.Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

                Features = tf.compat.v1.layers.dense(GO.Features, num_concepts, lambda x: isru(x, l=0, h=1, a=0, b=0))
                GO.Features = Features
                agg_list = []
                if qualitative_agg:
                    #qualitative agg
                    GO.Features_W = Features * GO.X_Freq[:, tf.newaxis]
                    GO.Features_Agg = tf.sparse.sparse_dense_matmul(GO.sp, GO.Features_W)
                    agg_list.append(GO.Features_Agg)
                if quantitative_agg:
                    #quantitative agg
                    GO.Features_W_c = Features * GO.X_Counts[:, tf.newaxis]
                    c_b = tf.Variable(name='c_b',initial_value=np.zeros(num_concepts), trainable=True,dtype=tf.float32)
                    GO.Features_Agg_c = isru(tf.sparse.sparse_dense_matmul(GO.sp, GO.Features_W_c)+c_b,l=0,h=1,a=0,b=0)
                    agg_list.append(GO.Features_Agg_c)

                GO.Features_Agg = tf.concat(agg_list,axis=1)

                if num_agg_layers != 0:
                    for lyr in range(num_agg_layers):
                        GO.Features_Agg = tf.compat.v1.layers.dropout(GO.Features_Agg, GO.prob)
                        GO.Features_Agg = tf.compat.v1.layers.dense(GO.Features_Agg, units_agg, tf.nn.relu)

                if self.regression is False:
                    if multisample_dropout:
                        GO.logits = MultiSample_Dropout(GO.Features_Agg,
                                                        num_masks=multisample_dropout_num_masks,
                                                        units=self.Y.shape[1],
                                                        activation=None,
                                                        rate=GO.prob_multisample)
                    else:
                        GO.logits = tf.compat.v1.layers.dense(GO.Features_Agg, self.Y.shape[1])

                    per_sample_loss = tf.nn.softmax_cross_entropy_with_logits(labels=GO.Y, logits=GO.logits)
                    per_sample_loss = per_sample_loss - hinge_loss_t
                    per_sample_loss = tf.cast((per_sample_loss > 0),tf.float32) * per_sample_loss
                    if loss_criteria == 'mean':
                        loss_func = tf.reduce_mean
                    elif loss_criteria == 'max':
                        loss_func = tf.reduce_max
                    elif loss_criteria == 'min':
                        loss_func = tf.reduce_min

                    if weight_by_class is True:
                        #class_weights = tf.constant([(1 / (np.sum(self.train[-1], 0) / np.sum(self.train[-1]))).tolist()])
                        class_weights = tf.constant([(1 / (np.sum(self.Y, 0) / np.sum(self.Y))).tolist()])
                        weights = tf.squeeze(tf.matmul(tf.cast(GO.Y, dtype='float32'), class_weights, transpose_b=True),axis=1)
                        GO.loss = loss_func(weights * per_sample_loss)
                    elif class_weights is not None:
                        weights = np.zeros([1,len(self.lb.classes_)]).astype(np.float32)
                        for key in class_weights:
                            weights[:,self.lb.transform([key])[0]]=class_weights[key]
                        class_weights = tf.constant(weights)
                        weights = tf.squeeze(tf.matmul(tf.cast(GO.Y, dtype='float32'), class_weights, transpose_b=True),axis=1)
                        GO.loss = loss_func(weights * per_sample_loss)
                    else:
                        GO.loss = loss_func(per_sample_loss)
                else:
                    if multisample_dropout:
                        GO.logits = MultiSample_Dropout(GO.Features_Agg,
                                                        num_masks=multisample_dropout_num_masks,
                                                        units=1,
                                                        activation=None,
                                                        rate=GO.prob_multisample)
                    else:
                        GO.logits = tf.compat.v1.layers.dense(GO.Features_Agg, 1)

                    GO.loss = tf.reduce_mean(input_tensor=tf.square(GO.Y-GO.logits))

                var_train = tf.compat.v1.trainable_variables()
                GO.loss = GO.loss + tf.compat.v1.losses.get_regularization_loss()
                if batch_size_update is None:
                    GO.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(GO.loss,var_list=var_train)
                else:
                    GO.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
                    GO.grads_and_vars = GO.opt.compute_gradients(GO.loss, var_train)
                    GO.gradients = tf.gradients(ys=GO.loss,xs=var_train)
                    GO.gradients,keep_ii = zip(*[(v,ii) for ii,v in enumerate(GO.gradients) if v is not None])
                    var_train = list(np.asarray(var_train)[list(keep_ii)])
                    GO.grads_accum = [tf.Variable(tf.zeros_like(v)) for v in GO.gradients]
                    GO.grads_and_vars = list(zip(GO.grads_accum,var_train))
                    GO.opt = GO.opt.apply_gradients(GO.grads_and_vars)


                if self.regression is False:
                    # Operations for validation/test accuracy
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
                self.kernel = kernel

    def _train(self,write=True,batch_seed=None,iteration=0,subsample=None,subsample_by_freq=False,subsample_valid_test=False):
        GO = self.GO
        graph_model = self.graph_model
        train_params = self.train_params

        batch_size = train_params.batch_size
        batch_size_update = train_params.batch_size_update
        epochs_min = train_params.epochs_min
        stop_criterion = train_params.stop_criterion
        stop_criterion_window = train_params.stop_criterion_window
        accuracy_min = train_params.accuracy_min
        train_loss_min = train_params.train_loss_min
        convergence = train_params.convergence
        suppress_output = train_params.suppress_output
        drop_out_rate = train_params.drop_out_rate
        multisample_dropout_rate = train_params.multisample_dropout_rate

        tf.compat.v1.reset_default_graph()
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(graph=graph_model,config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            val_loss_total = []
            train_accuracy_total = []
            train_loss_total = []
            stop_check_list = []
            kernel_weights = []
            var_train = tf.compat.v1.trainable_variables()
            e = 0

            while True:
                if batch_seed is not None:
                    np.random.seed(batch_seed)
                train_loss, train_accuracy, train_predicted,train_auc = \
                    Run_Graph_WF(self.train,sess,self,GO,batch_size,batch_size_update,random=True,train=True,
                                 drop_out_rate=drop_out_rate,multisample_dropout_rate=multisample_dropout_rate,
                                 subsample=subsample,subsample_by_freq=subsample_by_freq)

                train_accuracy_total.append(train_accuracy)
                train_loss_total.append(train_loss)
                #kernel_weights.append(tf.compat.v1.trainable_variables()[0].eval())
                kernel_weights.append(np.hstack([np.ndarray.flatten(x.eval()) for x in var_train]))

                if subsample_valid_test is False:
                    subsample_vt = None
                else:
                    subsample_vt = subsample

                valid_loss, valid_accuracy, valid_predicted, valid_auc = \
                    Run_Graph_WF(self.valid, sess, self, GO, batch_size,batch_size_update, random=False, train=False,
                                 subsample=subsample_vt,subsample_by_freq=subsample_by_freq)

                val_loss_total.append(valid_loss)

                test_loss, test_accuracy, test_predicted, test_auc = \
                    Run_Graph_WF(self.test, sess, self, GO, batch_size,batch_size_update, random=False, train=False,
                                 subsample=subsample_vt,subsample_by_freq=subsample_by_freq)

                self.y_pred = test_predicted
                self.y_test = self.test[-1]

                if suppress_output is False:
                    print("Training_Statistics: \n",
                          "Epoch: {}".format(e),
                          "Training loss: {:.5f}".format(train_loss),
                          "Validation loss: {:.5f}".format(valid_loss),
                          "Testing loss: {:.5f}".format(test_loss),
                          "Training Accuracy: {:.5}".format(train_accuracy),
                          "Validation Accuracy: {:.5}".format(valid_accuracy),
                          "Testing Accuracy: {:.5}".format(test_accuracy),
                          'Testing AUC: {:.5}'.format(test_auc))

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

                e +=  1

            self.train_loss = train_loss_total
            self.kernel_weights = kernel_weights

            test_loss, test_accuracy, test_predicted, test_auc = \
                Run_Graph_WF(self.test, sess, self, GO, batch_size, batch_size_update, random=False, train=False)

            self.y_pred = test_predicted
            self.y_test = self.test[-1]

            if write:
                batch_size_seq = round(len(self.sample_id)/(len(self.sample_list)/batch_size))
                Get_Seq_Features_Indices(self,batch_size_seq,GO,sess)
                self.features = Get_Latent_Features(self, batch_size_seq, GO, sess)

            pred, idx = Get_Sequence_Pred(self, batch_size, GO, sess)
            if len(idx.shape) == 0:
                idx = idx.reshape(-1,1)

            self.predicted[idx] += pred
            self.seq_idx = idx

            self.train_idx = np.isin(self.sample_id,self.train[0])
            self.valid_idx = np.isin(self.sample_id,self.valid[0])
            self.test_idx = np.isin(self.sample_id,self.test[0])

            if write:
                if self.use_alpha is True:
                    var_save = [self.alpha_features,self.alpha_indices,self.alpha_sequences]
                    with open(os.path.join(self.Name, self.Name) + '_alpha_features.pkl', 'wb') as f:
                        pickle.dump(var_save, f)

                if self.use_beta is True:
                    var_save = [self.beta_features,self.beta_indices,self.beta_sequences]
                    with open(os.path.join(self.Name, self.Name) + '_beta_features.pkl', 'wb') as f:
                        pickle.dump(var_save, f)

                with open(os.path.join(self.Name, self.Name) + '_kernel.pkl', 'wb') as f:
                    pickle.dump(self.kernel, f)

                if self.use_hla:
                    self.HLA_embed = GO.embedding_layer_hla.eval()

                # save model data and information for inference engine
                save_model_data(self, GO.saver, sess, name='WF', get=GO.predicted,iteration=iteration)

            print('Done Training')

    def Train(self,kernel=5,num_concepts=12,trainable_embedding = True,embedding_dim_aa=64, embedding_dim_genes=48, embedding_dim_hla=12,
               num_fc_layers=0, units_fc=12,weight_by_class=False, class_weights=None,
               use_only_seq=False, use_only_gene=False, use_only_hla=False, size_of_net='medium',graph_seed = None,
               qualitative_agg=True, quantitative_agg=False, num_agg_layers=0, units_agg=12,
               drop_out_rate=0.0,multisample_dropout=False, multisample_dropout_rate = 0.50,multisample_dropout_num_masks = 64,
               batch_size = 25,batch_size_update = None, epochs_min = 25,stop_criterion=0.25,stop_criterion_window=10,
              accuracy_min = None,train_loss_min=None,hinge_loss_t=0.0,convergence='validation',learning_rate=0.001, suppress_output=False,
              loss_criteria='mean',l2_reg=0.0,
              batch_seed = None,
              subsample=None,subsample_by_freq=False,subsample_valid_test=False):

        """
        Train Whole-Sample Classifier

        This method trains the network and saves features values at the
        end of training for motif analysis.

        Inputs
        ---------------------------------------

        Model Parameters

        kernel: int
            Size of convolutional kernel for first layer of convolutions.

        num_concepts: int
            Number of concepts for multi-head attention mechanism. Depending on the expected heterogeneity of the
            repertoires being analyed, one can adjust this hyperparameter.

        trainable_embedding; bool
            Toggle to control whether a trainable embedding layer is used or native
            one-hot representation for convolutional layers.

        embedding_dim_aa: int
            Learned latent dimensionality of amino-acids.

        embedding_dim_genes: int
            Learned latent dimensionality of VDJ genes

        embedding_dim_hla: int
            Learned latent dimensionality of HLA

        num_fc_layers: int
            Number of fully connected layers following convolutional layer.

        units_fc: int
            Number of nodes per fully-connected layers following convolutional layer.

        weight_by_class: bool
            Option to weight loss by the inverse of the class frequency. Useful for
            unbalanced classes.

        class_weights: dict
            In order to specify custom weights for each class during training, one
            can provide a dictionary with these weights.
                i.e. {'A':1.0,'B':2.0'}

        use_only_seq: bool
            To only use sequence feaures, set to True. This will turn off features learned
            from gene usage.

        use_only_gene: bool
            To only use gene-usage features, set to True. This will turn off features from
            the sequences.

        use_only_hla: bool
            To only use hla feaures, set to True.

        size_of_net: list or str
            The convolutional layers of this network have 3 layers for which the use can
            modify the number of neurons per layer. The user can either specify the size of the network
            with the following options:
                - small == [12,32,64] neurons for the 3 respective layers
                - medium == [32,64,128] neurons for the 3 respective layers
                - large == [64,128,256] neurons for the 3 respective layers
                - custom, where the user supplies a list with the number of nuerons for the respective layers
                    i.e. [3,3,3] would have 3 neurons for all 3 layers.
                    One can also adjust the number of layers for the convolutional stack by changing the length of
                    this list. [3,3,3] = 3 layers, [3,3,3,3] = 4 layers.

        graph_seed: int
            For deterministic initialization of weights of the graph, set this to value of choice.

        The following two options alter how the predictive signatures in the neural network are aggregated
        to make a prediction about the repertoire. If qualitative_agg or quantitative_agg are set to True,
        this will include these different types of aggregation in the predcitions. One can set either to True or
        both to True and this will allow a user to incorporate features from multiple modes of aggregation.

        qualitative_agg: bool
            If set to True, the model will aggregate the feature values per repertoire weighted by frequency of each
            TCR. This is considered a 'qualitative' aggregation as the prediction of the repertoire is based on the relative
            distribution of the repertoire. In other words, this type of aggregation is a count-independent measure of aggregation.
            This is the mode of aggregation that has been more thoroughly tested across multiple scientific examples.

        quantitative_agg: bool
            If set to True, the model will aggregate the feature values per repertoire weighted by counts of each TCR.
            This is considered a 'quantitative' aggregation as the prediction of the repertoire is based on teh absolute
            distribution of the repertoire. In other words, this type of aggregation is a count-dependent measure of aggregation.
            If one believes the counts are important for the predictive value of the model, one can set this parameter to True.

        num_agg_layers: int
            Following the aggregation layer in the network, one can choose to add more fully-connected layers before
            the final classification layer. This parameter will set how many layers to add after aggregation. This likely
            is helpful when using both types of aggregation (as detailed above) to combine those feature values.

        units_agg: int
            For the fully-connected layers after aggregation, this parameter sets the number of units/nodes per layer.

        drop_out_rate: float
            drop out rate for fully connected layers

        The following parameters are used to implement Multi-Sample Dropout at the final layer of the model as described in
        "Multi-Sample Dropout for Accelerated Training and Better Generalization"
        https://arxiv.org/abs/1905.09788
        This method has been shown to improve generalization of deep neural networks as well as inmprove convergence.

        multisample_dropout: bool
            Set this parameter to True to implement this method.

         multisample_dropout_rate: float
            The dropout rate for this multi-sample dropout layer.

         multisample_dropout_num_masks: int
            The number of masks to sample from for the Multi-Sample Dropout layer.


        Training Parameters

        batch_size: int
            Size of batch to be used for each training iteration of the net.

        batch_size_update: int
            In the case that the size of the samples are very large, one may not want to update
            the weights of the network as often as batches are put onto the gpu. Therefore, if
            one wants to update the weights less often than how often the batches of data are put onto the
            gpu, one can set this parameter to something other than None. An example would be if batch_size is set to 5
            and batch_size_update is set to 30, while only 5 samples will be put on the gpu at a time, the weights will
            only be updated after 30 samples have been put on the gpu. This parameter is only relevant when using
            gpu's for training and there are memory constraints from very large samples.

        epochs_min: int
            Minimum number of epochs for training neural network.

        stop_criterion: float
            Minimum percent decrease in determined interval (below) to continue
            training. Used as early stopping criterion.

        stop_criterion_window: int
            The window of data to apply the stopping criterion.

        accuracy_min: float
            Optional parameter to allow alternative training strategy until minimum
            training accuracy is achieved, at which point, training ceases.

        train_loss_min: float
            Optional parameter to allow alternative training strategy until minimum
            training loss is achieved, at which point, training ceases.

        hinge_loss_t: float
            The per sample loss minimum at which the loss of that sample is not used
            to penalize the model anymore. In other words, once a per sample loss has hit
            this value, it gets set to 0.0.

        convergence: str
            This parameter determines which loss to assess the convergence criteria on.
            Options are 'validation' or 'training'. This is useful in the case one wants
            to change the convergence criteria on the training data when the training and validation
            partitions have been combined and used to training the model.

        learning_rate: float
            The learning rate for training the neural network. Making this value larger will
            increase the rate of convergence but can introduce instability into training. For most,
            altering this value will not be necessary.

        suppress_output: bool
            To suppress command line output with training statisitcs, set to True.

        batch_seed: int
            For deterministic batching during training, set this value to an integer of choice.

        subsample: int
            Number of sequences to sub-sample from repertoire during training to improve speed of convergence
            as well as being a form of regularization.

        subsample_by_freq: bool
            Whether to sub-sample randomly in the repertoire or as a function of the frequency of the TCR.

        subsample_valid_test: bool
            Whether to sub-sample during valid/test cohorts while training. This is mostly as well to improve speed
            to convergence and generalizability.

        Returns
        ---------------------------------------

        """
        #Create directory for models
        self._reset_models()
        self._build(kernel,num_concepts,trainable_embedding,embedding_dim_aa, embedding_dim_genes, embedding_dim_hla,
               num_fc_layers, units_fc,weight_by_class, class_weights,
               use_only_seq, use_only_gene, use_only_hla, size_of_net,graph_seed,
               qualitative_agg, quantitative_agg, num_agg_layers, units_agg,
               drop_out_rate,multisample_dropout, multisample_dropout_rate,multisample_dropout_num_masks,
               batch_size,batch_size_update, epochs_min,stop_criterion,stop_criterion_window,
              accuracy_min,train_loss_min,hinge_loss_t,convergence,learning_rate, suppress_output,
                    loss_criteria,l2_reg)
        self._train(write=True,batch_seed=batch_seed,iteration=0,
                    subsample=subsample,subsample_by_freq=subsample_by_freq,subsample_valid_test=subsample_valid_test)

    def Monte_Carlo_CrossVal(self,folds=5,test_size=0.25,LOO=None,combine_train_valid=False,random_perm=False,seeds=None,
                             kernel=5, num_concepts=12, trainable_embedding=True, embedding_dim_aa=64, embedding_dim_genes=48, embedding_dim_hla=12,
                             num_fc_layers=0, units_fc=12, weight_by_class=False, class_weights=None,
                             use_only_seq=False, use_only_gene=False, use_only_hla=False, size_of_net='medium',graph_seed=None,
                             qualitative_agg=True, quantitative_agg=False, num_agg_layers=0, units_agg=12,
                             drop_out_rate=0.0, multisample_dropout=False, multisample_dropout_rate=0.50,multisample_dropout_num_masks=64,
                             batch_size=25, batch_size_update=None, epochs_min=25, stop_criterion=0.25, stop_criterion_window=10,
                             accuracy_min=None, train_loss_min=None, hinge_loss_t=0.0, convergence='validation',learning_rate=0.001, suppress_output=False,
                             loss_criteria='mean',l2_reg = 0.0,
                             batch_seed=None,
                             subsample=None,subsample_by_freq=False,subsample_valid_test=False):

        """
        Monte Carlo Cross-Validation for Whole Sample Classifier

        If the number of samples is small but training the whole sample classifier, one
        can use Monte Carlo Cross Validation to train a number of iterations before assessing
        predictive performance.After this method is run, the AUC_Curve method can be run to
        assess the overall performance.

        Inputs
        ---------------------------------------
        Monte-Carlo Parameters

        folds: int
            Number of iterations for Cross-Validation

        test_size: float
            Fraction of sample to be used for valid and test set.

        LOO: int
            Number of samples to leave-out in Leave-One-Out Cross-Validation

        combine_train_valid: bool
            To combine the training and validation partitions into one which will be used for training
            and updating the model parameters, set this to True. This will also set the validation partition
            to the test partition. In other words, new train set becomes (original train + original valid) and then
            new valid = original test partition, new test = original test partition. Therefore, if setting this parameter
            to True, change one of the training parameters to set the stop training criterion (i.e. train_loss_min)
            to stop training based on the train set. If one does not chanage the stop training criterion, the decision of
            when to stop training will be based on the test data (which is considered a form of over-fitting).

        random_perm: bool
            To do random permutation testing, one can set this parameter to True and this will shuffle the labels.

        seeds: nd.array
            In order to set a deterministic train/test split over the Monte-Carlo Simulations, one can provide an array
            of seeds for each MC simulation. This will result in the same train/test split over the N MC simulations.
            This parameter, if provided, should have the same size of the value of folds.


        Model Parameters

        kernel: int
            Size of convolutional kernel for first layer of convolutions.

        num_concepts: int
            Number of concepts for multi-head attention mechanism. Depending on the expected heterogeneity of the
            repertoires being analyed, one can adjust this hyperparameter.

        trainable_embedding; bool
            Toggle to control whether a trainable embedding layer is used or native
            one-hot representation for convolutional layers.

        embedding_dim_aa: int
            Learned latent dimensionality of amino-acids.

        embedding_dim_genes: int
            Learned latent dimensionality of VDJ genes

        embedding_dim_hla: int
            Learned latent dimensionality of HLA

        num_fc_layers: int
            Number of fully connected layers following convolutional layer.

        units_fc: int
            Number of nodes per fully-connected layers following convolutional layer.

        weight_by_class: bool
            Option to weight loss by the inverse of the class frequency. Useful for
            unbalanced classes.

        class_weights: dict
            In order to specify custom weights for each class during training, one
            can provide a dictionary with these weights.
                i.e. {'A':1.0,'B':2.0'}

        use_only_seq: bool
            To only use sequence feaures, set to True. This will turn off features learned
            from gene usage.

        use_only_gene: bool
            To only use gene-usage features, set to True. This will turn off features from
            the sequences.

        use_only_hla: bool
            To only use hla feaures, set to True.

        size_of_net: list or str
            The convolutional layers of this network have 3 layers for which the use can
            modify the number of neurons per layer. The user can either specify the size of the network
            with the following options:
                - small == [12,32,64] neurons for the 3 respective layers
                - medium == [32,64,128] neurons for the 3 respective layers
                - large == [64,128,256] neurons for the 3 respective layers
                - custom, where the user supplies a list with the number of nuerons for the respective layers
                    i.e. [3,3,3] would have 3 neurons for all 3 layers.

        graph_seed: int
            For deterministic initialization of weights of the graph, set this to value of choice.

        The following two options alter how the predictive signatures in the neural network are aggregated
        to make a prediction about the repertoire. If qualitative_agg or quantitative_agg are set to True,
        this will include these different types of aggregation in the predcitions. One can set either to True or
        both to True and this will allow a user to incorporate features from multiple modes of aggregation.

        qualitative_agg: bool
            If set to True, the model will aggregate the feature values per repertoire weighted by frequency of each
            TCR. This is considered a 'qualitative' aggregation as the prediction of the repertoire is based on the relative
            distribution of the repertoire. In other words, this type of aggregation is a count-independent measure of aggregation.
            This is the mode of aggregation that has been more thoroughly tested across multiple scientific examples.

        quantitative_agg: bool
            If set to True, the model will aggregate the feature values per repertoire weighted by counts of each TCR.
            This is considered a 'quantitative' aggregation as the prediction of the repertoire is based on teh absolute
            distribution of the repertoire. In other words, this type of aggregation is a count-dependent measure of aggregation.
            If one believes the counts are important for the predictive value of the model, one can set this parameter to True.

        num_agg_layers: int
            Following the aggregation layer in the network, one can choose to add more fully-connected layers before
            the final classification layer. This parameter will set how many layers to add after aggregation. This likely
            is helpful when using both types of aggregation (as detailed above) to combine those feature values.

        units_agg: int
            For the fully-connected layers after aggregation, this parameter sets the number of units/nodes per layer.

        drop_out_rate: float
            drop out rate for fully connected layers

        The following parameters are used to implement Multi-Sample Dropout at the final layer of the model as described in
        "Multi-Sample Dropout for Accelerated Training and Better Generalization"
        https://arxiv.org/abs/1905.09788
        This method has been shown to improve generalization of deep neural networks as well as inmprove convergence.

        multisample_dropout: bool
            Set this parameter to True to implement this method.

         multisample_dropout_rate: float
            The dropout rate for this multi-sample dropout layer.

         multisample_dropout_num_masks: int
            The number of masks to sample from for the Multi-Sample Dropout layer.


        Training Parameters

        batch_size: int
            Size of batch to be used for each training iteration of the net.

        batch_size_update: int
            In the case that the size of the samples are very large, one may not want to update
            the weights of the network as often as batches are put onto the gpu. Therefore, if
            one wants to update the weights less often than how often the batches of data are put onto the
            gpu, one can set this parameter to something other than None. An example would be if batch_size is set to 5
            and batch_size_update is set to 30, while only 5 samples will be put on the gpu at a time, the weights will
            only be updated after 30 samples have been put on the gpu. This parameter is only relevant when using
            gpu's for training and there are memory constraints from very large samples.

        epochs_min: int
            Minimum number of epochs for training neural network.

        stop_criterion: float
            Minimum percent decrease in determined interval (below) to continue
            training. Used as early stopping criterion.

        stop_criterion_window: int
            The window of data to apply the stopping criterion.

        accuracy_min: float
            Optional parameter to allow alternative training strategy until minimum
            training accuracy is achieved, at which point, training ceases.

        train_loss_min: float
            Optional parameter to allow alternative training strategy until minimum
            training loss is achieved, at which point, training ceases.

        hinge_loss_t: float
            The per sample loss minimum at which the loss of that sample is not used
            to penalize the model anymore. In other words, once a per sample loss has hit
            this value, it gets set to 0.0.

        convergence: str
            This parameter determines which loss to assess the convergence criteria on.
            Options are 'validation' or 'training'. This is useful in the case one wants
            to change the convergence criteria on the training data when the training and validation
            partitions have been combined and used to training the model.

        learning_rate: float
            The learning rate for training the neural network. Making this value larger will
            increase the rate of convergence but can introduce instability into training. For most,
            altering this value will not be necessary.

        suppress_output: bool
            To suppress command line output with training statisitcs, set to True.

        l2_reg: float
            When training the repertoire classifier, it may help to utilize L2 regularization to prevent sample-specific
            overfitting of the model. By setting the value of this parameter (i.e. 0.01), one will introduce L2 regularization
            through TCR featurization layers of the network.

        batch_seed: int
            For deterministic batching during training, set this value to an integer of choice.

        subsample: int
            Number of sequences to sub-sample from repertoire during training to improve speed of convergence
            as well as being a form of regularization.

        subsample_by_freq: bool
            Whether to sub-sample randomly in the repertoire or as a function of the frequency of the TCR.

        subsample_valid_test: bool
            Whether to sub-sample during valid/test cohorts while training. This is mostly as well to improve speed
            to convergence and generalizability.

        Returns

        self.DFs_pred: dict of dataframes
            This method returns the samples in the test sets of the Monte-Carlo and their
            predicted probabilities for each class.
        ---------------------------------------

        """

        y_pred = []
        y_test = []
        files = []
        self.predicted = np.zeros_like(self.predicted)
        counts = np.zeros_like(self.predicted)
        self._reset_models()
        self._build(kernel, num_concepts, trainable_embedding, embedding_dim_aa, embedding_dim_genes, embedding_dim_hla,
                    num_fc_layers, units_fc, weight_by_class, class_weights,
                    use_only_seq, use_only_gene, use_only_hla, size_of_net, graph_seed,
                    qualitative_agg, quantitative_agg, num_agg_layers, units_agg,
                    drop_out_rate, multisample_dropout, multisample_dropout_rate, multisample_dropout_num_masks,
                    batch_size, batch_size_update, epochs_min, stop_criterion, stop_criterion_window,
                    accuracy_min, train_loss_min, hinge_loss_t, convergence, learning_rate, suppress_output,
                    loss_criteria,l2_reg)

        for i in range(0, folds):
            if suppress_output is False:
                print(i)
            if seeds is not None:
                np.random.seed(seeds[i])

            self.Get_Train_Valid_Test(test_size=test_size, LOO=LOO,combine_train_valid=combine_train_valid,
                                      random_perm=random_perm)
            self._train(write=True,batch_seed=batch_seed,iteration=i,
                        subsample=subsample,subsample_by_freq=subsample_by_freq,subsample_valid_test=subsample_valid_test)

            y_test.append(self.y_test)
            y_pred.append(self.y_pred)
            files.append(self.test[0])

            counts[self.seq_idx] += 1

            y_test2 = np.vstack(y_test)
            y_pred2 = np.vstack(y_pred)

            if self.regression is False:
                if suppress_output is False:
                    print("Accuracy = {}".format(np.average(np.equal(np.argmax(y_pred2,1),np.argmax(y_test2,1)))))

                    if self.y_test.shape[1] == 2:
                        if i > 0:
                            y_test2 = np.vstack(y_test)
                            if (np.sum(y_test2[:, 0]) != len(y_test2)) and (np.sum(y_test2[:, 0]) != 0):
                                print("AUC = {}".format(roc_auc_score(np.vstack(y_test), np.vstack(y_pred))))


        self.y_test = np.vstack(y_test)
        self.y_pred = np.vstack(y_pred)
        if self.regression is False:
            files = np.hstack(files)
            DFs =[]
            for ii,c in enumerate(self.lb.classes_,0):
                df_out = pd.DataFrame()
                df_out['Samples'] = files
                df_out['y_test'] = self.y_test[:,ii]
                df_out['y_pred'] = self.y_pred[:,ii]
                DFs.append(df_out)

            self.DFs_pred = dict(zip(self.lb.classes_,DFs))

        self.predicted = np.divide(self.predicted,counts, out = np.zeros_like(self.predicted), where = counts != 0)
        print('Monte Carlo Simulation Completed')

    def K_Fold_CrossVal(self,folds=None,combine_train_valid=False,seeds=None,
                        kernel=5, num_concepts=12, trainable_embedding=True, embedding_dim_aa=64, embedding_dim_genes=48, embedding_dim_hla=12,
                        num_fc_layers=0, units_fc=12, weight_by_class=False, class_weights=None,
                        use_only_seq=False, use_only_gene=False, use_only_hla=False, size_of_net='medium', graph_seed=None,
                        qualitative_agg=True, quantitative_agg=False, num_agg_layers=0, units_agg=12,
                        drop_out_rate=0.0, multisample_dropout=False, multisample_dropout_rate=0.50, multisample_dropout_num_masks=64,
                        batch_size=25, batch_size_update=None, epochs_min=25, stop_criterion=0.25, stop_criterion_window=10,
                        accuracy_min=None, train_loss_min=None, hinge_loss_t=0.0, convergence='validation', learning_rate=0.001, suppress_output=False,
                        loss_criteria='mean',l2_reg=0.0,
                        batch_seed=None,
                        subsample=None,subsample_by_freq=False,subsample_valid_test=False):

        """
        K_Fold Cross-Validation for Whole Sample Classifier

        If the number of samples is small but training the whole sample classifier, one
        can use K_Fold Cross Validation to train on all but one before assessing
        predictive performance.After this method is run, the AUC_Curve method can be run to
        assess the overall performance.

        Inputs
        ---------------------------------------

        K-Fold Parameters

        folds: int
            Number of Folds

        combine_train_valid: bool
            To combine the training and validation partitions into one which will be used for training
            and updating the model parameters, set this to True. This will also set the validation partition
            to the test partition. In other words, new train set becomes (original train + original valid) and then
            new valid = original test partition, new test = original test partition. Therefore, if setting this parameter
            to True, change one of the training parameters to set the stop training criterion (i.e. train_loss_min)
            to stop training based on the train set. If one does not chanage the stop training criterion, the decision of
            when to stop training will be based on the test data (which is considered a form of over-fitting).

        seeds: nd.array
            In order to set a deterministic train/test split over the K-Fold Simulations, one can provide an array
            of seeds for each K-fold simulation. This will result in the same train/test split over the N Fold simulations.
            This parameter, if provided, should have the same size of the value of folds.

        Model Parameters

        kernel: int
            Size of convolutional kernel for first layer of convolutions.

        num_concepts: int
            Number of concepts for multi-head attention mechanism. Depending on the expected heterogeneity of the
            repertoires being analyed, one can adjust this hyperparameter.

        trainable_embedding; bool
            Toggle to control whether a trainable embedding layer is used or native
            one-hot representation for convolutional layers.

        embedding_dim_aa: int
            Learned latent dimensionality of amino-acids.

        embedding_dim_genes: int
            Learned latent dimensionality of VDJ genes

        embedding_dim_hla: int
            Learned latent dimensionality of HLA

        num_fc_layers: int
            Number of fully connected layers following convolutional layer.

        units_fc: int
            Number of nodes per fully-connected layers following convolutional layer.

        weight_by_class: bool
            Option to weight loss by the inverse of the class frequency. Useful for
            unbalanced classes.

        class_weights: dict
            In order to specify custom weights for each class during training, one
            can provide a dictionary with these weights.
                i.e. {'A':1.0,'B':2.0'}

        use_only_seq: bool
            To only use sequence feaures, set to True. This will turn off features learned
            from gene usage.

        use_only_gene: bool
            To only use gene-usage features, set to True. This will turn off features from
            the sequences.

        use_only_hla: bool
            To only use hla feaures, set to True.

        size_of_net: list or str
            The convolutional layers of this network have 3 layers for which the use can
            modify the number of neurons per layer. The user can either specify the size of the network
            with the following options:
                - small == [12,32,64] neurons for the 3 respective layers
                - medium == [32,64,128] neurons for the 3 respective layers
                - large == [64,128,256] neurons for the 3 respective layers
                - custom, where the user supplies a list with the number of nuerons for the respective layers
                    i.e. [3,3,3] would have 3 neurons for all 3 layers.
                    One can also adjust the number of layers for the convolutional stack by changing the length of
                    this list. [3,3,3] = 3 layers, [3,3,3,3] = 4 layers.

        graph_seed: int
            For deterministic initialization of weights of the graph, set this to value of choice.

        The following two options alter how the predictive signatures in the neural network are aggregated
        to make a prediction about the repertoire. If qualitative_agg or quantitative_agg are set to True,
        this will include these different types of aggregation in the predcitions. One can set either to True or
        both to True and this will allow a user to incorporate features from multiple modes of aggregation.

        qualitative_agg: bool
            If set to True, the model will aggregate the feature values per repertoire weighted by frequency of each
            TCR. This is considered a 'qualitative' aggregation as the prediction of the repertoire is based on the relative
            distribution of the repertoire. In other words, this type of aggregation is a count-independent measure of aggregation.
            This is the mode of aggregation that has been more thoroughly tested across multiple scientific examples.

        quantitative_agg: bool
            If set to True, the model will aggregate the feature values per repertoire weighted by counts of each TCR.
            This is considered a 'quantitative' aggregation as the prediction of the repertoire is based on teh absolute
            distribution of the repertoire. In other words, this type of aggregation is a count-dependent measure of aggregation.
            If one believes the counts are important for the predictive value of the model, one can set this parameter to True.

        num_agg_layers: int
            Following the aggregation layer in the network, one can choose to add more fully-connected layers before
            the final classification layer. This parameter will set how many layers to add after aggregation. This likely
            is helpful when using both types of aggregation (as detailed above) to combine those feature values.

        units_agg: int
            For the fully-connected layers after aggregation, this parameter sets the number of units/nodes per layer.

        drop_out_rate: float
            drop out rate for fully connected layers

        The following parameters are used to implement Multi-Sample Dropout at the final layer of the model as described in
        "Multi-Sample Dropout for Accelerated Training and Better Generalization"
        https://arxiv.org/abs/1905.09788
        This method has been shown to improve generalization of deep neural networks as well as inmprove convergence.

        multisample_dropout: bool
            Set this parameter to True to implement this method.

         multisample_dropout_rate: float
            The dropout rate for this multi-sample dropout layer.

         multisample_dropout_num_masks: int
            The number of masks to sample from for the Multi-Sample Dropout layer.


        Training Parameters

        batch_size: int
            Size of batch to be used for each training iteration of the net.

        batch_size_update: int
            In the case that the size of the samples are very large, one may not want to update
            the weights of the network as often as batches are put onto the gpu. Therefore, if
            one wants to update the weights less often than how often the batches of data are put onto the
            gpu, one can set this parameter to something other than None. An example would be if batch_size is set to 5
            and batch_size_update is set to 30, while only 5 samples will be put on the gpu at a time, the weights will
            only be updated after 30 samples have been put on the gpu. This parameter is only relevant when using
            gpu's for training and there are memory constraints from very large samples.

        epochs_min: int
            Minimum number of epochs for training neural network.

        stop_criterion: float
            Minimum percent decrease in determined interval (below) to continue
            training. Used as early stopping criterion.

        stop_criterion_window: int
            The window of data to apply the stopping criterion.

        accuracy_min: float
            Optional parameter to allow alternative training strategy until minimum
            training accuracy is achieved, at which point, training ceases.

        train_loss_min: float
            Optional parameter to allow alternative training strategy until minimum
            training loss is achieved, at which point, training ceases.

        hinge_loss_t: float
            The per sample loss minimum at which the loss of that sample is not used
            to penalize the model anymore. In other words, once a per sample loss has hit
            this value, it gets set to 0.0.

        convergence: str
            This parameter determines which loss to assess the convergence criteria on.
            Options are 'validation' or 'training'. This is useful in the case one wants
            to change the convergence criteria on the training data when the training and validation
            partitions have been combined and used to training the model.

        learning_rate: float
            The learning rate for training the neural network. Making this value larger will
            increase the rate of convergence but can introduce instability into training. For most,
            altering this value will not be necessary.

        suppress_output: bool
            To suppress command line output with training statisitcs, set to True.

        l2_reg: float
            When training the repertoire classifier, it may help to utilize L2 regularization to prevent sample-specific
            overfitting of the model. By setting the value of this parameter (i.e. 0.01), one will introduce L2 regularization
            through TCR featurization layers of the network.

        batch_seed: int
            For deterministic batching during training, set this value to an integer of choice.

        subsample: int
            Number of sequences to sub-sample from repertoire during training to improve speed of convergence
            as well as being a form of regularization.

        subsample_by_freq: bool
            Whether to sub-sample randomly in the repertoire or as a function of the frequency of the TCR.

        subsample_valid_test: bool
            Whether to sub-sample during valid/test cohorts while training. This is mostly as well to improve speed
            to convergence and generalizability.

        Returns
        ---------------------------------------

        """
        self.predicted = np.zeros((len(self.Y),len(self.lb.classes_)))
        num_samples = len(self.sample_list)
        if folds is None:
            folds = num_samples

        Y = []
        for s in self.sample_list:
            Y.append(self.Y[np.where(self.sample_id == s)[0][0]])
        Y = np.vstack(Y)

        Vars = [np.asarray(self.sample_list)]

        #Create Folds
        idx = list(range(num_samples))
        idx_left = idx
        file_per_sample = num_samples // folds
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

        self._reset_models()
        self._build(kernel, num_concepts, trainable_embedding, embedding_dim_aa, embedding_dim_genes, embedding_dim_hla,
                    num_fc_layers, units_fc, weight_by_class, class_weights,
                    use_only_seq, use_only_gene, use_only_hla, size_of_net, graph_seed,
                    qualitative_agg, quantitative_agg, num_agg_layers, units_agg,
                    drop_out_rate, multisample_dropout, multisample_dropout_rate, multisample_dropout_num_masks,
                    batch_size, batch_size_update, epochs_min, stop_criterion, stop_criterion_window,
                    accuracy_min, train_loss_min, hinge_loss_t, convergence, learning_rate, suppress_output,
                    loss_criteria,l2_reg)

        y_test = []
        y_pred = []
        for ii in range(folds):
            if suppress_output is False:
                print(ii)
            train_idx = np.setdiff1d(idx,test_idx[ii])
            valid_idx = np.random.choice(train_idx,len(train_idx)//(folds-1),replace=False)
            train_idx = np.setdiff1d(train_idx,valid_idx)

            self.train,self.valid,self.test = Get_Train_Valid_Test_KFold(Vars=Vars,
                                                               train_idx=train_idx,
                                                               valid_idx = valid_idx,
                                                               test_idx = test_idx[ii],Y=Y)
            if combine_train_valid:
                for i in range(len(self.train)):
                    self.train[i] = np.concatenate((self.train[i], self.valid[i]), axis=0)
                    self.valid[i] = self.test[i]

            self.LOO = None
            self._train(write=True, batch_seed=batch_seed, iteration=ii,
                        subsample=subsample,subsample_by_freq=subsample_by_freq,subsample_valid_test=subsample_valid_test)

            y_test.append(self.y_test)
            y_pred.append(self.y_pred)

            y_test2 = np.vstack(y_test)
            y_pred2 = np.vstack(y_pred)

            if self.regression is False:
                if suppress_output is False:
                    print("Accuracy = {}".format(np.average(np.equal(np.argmax(y_pred2, 1), np.argmax(y_test2, 1)))))

                    if self.y_test.shape[1] == 2:
                        if ii > 0:
                            if (np.sum(y_test2[:, 0]) != len(y_test2)) and (np.sum(y_test2[:, 0]) != 0):
                                print("AUC = {}".format(roc_auc_score(np.vstack(y_test), np.vstack(y_pred))))

        self.y_test = np.vstack(y_test)
        self.y_pred = np.vstack(y_pred)
        print('K-fold Cross Validation Completed')

    def _inf(self,data,model='model_0'):
        X_Seq_alpha = data.X_Seq_alpha
        X_Seq_beta = data.X_Seq_beta
        v_beta_num = data.v_beta_num
        d_beta_num = data.d_beta_num
        j_beta_num = data.j_beta_num
        v_alpha_num = data.v_alpha_num
        j_alpha_num = data.j_alpha_num
        hla_data_seq_num = data.hla_data_seq_num
        freq = data.freq
        counts = data.counts
        batch_size = data.batch_size
        sample_labels = data.sample_labels
        get = data.get

        tf.compat.v1.reset_default_graph()
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.device(self.device):
            saver = tf.compat.v1.train.import_meta_graph(os.path.join(self.Name, 'models', model, 'model.ckpt.meta'),clear_devices=True)
        graph = tf.compat.v1.get_default_graph()
        with tf.compat.v1.Session(graph=graph,config=config) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join(self.Name,'models', model)))

            X_Freq = graph.get_tensor_by_name('Freq:0')
            X_Counts = graph.get_tensor_by_name('Counts:0')
            sp_i = graph.get_tensor_by_name('sp/indices:0')
            sp_v = graph.get_tensor_by_name('sp/values:0')
            sp_s = graph.get_tensor_by_name('sp/shape:0')

            if self.use_alpha is True:
                X_Seq_alpha_v = graph.get_tensor_by_name('Input_Alpha:0')

            if self.use_beta is True:
                X_Seq_beta_v = graph.get_tensor_by_name('Input_Beta:0')

            if self.use_v_beta is True:
                X_v_beta = graph.get_tensor_by_name('Input_V_Beta:0')

            if self.use_d_beta is True:
                X_d_beta = graph.get_tensor_by_name('Input_D_Beta:0')

            if self.use_j_beta is True:
                X_j_beta = graph.get_tensor_by_name('Input_J_Beta:0')

            if self.use_v_alpha is True:
                X_v_alpha = graph.get_tensor_by_name('Input_V_Alpha:0')

            if self.use_j_alpha is True:
                X_j_alpha = graph.get_tensor_by_name('Input_J_Alpha:0')

            if self.use_hla:
                X_hla = graph.get_tensor_by_name('HLA:0')

            get_obj = graph.get_tensor_by_name(get)

            out_list = []
            sample_list = np.unique(sample_labels)
            for vars in get_batches([sample_list], batch_size=batch_size, random=False):
                var_idx = np.where(np.isin(sample_labels, vars[0]))[0]
                lb = LabelEncoder()
                lb.fit(vars[0])
                i = lb.transform(sample_labels[var_idx])

                OH = OneHotEncoder(categories='auto')
                sp = OH.fit_transform(i.reshape(-1, 1)).T
                sp = sp.tocoo()
                indices = np.mat([sp.row, sp.col]).T

                feed_dict = {X_Freq: freq[var_idx],
                             X_Counts: counts[var_idx],
                             sp_i: indices,
                             sp_v: sp.data,
                             sp_s: sp.shape}

                if self.use_alpha is True:
                    feed_dict[X_Seq_alpha_v] = X_Seq_alpha[var_idx]
                if self.use_beta is True:
                    feed_dict[X_Seq_beta_v] = X_Seq_beta[var_idx]

                if self.use_v_beta is True:
                    feed_dict[X_v_beta] = v_beta_num[var_idx]

                if self.use_d_beta is True:
                    feed_dict[X_d_beta] = d_beta_num[var_idx]

                if self.use_j_beta is True:
                    feed_dict[X_j_beta] = j_beta_num[var_idx]

                if self.use_v_alpha is True:
                    feed_dict[X_v_alpha] = v_alpha_num[var_idx]

                if self.use_j_alpha is True:
                    feed_dict[X_j_alpha] = j_alpha_num[var_idx]

                if self.use_hla:
                    feed_dict[X_hla] = hla_data_seq_num[var_idx]

                out_list.append(sess.run(get_obj,feed_dict=feed_dict))

        out_list = np.vstack(out_list)
        return sample_list, out_list

    def Sample_Inference(self,sample_labels=None,alpha_sequences=None, beta_sequences=None, v_beta=None, d_beta=None, j_beta=None,
                  v_alpha=None, j_alpha=None, p=None,hla=None,freq=None,counts=None, batch_size=10,models=None,return_dist=False):

        """
        Predicting outputs of sample/repertoire model on new data

        This method allows a user to take a pre-trained sample/repertoire classifier
        and generate outputs from the model on new data. This will return predicted probabilites
        for the given classes for the new data. If the model has been trained in Monte-Carlo or K-Fold Cross-Validation,
        there will be a model created for each iteration of the cross-validation. if the 'models' parameter is left as None,
        this method will conudct inference for all models trained in cross-validation and output the average predicted value per sample
        along with the distribution of predictions for futher downstream use. For example, by looking at the distribution of
        predictions for a given sample over all models trained, one can determine which samples have a high level of certainty
        in their predictions versus those with lower level of certainty. In essense, by training a multiple models in cross-validation schemes,
        this can allow the user to generate a distribution of predictions on a per-sample basis which provides a better understanding of the prediction.
        Alternatively, one can choose to fill in the the models parameter with a list of models the user wants to use for inference.

        To load data from directories, one can use the Get_Data method from the base class to automatically
        format the data into the proper format to be then input into this method.

        One can also use this method to get per-sequence predictions from the sample/repertoire classifier. To do this,
        provide all inputs except for sample_labels. The method will then return an array of dimensionality [N,n_classes] where
        N is the number of sequences provided. When using the method in this way, be sure to change the batch_size is adjusted to a larger
        value as 10 sequences per batch will be rather slow. We recommend changing into the order of thousands (i.e. 10 - 100k).

        Inputs
        ---------------------------------------

        sample_labels: ndarray of strings
            A 1d array with sample labels for the sequence.

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

        counts: ndarray of ints
            A 1d array with the counts for each sequence.

        freq: ndarray of float values
            A 1d array with the frequencies for each sequence.

        hla: ndarray of tuples/arrays
            To input the hla context for each sequence fed into DeepTCR, this will need to formatted
            as an ndarray that is (N,) for each sequence where each entry is a tuple or array of strings referring
            to the alleles seen for that sequence.
                ('A*01:01', 'A*11:01', 'B*35:01', 'B*35:02', 'C*04:01')

            If the model used for inference was trained to use HLA-supertypes, one should still enter the HLA
            in the format it was provided to the original model (i.e. A0101). This mehthod will then convert those
            HLA alleles into the appropriaet supertype designation. The HLA alleles DO NOT need to be provided to
            this method in the supertype designation.

        p: multiprocessing pool object
            a pre-formed pool object can be passed to method for multiprocessing tasks.

        batch_size: int
            Batch size for inference.

        models: list
            List of models in Name_Of_Object/models to use for inference. If left as None, this method will use all models
            in that directory.

        return_dist: bool
            If the user wants to also return teh distribution of sample/sequence predictions over all models used for inference,
            one should set this value to True.

        Returns

        self.Inference_Sample_List: ndarray
            An array with the sample list corresponding the predicted probabilities.

        self.Inference_Pred: ndarray
            An array with the predicted probabilites for all classes. These represent the average probability
            of all models used for inference.

        self.Inference_Pred_Dict: dict
            A dictionary of predicted probabilities for the respective classes. These represent the average probability
            of all models used for inference.

        self.Inference_Pred_Dist: ndarray
            An array with the predicted probabilites for all classes on a per model basis.
            shape = [Number of Models, Number of Samples, Number of Classes]

        If sample_labels is not provided, the method will perform per-sequence predictions and will return the an array
        of [N,n_classes]. If return_dist is set to True, the method will return two outputs. One containing the mean predictions
        and the other containing the full distribution over all models.

        ---------------------------------------

        """
        inputs = [alpha_sequences,beta_sequences,v_beta,d_beta,j_beta,v_alpha,j_alpha, hla]

        for i in inputs:
            if i is not None:
                assert isinstance(i,np.ndarray),'Inputs into DeepTCR must come in as numpy arrays!'

        inputs = [alpha_sequences,beta_sequences,v_beta,d_beta,j_beta,v_alpha,j_alpha,hla]
        for i in inputs:
            if i is not None:
                len_input = len(i)
                break

        seq_inf = False
        if sample_labels is None:
            sample_labels = np.array([str(x) for x in range(len_input)])
            seq_inf = True

        model_type, get  = load_model_data(self)

        if p is None:
            p = Pool(40)

        if alpha_sequences is not None:
            args = list(
                zip(alpha_sequences, [self.aa_idx] * len(alpha_sequences), [self.max_length] * len(alpha_sequences)))
            result = p.starmap(Embed_Seq_Num, args)
            sequences_num = np.vstack(result)
            X_Seq_alpha = np.expand_dims(sequences_num, 1)
        else:
            X_Seq_alpha = np.zeros(shape=[len_input,self.max_length])
            X_Seq_alpha = np.expand_dims(X_Seq_alpha, 1)
            alpha_sequences = np.asarray([None] * len_input)

        if beta_sequences is not None:
            args = list(
                zip(beta_sequences, [self.aa_idx] * len(beta_sequences), [self.max_length] * len(beta_sequences)))
            result = p.starmap(Embed_Seq_Num, args)
            sequences_num = np.vstack(result)
            X_Seq_beta = np.expand_dims(sequences_num, 1)
        else:
            X_Seq_beta = np.zeros(shape=[len_input,self.max_length])
            X_Seq_beta = np.expand_dims(X_Seq_beta, 1)
            beta_sequences = np.asarray([None] * len_input)

        if v_beta is not None:
            v_beta = v_beta.astype(self.lb_v_beta.classes_.dtype)
            i_r = np.where(np.invert(np.isin(v_beta,self.lb_v_beta.classes_)))[0]
            v_beta[i_r]= np.random.choice(self.lb_v_beta.classes_,len(i_r))
            v_beta_num = self.lb_v_beta.transform(v_beta)
        else:
            v_beta_num = np.zeros(shape=[len_input])
            v_beta = np.asarray([None] * len_input)

        if d_beta is not None:
            d_beta = d_beta.astype(self.lb_d_beta.classes_.dtype)
            i_r = np.where(np.invert(np.isin(d_beta,self.lb_d_beta.classes_)))[0]
            d_beta[i_r]= np.random.choice(self.lb_d_beta.classes_,len(i_r))
            d_beta_num = self.lb_d_beta.transform(d_beta)
        else:
            d_beta_num = np.zeros(shape=[len_input])
            d_beta = np.asarray([None] * len_input)

        if j_beta is not None:
            j_beta = j_beta.astype(self.lb_j_beta.classes_.dtype)
            i_r = np.where(np.invert(np.isin(j_beta,self.lb_j_beta.classes_)))[0]
            j_beta[i_r]= np.random.choice(self.lb_j_beta.classes_,len(i_r))
            j_beta_num = self.lb_j_beta.transform(j_beta)
        else:
            j_beta_num = np.zeros(shape=[len_input])
            j_beta = np.asarray([None] * len_input)

        if v_alpha is not None:
            v_alpha = v_alpha.astype(self.lb_v_alpha.classes_.dtype)
            i_r = np.where(np.invert(np.isin(v_alpha,self.lb_v_alpha.classes_)))[0]
            v_alpha[i_r]= np.random.choice(self.lb_v_alpha.classes_,len(i_r))
            v_alpha_num = self.lb_v_alpha.transform(v_alpha)
        else:
            v_alpha_num = np.zeros(shape=[len_input])
            v_alpha = np.asarray([None] * len_input)

        if j_alpha is not None:
            j_alpha = j_alpha.astype(self.lb_j_alpha.classes_.dtype)
            i_r = np.where(np.invert(np.isin(j_alpha,self.lb_j_alpha.classes_)))[0]
            j_alpha[i_r]= np.random.choice(self.lb_j_alpha.classes_,len(i_r))
            j_alpha_num = self.lb_j_alpha.transform(j_alpha)
        else:
            j_alpha_num = np.zeros(shape=[len_input])
            j_alpha = np.asarray([None] * len_input)

        if hla is not None:
            if self.use_hla_sup:
                hla = supertype_conv_op(hla,self.keep_non_supertype_alleles)
            hla_data_seq_num = self.lb_hla.transform(hla)
        else:
            try:
                hla_data_seq_num = np.zeros(shape=[len_input,self.lb_hla.classes_.shape[0]])
            except:
                hla_data_seq_num = np.zeros(shape=[len_input,1])
                pass

        if p is None:
            p.close()
            p.join()

        if (counts is None) & (freq is None):
            counts = np.ones(shape=len_input)

        if counts is not None:
            count_dict={}
            for s in np.unique(sample_labels):
                idx = sample_labels==s
                count_dict[s]=np.sum(counts[idx])

        if freq is None:
            freq = []
            for c,n in zip(counts,sample_labels):
                freq.append(c/count_dict[n])
            freq = np.asarray(freq)

        data = graph_object()
        data.X_Seq_alpha = X_Seq_alpha
        data.X_Seq_beta = X_Seq_beta
        data.v_beta_num = v_beta_num
        data.d_beta_num = d_beta_num
        data.j_beta_num = j_beta_num
        data.v_alpha_num = v_alpha_num
        data.j_alpha_num = j_alpha_num
        data.hla_data_seq_num = hla_data_seq_num
        data.freq = freq
        data.counts = counts
        data.batch_size = batch_size
        data.sample_labels = sample_labels
        data.get = get

        if models is None:
            directory = os.path.join(self.Name,'models')
            models = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
            models = [f for f in models if not f.startswith('.')]

        predicted = []
        for m in models:
            sample_list,pred = self._inf(data,model=m)
            predicted.append(pred)

        predicted_dist = []
        for p in predicted:
            predicted_dist.append(np.expand_dims(p,0))
        predicted_dist = np.vstack(predicted_dist)

        self.Inference_Sample_List = sample_list
        self.Inference_Pred_Dist = predicted_dist
        self.Inference_Pred =  np.mean(predicted_dist,0)

        if self.regression is False:
            DFs = []
            for ii,c in enumerate(self.lb.classes_,0):
                df_temp = pd.DataFrame()
                df_temp['Samples'] = self.Inference_Sample_List
                df_temp['Pred'] = self.Inference_Pred[:,ii]
                DFs.append(df_temp)
            self.Inference_Pred_Dict = dict(zip(self.lb.classes_,DFs))

        if seq_inf:
            df_temp['Samples'] = df_temp['Samples'].astype(int)
            df_temp.sort_values(by='Samples', inplace=True)
            resort_idx = np.array(list(df_temp.index))

            if return_dist:
                return self.Inference_Pred[resort_idx], self.Inference_Pred_Dist[resort_idx]
            else:
                return self.Inference_Pred[resort_idx]























