## Release History

### 1.1
Initial release including two methods for unsupervised learning (VAE & GAN). Also included
ability to handle paired alpha/beta data.

### 1.2
Second release included major refactoring in code to streamline and share methods across 
classes. Included ability for algorithm to accept v/d/j gene usage. Added more analytical fetures and
visualization methods. Removed GAN from unsupervised learning techniques. 

#### 1.2.7
On-graph clustering method introduced for repertoire classifier to improve classification performance.

#### 1.2.13
Ability for HLA information to be incorporated in the analysis of TCR-Seq. 

#### 1.2.24
Added ability to do regression for sequence-based model.

### 1.3
Third release including improved repertoire classification architecture. Details in method will follow 
in manuscript.

### 1.4
Fourth release includes major refactoring of code and adding more features including:

- Multi-Model Inference. When training the supervised sequence or repertoire classifier, in Monte-Carlo or K-Fold 
Cross Validation, a separate model will be stored for each cross-validation. When using the inference engine, 
users can choose to do an ensemble inference of some or many of the trained models.
- HLA Supertype Integration. Previous versions allowed users to provide HLA alleles for additional dimension of featurization
for the TCR. In this version, when providing HLA (either via the Get_Data or Load_Data methods), one now has the option of 
assigning the HLA-A and B genes to known supertypes for a more biologically functional representation of HLA.
- VAE now has an optional method by which to find a minimal number of latent features to model the underlying distribution
by incorporating a sparsity regularization on the latent layer. When using this feature, the VAE will provide a more 
compact latent space even if the initial latent_dim is unnecessarily high to model the distribution of data.
- Supervised models now have an additional option to use Multi-Sample Dropout to improve training and generalization.
- Incorporation of LogoMaker so now when Representative Sequences are generated along with enriched motifs,
seq logos are made and saved directly in the results folder under Motifs.
- Improved Motif Identification algorithm behind supervised method Representative_Sequences that uses a multinomial
linear model to identify which motifs are associated to predicted probabilites from neural network.
- Supervised Repertoire Model now able to do regression. By providing per-instance label with regression value with Load_Data method, this will automatically use the average of all instance level labels as the sample level value to regress the model against.

### 2.0
Fifth release: 

- Upgrading to use python 3.8 & Tensorflow 2.0
- For large repertoires, we have incorporated the ability to randomly subsample the repertoire over the course of training. Two methods of sub-sampling exist. 1) Completely randomly sampled from across the entire repertoire vs 2) randomly sampled as a probability function of the frequency of the TCR (at the amino acid level), meaning that a TCR with a 25% frequency will be sample at that probability.


