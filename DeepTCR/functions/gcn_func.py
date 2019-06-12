import tensorflow as tf
import numpy as np
from DeepTCR.functions.custom_act import *

def GCN(GO,Features,num_clusters):
    A = Get_Adjacency_Matrix(GO,Features)
    GO.A = A
    X = Reshape_X(Features,GO.i,GO.j)
    Features,S = GCN_Features(GO,A,X,num_clusters)
    Features = Flatten_X(Features,GO.i,GO.j)
    GO.S = Flatten_X(S,GO.i,GO.j)
    return Features

def Get_Adjacency_Matrix(GO,X):
    X = Reshape_X(X,GO.i,GO.j)
    D = Pairwise_Distance_TF(X)
    A = gbell(D,a_init=1.0)
    return A

def Reshape_X(X,i,j):
    return tf.scatter_nd(tf.concat((i[:, tf.newaxis], j[:, tf.newaxis]), -1),
                  X, [tf.reduce_max(i) + 1, tf.reduce_max(j) + 1,X.shape[-1]] )
def Flatten_X(X,i,j):
    return tf.gather_nd(X,tf.concat((i[:, tf.newaxis], j[:, tf.newaxis]), -1))

def GCN_Features(GO,A,X,num_clusters=12):
    #Norm
    D_norm = tf.sqrt(1 / tf.reduce_sum(A, -1))
    Lap_D = tf.expand_dims(D_norm, -1) * A * tf.expand_dims(D_norm, -2)
    A = Lap_D

    #Graph Convolution
    Z = tf.layers.dense(tf.matmul(A,X), num_clusters, activation=tf.nn.relu, use_bias=False)

    return Z

def Pairwise_Distance_TF(A):
   r = tf.reduce_sum(A*A,-1)
   r = tf.expand_dims(r,-1)
   D = tf.sqrt(tf.nn.relu(r - 2 *tf.matmul(A,tf.linalg.transpose(A)) + tf.linalg.transpose(r)))
   return D

