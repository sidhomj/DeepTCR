import tensorflow as tf
import numpy as np
from DeepTCR.functions.custom_act import *

def GCN(GO,Features,num_clusters):
    A= Get_Adjacency_Matrix(GO,Features)
    GO.A = A
    X = Reshape_X(Features,GO.i,GO.j)
    Features = GCN_Features(GO,A,X,num_clusters)
    Features = Flatten_X(Features,GO.i,GO.j)
    return Features

def Get_Adjacency_Matrix(GO,X):
    X = Reshape_X(X,GO.i,GO.j)
    D = Pairwise_Distance_TF(X)
    A,_,_,_ = gbell(D,a_init=0.0)
    #id = tf.eye(tf.shape(D)[-1])[tf.newaxis,:,:]
    #A = A*id
    GO.A = A
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
    X = tf.matmul(A,X)
    Z = tf.layers.dense(X, num_clusters, activation=tf.nn.relu, use_bias=False)

    return Z

def Pairwise_Distance_TF(A):
   r = tf.reduce_sum(A*A,-1)
   r = tf.expand_dims(r,-1)
   D = tf.sqrt(tf.nn.relu(r - 2 *tf.matmul(A,tf.linalg.transpose(A)) + tf.linalg.transpose(r)))
   return D

def MultiLevel_Dropout(X,num_masks=2,activation=tf.nn.relu,use_bias=True,
                       rate=0.2,units=12,name='ml_weights'):
    out = []
    for i in range(num_masks):
        fc = tf.layers.dropout(X,rate=rate)
        if i==0:
            reuse=False
        else:
            reuse=True

        with tf.variable_scope(name,reuse=reuse):
            out.append(tf.layers.dense(fc,units=units,activation=activation,use_bias=use_bias))

    return tf.reduce_mean(tf.stack(out),0)

