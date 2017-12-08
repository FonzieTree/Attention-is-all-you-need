# This project is inspired by https://www.github.com/kyubyong/tacotron
# December 2017 by Shuming Fang. 
# fangshuming519@gmail.com.
# https://github.com/FonzieTree
# -*- coding: utf-8 -*-
import numpy as np
from hyperparams import Hyperparams as hp
def normalize(inputs, 
              epsilon,
              beta,
              gamma):
    # Layer mean
    params_shape = inputs.shape[-1]
    mean = np.mean(inputs, axis=-1, keepdims=True)
    variance = np.std(inputs, axis=-1, keepdims=True)
    normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
    outputs = gamma * normalized + beta
    
    return mean, variance, normalized, outputs
def de_normalize(beta,
				 gamma,
				 mean,
				 variance,
				 normalized,
				 outputs):
    dinputs = mean + ((variance + 1e-8)**.5)*(outputs - bata)/gamma
    dgamma = (outputs - beta)/normalized
    dbeta = (outputs - gamma * normalized)
    return dbeta, dgamma, dinputs

def embedding(inputs, 
              lookup_table, 
              num_units, 
              zero_pad=True, 
              scale=True,
              reuse=None):
    if zero_pad:
        lookup_table = np.concatenate((np.zeros((1,num_units), dtype=int), lookup_table[1:,:]), axis=0) 
    outputs = lookup_table[inputs]
        
    if scale:
        outputs = outputs * (num_units ** 0.5) 
            
    return outputs
    

def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True):
    N, T = inputs.shape
    position_ind = np.tile(np.array(range(T)),(N,1))

    # First part of the PE function: sin and cos argument
    position_enc = np.array([[pos/np.power(10000, 2.*i/num_units) for i in range(num_units)]
        for pos in range(T)])

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

    # Convert to a tensor
    lookup_table = position_enc

    if zero_pad:
        lookup_table = np.concatenate((np.zeros((1,num_units), dtype=int), lookup_table[1:,:]), axis=0)
    outputs = lookup_table[position_ind]

    if scale:
        outputs = outputs * (num_units ** 0.5)

    return outputs

def multihead_attention(queries,
                        keys,
                        attention_w):
    Q = np.dot(queries, attention_w[0,:,:])
    K = np.dot(keys, attention_w[1,:,:])
    V = np.dot(keys, attention_w[2,:,:])
    # Multiplication
    outputs1 = np.array([np.dot(Q[i,:,:], K[i,:,:].T) for i in range(hp.batch_size)])
    # Scale
    outputs2 = outputs1/(K.shape[1]** 0.5)
    # SoftMax
    index = np.argwhere(outputs2<=0)
    outputs3 = np.maximum(0,outputs2)
    outputs4 = np.array([np.dot(outputs3[i,:,:], V[i,:,:]) for i in range(hp.batch_size)])
    outputs5 = np.array([np.dot(outputs4[i,:,:], attention_w[3,:,:]) for i in range(hp.batch_size)])
    # Add residual connections
    outputs6 = outputs5 + queries
    return index, outputs1, outputs2, outputs3,  outputs4,  outputs5,  outputs6, Q, K, V

def de_multihead_attention(index, outputs1, outputs2, outputs3,  outputs4,  outputs5, outputs, dattention_w, Q, K, V):
    doutputs5 = outputs
    doutputs4 = np.array([np.dot(doutputs5[i,:,:], dattention_w[3,:,:].T) for i in range(hp.batch_size)])
    dattention_w[3,:,:] = np.array([np.dot(outputs4[i,:,:].T, doutputs5[i,:,:]) for i in range(hp.batch_size)])
    dV = np.array([np.dot(outputs3[i,:,:].T, doutputs4[i,:,:]) for i in range(hp.batch_size)])
    doutputs3 = np.array([np.dot(doutputs4[i,:,:], V[i, :, :].T) for i in range(hp.batch_size)])
    doutputs2 = doutputs3[index]
    doutputs1 = doutputs2(K.shape[1]** 0.5)
    dQ = np.array([np.dot(doutputs1, K[i,:,:]) for i in range(hp.batch_size)])
    dK = np.array([np.dot(Q.T, doutputs1).T for i in range(hp.batch_size)])
    dqueries = np.dot(dQ, attention_w[0, :, :].T)
    dkeys = np.dot(dK, attention[1, :, :]) + np.dot(dV, attention[2, :, :])
    return dqueris, dkeys, dattention_w

def feedforward(inputs,
                w1,
                w2):
    outputs1 = np.dot(inputs,w1)
    index = np.argwhere(outputs1<=0)
    outputs2 = np.maximum(0,outputs1)
    outputs3 = np.dot(outputs2,w2)
    outputs4 = outputs3 + inputs    
    return index, outputs1, outputs2, outputs3, outputs4

def backward(w1,
			 w2,
             index,
             outputs1,
             outputs2,
             outputs3,
             outputs):
    doutputs3 = outputs
    doutputs2 = np.dot(doutputs3, w2.T)
    dw2 = np.dot(outputs2.T, doutputs3)
    doutputs2[index] = 0
    doutputs1 = np.dot(doutputs2,w1.T)
    dw1 = np.dot(doutputs1, inputs)
    dinputs = np.dot(doutputs1, w1.T)
    dinputs = dinputs + doutputs
    return dw1, dw2, dinputs

def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)
