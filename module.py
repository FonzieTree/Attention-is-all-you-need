# This project is inspired by https://www.github.com/kyubyong/tacotron
# December 2017 by Shuming Fang. 
# fangshuming519@gmail.com.
# https://github.com/FonzieTree
# -*- coding: utf-8 -*-
import numpy as np
from hyperparams import Hyperparams as hp
def normalize(inputs):
    # Layer mean
    params_shape = inputs.shape[-1]
    mean = np.mean(inputs, axis=-1, keepdims=True)
    variance = np.std(inputs, axis=-1, keepdims=True)
    outputs = (inputs - mean) / ((variance + 1e-8) ** (.5))
    
    return [mean, variance, outputs]

def de_normalize(mean,
		 variance,
		 outputs):
    dinputs = mean + ((variance + 1e-8)**.5)*outputs
    return dinputs

def embedding(inputs, 
              lookup_table, 
              num_units, 
              zero_pad=True, 
              scale=False,
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
                        scale=False):
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
    outputs2 = outputs1/(K.shape[2]** 0.5)
    outputs2[outputs2==0] = -2**32 + 1
    # SoftMax
    outputs3 = np.exp(outputs2)
    outputs4 = np.sum(outputs3,axis=(1))
    outputs5 = np.array([outputs3[i,:,:]/outputs4[i,:] for i in range(hp.batch_size)])
    outputs6 = np.array([np.dot(outputs5[i,:,:], V[i,:,:]) for i in range(hp.batch_size)])
    outputs7 = np.array([np.dot(outputs6[i,:,:], attention_w[3,:,:]) for i in range(hp.batch_size)])
    # Add residual connections
    outputs8 = outputs7 + queries
    return [outputs1, outputs2, outputs5,  outputs6, outputs7, outputs8, Q, K, V, queries, keys]

def de_multihead_attention(outputs1, outputs2, outputs5, outputs6, outputs7, outputs, Q, K, V, queries, keys, attention_w):
    dattention_w = np.zeros((attention_w.shape)) 
    doutputs7 = outputs
    doutputs6 = np.array([np.dot(doutputs7[i,:,:], attention_w[3,:,:].T) for i in range(hp.batch_size)])
    dV = np.array([np.dot(outputs5[i,:,:].T, doutputs6[i,:,:]) for i in range(hp.batch_size)])
    doutputs5 = np.array([np.dot(doutputs6[i,:,:], V[i, :, :].T) for i in range(hp.batch_size)])
    doutputs2 = doutputs5/(hp.batch_size*hp.maxlen)
    doutputs2[outputs2==0] = 0
    doutputs1 = doutputs2 * (K.shape[2]** 0.5)
    dQ = np.array([np.dot(doutputs1[i, :, :], K[i,:,:]) for i in range(hp.batch_size)])
    dK = np.array([np.dot(Q[i, :, :].T, doutputs1[i, :, :]).T for i in range(hp.batch_size)])
    dqueries = np.dot(dQ, attention_w[0, :, :].T) + outputs
    dkeys = (np.dot(dK, attention_w[1, :, :]) + np.dot(dV, attention_w[2, :, :]))/2
    d0 = np.array([np.dot(queries[i, :, :].T, Q[i, :, :]) for i in range(hp.batch_size)])/hp.batch_size
    dattention_w[0, :, :] = np.sum(d0, axis = 0)
    d1 = np.array([np.dot(keys[i, :, :].T, K[i, :, :]) for i in range(hp.batch_size)])/hp.batch_size
    dattention_w[1, :, :] = np.sum(d1, axis = 0)
    d2 = np.array([np.dot(keys[i, :, :].T, V[i, :, :]) for i in range(hp.batch_size)])/hp.batch_size
    dattention_w[2, :, :] = np.sum(d2, axis = 0)
    d3 = np.array([np.dot(outputs6[i,:,:].T, doutputs7[i,:,:]) for i in range(hp.batch_size)])/hp.batch_size
    dattention_w[3, :, :] = np.sum(d3, axis = 0)
    return [dqueries, dkeys, dattention_w]

def feedforward(inputs,
                w1,
                w2):
    outputs1 = np.dot(inputs,np.tile(w1,(512,1)))
    index = (outputs1<=0)
    outputs2 = np.maximum(0,outputs1)
    outputs3 = np.dot(outputs2,np.tile(w2,(2048,1)))
    outputs4 = outputs3 + inputs
    return [index, outputs1, outputs2, outputs3, outputs4]

def backward(inputs,
             w1,
	     w2,
             index,
             outputs1,
             outputs2,
             outputs3,
             outputs):
    doutputs3 = outputs
    doutputs2 = np.dot(doutputs3, np.tile(w2,(2048,1)).T)
    
    dw2 = np.array([np.dot(outputs2[i, :, :].T, doutputs3[i, :, :]) for i in range(hp.batch_size)])
    dw2 = np.sum(dw2, axis = (0,1))/(hp.batch_size*2048)
    doutputs2[index] = 0
    doutputs1 = doutputs2
    
    dw1 = np.array([np.dot(inputs[i, :, :].T, doutputs1[i, :, :]) for i in range(hp.batch_size)])
    dw1 = np.sum(dw1, axis = (0,1))/(hp.batch_size*512)
    
    dinputs = np.dot(doutputs1, np.tile(w1, (512,1)).T)
    dinputs = dinputs + outputs
    return [dw1, dw2, dinputs]

    

            
