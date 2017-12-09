# This project is inspired by https://www.github.com/kyubyong/tacotron
# December 2017 by Shuming Fang. 
# fangshuming519@gmail.com.
# https://github.com/FonzieTree
from __future__ import print_function
from hyperparams import Hyperparams as hp
from data_load import load_train_data, get_batch_data, load_de_vocab, load_en_vocab
import os, codecs
from tqdm import tqdm
import numpy as np
from modules2 import *
np.random.seed(0)
print('loading vocabulary...')
de2idx, idx2de = load_de_vocab()
en2idx, idx2en = load_en_vocab()
print('done')
print('loading datasets...')
X, Y = load_train_data()
print('done')
num_samples = X.shape[0]
dff = 2048 # dimention of inner layer
# Some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength
epoch = 2
lr = 1

# Encoder parameters
encoder_w1 = 0.01*np.random.randn(4,hp.hidden_units,hp.hidden_units)
encoder_w2 = 0.01*np.random.randn(hp.hidden_units,dff)
encoder_w3 = 0.01*np.random.randn(dff,hp.hidden_units)
encoder_beta1 = np.zeros((hp.hidden_units))
encoder_gamma1 = np.ones((hp.hidden_units))
encoder_beta2 = np.zeros((hp.hidden_units))
encoder_gamma2 = np.ones((hp.hidden_units))
lookup_table1 = 0.01*np.random.randn(len(de2idx), hp.hidden_units)
# Decoder parameters
decoder_w1 = 0.01*np.random.randn(4,hp.hidden_units,hp.hidden_units)
decoder_w2 = 0.01*np.random.randn(4,hp.hidden_units,hp.hidden_units)
decoder_w3 = 0.01*np.random.randn(hp.hidden_units, dff)
decoder_w4 = 0.01*np.random.randn(dff,hp.hidden_units)
decoder_w5 = 0.01*np.random.randn(hp.hidden_units,len(en2idx))
ddecoder_w2 = 0.01*np.random.randn(4,hp.hidden_units,hp.hidden_units)
decoder_beta1 = np.zeros((hp.hidden_units))
decoder_gamma1 = np.ones((hp.hidden_units))
decoder_beta2 = np.zeros((hp.hidden_units))
decoder_gamma2 = np.ones((hp.hidden_units))
decoder_beta3 = np.zeros((hp.hidden_units))
decoder_gamma3 = np.ones((hp.hidden_units))
lookup_table2 = 0.01*np.random.randn(len(en2idx), hp.hidden_units)

for i in range(epoch):
    select = np.random.randint(0,num_samples,hp.batch_size)
    x = X[select, :]
    y = Y[select, :]
    # Forward path
    # Encoder
    encoder1 = embedding(x,lookup_table1,num_units=hp.hidden_units,scale=True)
    position_encoder = positional_encoding(x,num_units=hp.hidden_units,zero_pad=True,scale=True)
    encoder2 = encoder1 + position_encoder
    encoder3 = multihead_attention(queries=encoder2,keys=encoder2,attention_w=encoder_w1)
    encoder4 = normalize(inputs = encoder3[6], beta = encoder_beta1, gamma = encoder_gamma1)
    encoder5 = feedforward(encoder4[3],encoder_w2,encoder_w3)
    encoder6 = normalize(encoder5[4],beta=encoder_beta2,gamma=encoder_gamma2)
    # Decoder
    decoder_inputs = np.concatenate((np.ones((hp.batch_size,1), dtype=int)*2, y[:,:-1]), axis=1) # 2:<S>
    decoder1 = embedding(decoder_inputs,lookup_table2,num_units=hp.hidden_units,scale=True)
    position_decoder = positional_encoding(decoder_inputs,num_units=hp.hidden_units,zero_pad=True,scale=True)
    decoder2 = decoder1 + position_decoder
    decoder3 = multihead_attention(queries=decoder2,keys=decoder2,attention_w=decoder_w1)
    decoder4 = normalize(inputs = decoder3[6], beta = decoder_beta1, gamma = decoder_gamma1)
    decoder5 = multihead_attention(queries = decoder4[3],keys=encoder6[3],attention_w=decoder_w2)
    decoder6 = normalize(inputs = decoder5[6], beta = decoder_beta2, gamma = decoder_gamma2)
    decoder7 = feedforward(inputs = encoder6[3], w1 = decoder_w3, w2 = decoder_w4)
    decoder8 = normalize(inputs = decoder7[4], beta=decoder_beta3, gamma = decoder_gamma3)
    scores = np.dot(decoder8[3], decoder_w5)
    exp_scores = np.exp(scores)
    probs = exp_scores/np.sum(exp_scores,axis=-1,keepdims=True)

    # Backpropegation   
    # compute the loss: average cross-entropy loss and regularization
    correct_logprobs = -np.log(np.array([probs[j][range(hp.maxlen),y[j]] for j in range(hp.batch_size)]))
    data_loss = np.sum(correct_logprobs)/hp.batch_size
    reg_loss = 0.5*reg*(np.sum(encoder_w1*encoder_w1) + np.sum(encoder_w2*encoder_w2) + np.sum(encoder_w3*encoder_w3) +
    np.sum(lookup_table1*lookup_table1) + np.sum(decoder_w1*decoder_w1) + np.sum(decoder_w2*decoder_w2) + np.sum(decoder_w3*decoder_w3) +
    np.sum(decoder_w4*decoder_w4) + np.sum(decoder_w5*decoder_w5) + np.sum(lookup_table2*lookup_table2))    
    loss = data_loss + reg_loss
    print("iteration %d: loss %f" % (i, loss))
    # compute the gradient on scores
    dscores = probs
    for j in range(hp.batch_size):
        dscores[j][range(hp.maxlen),y[j]] -=1
    
    dscores /= hp.batch_size
    ddecoder8 = np.dot(dscores, decoder_w5.T)
    ddecoder_w5 = np.array([np.dot(ddecoder8[j,:,:].T,dscores[j,:,:]) for j in range(hp.batch_size)])
    ddecoder_w5 = np.sum(ddecoder_w5, axis=0)/hp.batch_size
    # de_layer normalization
    ddecoder7 = de_normalize(decoder_beta3, decoder_gamma3, decoder8[0], decoder8[1], decoder8[2], ddecoder8)
    ddecoder_beta3 = ddecoder7[0]
    ddecoder_gamma3 = ddecoder7[1]
    # dense backward
    ddecoder6 = backward(w1 = decoder_w3,
	                 w2 = decoder_w4,
			 index = decoder7[0],
			 outputs1 = decoder7[1],
			 outputs2 = decoder7[2],
			 outputs3 = decoder7[3],
			 outputs = ddecoder7[2])
    ddecoder_w3 = ddecoder6[0]
    ddecoder_w4 = ddecoder6[1]
    

    # de_layer normalization
    ddecoder5 = de_normalize(beta = decoder_beta2,
	                     gamma = decoder_gamma2,
			     mean = decoder6[0],
			     variance = decoder6[1],
			     normalized = decoder6[2],
	         	     outputs = ddecoder6[2])
    ddecoder_beta2 = ddecoder5[0]
    ddecoder_gamma2 = ddecoder5[1]
    # multi_head attention3 backward
    dmulti3 = de_multihead_attention(index = decoder5[0],
                                     outputs1 = decoder5[1],
				     outputs2 = decoder5[2],
				     outputs3 = decoder5[3],
				     outputs4 = decoder5[4],
				     outputs5 = decoder5[5],
                                     Q = decoder5[6],
				     K = decoder5[7],
				     V = decoder5[8],
                                     queries = decoder5[9],
                                     keys = decoder5[10],
				     outputs = ddecoder5[2], 
				     attention_w = decoder_w2)
    ddecoder4 = dmulti3[0]
    dencoder6 = dmulti3[1]
    ddecoder_w2 = dmulti3[2]

    # de_layer normalization
    ddecoder3 = de_normalize(decoder_beta2, decoder_gamma2, decoder4[0], decoder4[1], decoder4[2], ddecoder4)
    ddecoder_beta2 = ddecoder3[0]
    ddecoder_gamma2 = ddecoder3[1]

    # multi_head attention2 backward
    dmulti2 = de_multihead_attention(index = decoder3[0],
                                     outputs1 = decoder3[1],
				     outputs2 = decoder3[2],
				     outputs3 = decoder3[3],
				     outputs4 = decoder3[4],
				     outputs5 = decoder3[5], 
				     Q = decoder3[6],
				     K = decoder3[7],
				     V = decoder3[8],
                                     queries = decoder3[9],
                                     keys = decoder3[10],
                                     outputs = ddecoder3[2],
				     attention_w = decoder_w1)
    ddecoder2 = (dmulti2[0] + dmulti2[1])/2
    ddecoder_w1 = dmulti2[2]

    # de_decoder_embedding
    ddecoder1 = ddecoder2
    ddecoder1 = ddecoder1/(hp.num_units ** 0.5)
    lookup_table2[decoder_inputs] += ddecoder1

    # de_layer normalization
    dencoder5 = de_normalize(encoder_beta2, encoder_gamma2, encoder6[0], encoder6[1], encoder6[2], dencoder6)
    dencoder_beta2 = dencoder5[0]
    dencoder_gamma2 = dencoder5[1]

    # backward
    dencoder4 = backward(w1 = encoder_w2,
	                 w2 = encoder_w3,
			 index = encoder5[0],
			 outputs1 = encoder5[1],
			 outputs2 = encoder5[2],
			 outputs3 = encoder5[3],
			 outputs = dencoder5[2])
    dencoder_w2 = dencoder4[0]
    dencoder_w3 = dencoder4[1]

    # de_layer normalization
    dencoder3 = de_normalize(encoder_beta1, encoder_gamma1, encoder4[0], encoder4[1], encoder4[2], dencoder4)
    dencoder_beta1 = dencoder3[0]
    dencoder_gamma1 = dencoder3[1]

    # multi_head attention1 backward
    dmulti1 = de_multihead_attention(index = encoder3[0],
                                     outputs1 = encoder3[1],
				     outputs2 = encoder3[2],
				     outputs3 = encoder3[3],
				     outputs4 = encoder3[4],
				     outputs5 = encoder3[5], 
				     Q = encoder3[6],
				     K = encoder3[7],
				     V = encoder3[8],
                                     queries = encoder3[9],
                                     keys = encoder3[10],
                                     outputs = dencoder3[2],
				     attention_w = encoder_w1)
    dencoder2 = (dmulti1[0] + dmulti1[1])/2
    dencoder_w1 = dmulti1[2]
    # de_encoder_embedding
    dencoder1 = dencoder2
    dencoder1 = dencoder1/(hp.num_units ** 0.5)
    lookup_table1[x] += dencoder1

    # parameter update
    # Encoder parameters
    encoder_w1 += -lr*dencoder_w1
    encoder_w2 += -lr*dencoder_w2
    encoder_w3 += -lr*dencoder_w2
    encoder_beta1 += -lr*dencoder_beta1
    encoder_gamma1 += -lr*dencoder_gamma1
    encoder_beta2 += -lr*dencoder_beta2 
    encoder_gamma2 += -lr*dencoder_gamma2

    # Decoder parameters
    decoder_w1 += -lr*ddecoder_w1 
    decoder_w2 += -lr*ddecoder_w2
    decoder_w3 += -lr*ddecoder_w3
    decoder_w4 += -lr*ddecoder_w4
    decoder_w5 += -lr*ddecoder_w5
    decoder_beta1 += -lr*ddecoder_beta1 
    decoder_gamma1 += -lr*ddecoder_gamma1 
    decoder_beta2 += -lr*ddecoder_beta2 
    decoder_gamma2 += -lr*ddecoder_gamma2
    decoder_beta3 += -lr*ddecoder_beta3 
    decoder_gamma3 += -lr*ddecoder_gamma3
    print(i, ' round finished')
