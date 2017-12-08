import numpy as np
from modules import *
np.random.seed(0)
print('loading vocabulary...')
de2idx, idx2de = load_de_vocab()
en2idx, idx2en = load_en_vocab()
print('done')
print('loading datasets...')
X, Y = load_train_data()
print('done')
num_samples = X.shape[0]
index = np.random.randint(0,num_samples,hp.batch_size)
x = X[index,:]
y = Y[index,:]
dff = 2048 # dimention of inner layer
# Some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength
epoch = 2


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

decoder_w5 = 0.01*np.random.randn(dff,hp.hidden_units)

decoder_w6 = 0.01*np.random.randn(hp.hidden_units,len(en2idx))

ddecoder_w2 = 0.01*np.random.randn(4,hp.hidden_units,hp.hidden_units)

decoder_beta1 = np.zeros((hp.hidden_units))

decoder_gamma1 = np.ones((hp.hidden_units))

decoder_beta2 = np.zeros((hp.hidden_units))

decoder_gamma2 = np.ones((hp.hidden_units))

decoder_beta3 = np.zeros((hp.hidden_units))

decoder_gamma3 = np.ones((hp.hidden_units))

lookup_table2 = 0.01*np.random.randn(len(en2idx), hp.hidden_units)

# Encoder
encoder1 = embedding(x,lookup_table1,num_units=hp.hidden_units,scale=True)

position_encoder = positional_encoding(x,num_units=hp.hidden_units,zero_pad=True,scale=True)

encoder2 = encoder1 + position_encoder

encoder3 = multihead_attention(queries=encoder2,keys=encoder2,attention_w=encoder_w1)

mean1, variance1, normalized1, encoder4 = normalize(inputs = encoder3,epsilon = 1e-8,beta = encoder_beta1,gamma = encoder_gamma1)

index1, encoder51, encoder52,encoder53,encoder54 = feedforward(encoder4,encoder_w2,encoder_w3)

mean2, variance2, normalized2, encoder6 = normalize(encoder54,epsilon = 1e-8,beta=encoder_beta2,gamma=encoder_gamma2)

# Decoder
decoder_inputs = np.concatenate((np.ones((hp.batch_size,1), dtype=int)*2, y[:,:-1]), axis=1) # 2:<S>

decoder1 = embedding(decoder_inputs,lookup_table2,num_units=hp.hidden_units,scale=True)

position_decoder = positional_encoding(decoder_inputs,num_units=hp.hidden_units,zero_pad=True,scale=True)

decoder2 = decoder1 + position_decoder

decoder3 = multihead_attention(queries=decoder2,keys=decoder2,attention_w=decoder_w1)

mean3, variance3, normalized3, decoder4 = normalize(inputs = decoder3,epsilon = 1e-8, beta = decoder_beta1, gamma = decoder_gamma1)

index3, decoder51, decoder52, decoder53, decoder54, decoder55, decoder56, Q3, K3, V3 = multihead_attention(queries = decoder4,keys=encoder2,attention_w=decoder_w2)

mean4, variance4, normalized4, decoder6 = normalize(inputs = decoder5,epsilon = 1e-8, beta = decoder_beta2, gamma = decoder_gamma2)

index4, decoder71,decoder72,decoder73,decoder74 = feedforward(decoder6, decoder_w3, decoder_w4)

mean5, variance5, normalized5, decoder8 = normalize(decoder74, epsilon = 1e-8, beta=decoder_beta3, gamma = decoder_gamma3)

scores = np.dot(decoder8, decoder_w6)

exp_scores = np.exp(scores)

probs = exp_scores/np.sum(exp_scores,axis=-1,keepdims=True)

# Backpropegation
for i in range(epoch):
    
    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(np.array([probs[j][range(hp.maxlen),y[i]] for j in range(hp.batch_size)]))

    data_loss = np.sum(corect_logprobs)/hp.batch_size

    reg_loss = 0.5*reg*(np.sum(encoder_w1*encoder_w1) + np.sum(encoder_w2*encoder_w2) + np.sum(encoder_w3*encoder_w3) +

    np.sum(lookup_table1*lookup_table1) + np.sum(decoder_w1*decoder_w1) + np.sum(decoder_w2*decoder_w2) + np.sum(decoder_w3*decoder_w3) +

    np.sum(decoder_w4*decoder_w4) + np.sum(decoder_w5*decoder_w5) + np.sum(decoder_w6*decoder_w6) + np.sum(lookup_table2*lookup_table2))    

    loss = data_loss + reg_loss

    print("iteration %d: loss %f" % (i, loss))
  
    # compute the gradient on scores
    dscores = probs
    for j in range(hp.batch_size):
        dscores[j][range(hp.maxlen),y[j]] -=1
    
    dscores /= hp.batch_size
    ddecoder8 = np.dot(dscores, decoder_w6.T)
    ddecoder6_w6 = np.array([np.dot(decoder8[j,:,:].T,dscores[j,:,:]) for j in range(hp.batch_size)])
    
    # de_layer normalization
    ddecoder_beta3, ddecoder_gamma3, ddecoder74 = de_normalize(decoder_beta3, decoder_gamma3, mean5, variance5, normalized5, ddecoder8)
    # dense backward
    ddecoder_w3, ddecoder_w4, ddecoder6 = backward(w1 = decoder_w3,
	                                               w2 = decoder_w4,
												   index = index4,
												   outputs1 = decoder71,
												   outputs2 = decoder72,
												   outputs3 = decoder73,
												   outputs = ddecoder74) 

    # de_layer normalization
    ddecoder_beta2, ddecoder_gamma2, ddecoder56 = de_normalize(beta = decoder_beta2,
	                                                           gamma = decoder_gamma2,
															   mean = mean4,
															   variance = variance4,
															   normalized = normalized4,
															   outputs = ddecoder6)
	# multi_head attention backward
    ddecoder4, dencoder2, ddecoder_w2 = de_multihead_attention(index = index3, outputs1 = decoder51,
								   outputs2 = decoder52,
								   outputs3 = decoder53,
								   outputs4 = decoder54,
								   outputs5 = decoder55,
								   outputs = decoder56, 
								   Q = Q3,
								   K = K3,
								   V = V3,
								   attention_w = decoder_w2)
