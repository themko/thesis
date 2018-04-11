'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
from __future__ import print_function
from keras.models import Sequential,Model,load_model,model_from_json,model_from_yaml
from keras.layers import Dense, Activation,Input,RepeatVector,Dropout
from keras.layers import LSTM, Lambda
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras import backend as K
from keras import objectives, regularizers
from keras.callbacks import TensorBoard
from keras.callbacks import Callback as Callback
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import regularizers
from collections import defaultdict
import keras
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import sys
import itertools
import argparse
import datetime

#Arguments in order:
#1:batch_size, 2:epochs, 3:retrain, 4:preprocessing, 5: percentage training size
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('train_type', type=str,nargs='?', default='kb',const='kb',
                    help='Parameter indicating if currently retraining model or just running existing one')
parser.add_argument('batch_sz', type=int,nargs='?', default=100,const=100,
                    help='Size of the batches in training')
parser.add_argument('eps', type=int,nargs='?', default=20,const=20,
                    help='Number of epochs')
parser.add_argument('retrain', type=str,nargs='?', default='retrain',const='retrain',
                    help='Parameter indicating if currently retraining model or just running existing one')
parser.add_argument('dropout', type=float,nargs='?', default=0.0,const=0.0,
                    help='Percentage of dataset in use')

args        = parser.parse_args()
train_type  = args.train_type
batch_size  = args.batch_sz
epochs      = args.eps
ignore_names= True
train_anew  = False
set_len     = True
dropout_rate= args.dropout

if(not args.retrain == 'load'):
    train_anew = True
    print('starting fresh')
else:
    print('loading')

predicate_dict      = defaultdict(set)
predicate_dict_test = defaultdict(set)

df      = pd.read_csv('../data/e2e-dataset/trainset.csv', delimiter=',')
df_test = pd.read_csv('../data/e2e-dataset/testset.csv' , delimiter=',')
tuples  = [tuple(x) for x in df.values]
for t in tuples:
    for r in t[0].split(','):
        r_ind1 = r.index('[')
        r_ind2 = r.index(']')
        rel = r[0:r_ind1].strip()
        rel_val = r[r_ind1+1:r_ind2]
        predicate_dict[rel].add(rel_val)    

#print(predicate_dict)
rel_lens        = [len(predicate_dict[p]) for p in predicate_dict.keys()]
rel_lens_test   = [len(predicate_dict_test[p]) for p in predicate_dict_test.keys()]

print('Padding sequences')
##COULD BE ADAPTED INTO SLICE IF NECESSARY
sentences = [a[-1] for a in tuples]
maxlen = max([len(b) for b in sentences])
##Abbreviated to make training more managable
maxlen = 20

for i,line in enumerate(sentences):
    while len(sentences[i])< maxlen:
        sentences[i]+=('-')

chars = sorted(list(set(''.join([item for sublist in sentences for item in sublist]))))
print('chars',chars)
print('total chars:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))



#Add language labels to vectorization
print('Vectorization...')
X = np.zeros((len(tuples), sum(rel_lens)), dtype=np.bool)
for i, tup in enumerate(tuples):
    for relation in tup[0].split(',')[1:]:
        rel_name = relation[0:relation.index('[')].strip()
        rel_value= relation[relation.index('[')+1:-1].strip()
        name_ind = predicate_dict.keys().index(rel_name)
        value_ind= list(predicate_dict[rel_name]).index(rel_value)
        j = sum(rel_lens[0:name_ind]) + value_ind
        X[i,j] = 1    

X_seq = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
print('X_seq',X_seq.shape)
for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                if(t<maxlen):
                    X_seq[i, t, char_indices[char]] = 1

intermediate_dim1   = 16
intermediate_dim2   = 24
intermediate_dim3   = 32
latent_dim          = 2
epsilon_std         = 1.0

#Sampling from N-dimm gaussian, for debugging decoder
decX = np.zeros((len(tuples),latent_dim))
for i in range(decX.shape[0]):
    for j in range(latent_dim):
        decX[i][j] = random.gauss(0,1)

"""
SEQ VAE:
"""
sen_len         = X_seq.shape[1]
voc_len         = X_seq.shape[2]

seq_x           = Input(shape=((sen_len,voc_len)),name='seq_inputs')
seq_h           = LSTM(intermediate_dim1, name='seq_h_layer_enc')(seq_x)
seq_z_mean      = Dense(latent_dim,name='seq_z_mean')(seq_h)
seq_z_log_sigma = Dense(latent_dim,name='seq_z_log_sigma')(seq_h)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon

seq_decoder_h           = Dense(intermediate_dim1,name='seq_dec_h')
seq_decoder_mean        = LSTM(voc_len,name='seq_dec_mean',return_sequences=True,activation='softmax')

seq_z                   = Lambda(sampling,output_shape=(latent_dim,))([seq_z_mean,seq_z_log_sigma])
seq_repeat_z            = RepeatVector(sen_len,name='repeat_z')(seq_z)
seq_h_decoded           = seq_decoder_h(seq_repeat_z)
seq_x_decoded_mean      = seq_decoder_mean(seq_h_decoded)

##ALTER LATER FOR SEQ
seq_decoder_input       = Input(shape=(latent_dim,))
_seq_repeat_z           = RepeatVector(sen_len)(seq_decoder_input)
_seq_h_decoded          = seq_decoder_h(_seq_repeat_z)
_seq_x_decoded_mean     = seq_decoder_mean(_seq_h_decoded)

"""
KB VAE: 
"""

#Decide how many of features of kb dataset to ignore
#Currently leaves sen_len = 10
data_offset = sum(rel_lens[0:5])
X = np.array([a[data_offset:] for a in X])

sen_len = X.shape[1] #Length of sequence
print('Build model...')

#VAE model based on same input
kb_x = Input(shape=(sen_len,),name='Inputs')
kb_enc1 = Dense(intermediate_dim1,name='kb_enc1')(kb_x)
kb_z_mean = Dense(latent_dim,name='z_mean')(kb_enc1)
kb_z_log_sigma= Dense(latent_dim,name='z_log_sigma')(kb_enc1)

kb_z = Lambda(sampling,output_shape=(latent_dim,))([kb_z_mean,kb_z_log_sigma])
kb_dec_h1           = Dense(intermediate_dim1,name='dec_h1',activation='sigmoid')
"""
kb_dec_h1_drop      = Dropout(dropout_rate)
kb_dec_h2           = Dense(intermediate_dim2,name='dec_h2',activation='relu')
kb_dec_h2_drop      = Dropout(dropout_rate)
kb_dec_h3           = Dense(intermediate_dim3,name='dec_h3',activation='relu')
"""
kb_decoder_output   = Dense(sen_len,name='dec_out',activation='sigmoid')

kb_h_decoded1       = kb_dec_h1(kb_z)
kb_x_decoded_mean   = kb_decoder_output(kb_h_decoded1)

kb_decoder_input    = Input(shape=(latent_dim,))

_kb_h_decoded1      = kb_dec_h1(kb_decoder_input)
_kb_x_decoded_mean  = kb_decoder_output(_kb_h_decoded1)


"""
COMBI VAE below:

"""

#how to redifine based on unknown true values?
def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + tf.reshape(z_log_sigma,[-1]) - K.square(tf.reshape(z_mean,[-1])) - K.exp(tf.reshape(z_log_sigma,[-1])), axis=-1)
    return  kl_loss + xent_loss

def vae_loss_seq(x, x_decoded_mean):
    #orig xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    xent_loss = K.mean(objectives.categorical_crossentropy(x, x_decoded_mean))
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)    #Extra K.mean added to use with sequential input shape
    #orig kl_loss = - 0.5 * K.mean(K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1))
    #kl_loss = - 0.5 * K.mean(1 + tf.reshape(z_log_sigma,[-1]) - K.square(tf.reshape(z_mean,[-1])) - K.exp(tf.reshape(z_log_sigma,[-1])), axis=-1)
    #How to track the two losses in Tensorboard?
    #kl_loss weird !!!!
    return  kl_loss + xent_loss

seq_vae     = Model(seq_x,seq_x_decoded_mean)
seq_vae_enc = Model(seq_x,seq_z_mean)
seq_vae_dec = Model(seq_decoder_input, _seq_x_decoded_mean)

kb_vae      = Model(kb_x,kb_x_decoded_mean)
kb_vae_enc  = Model(kb_x,kb_z_mean)
kb_vae_dec  = Model(kb_decoder_input, _kb_x_decoded_mean)

def example_results(vae_d,ex_n):
    vp2 = vae_d.predict(decX[range(0,100)])
    for i in range(ex_n):
        num1,num2 = [random.randint(0,99),random.randint(0,99)]
        print('\n sample nums: ',num1,' ',num2)
        round_dec1 = [round(x) for x in vp2[num1]]
        round_dec2 = [round(x) for x in vp2[num2]]
        print('num1')
        print([i for i, x in enumerate(X[num1]) if x == True])
        print([i for i, x in enumerate(round_dec1) if x == True])
        print('num2')
        print([i for i, x in enumerate(X[num2]) if x == True])
        print([i for i, x in enumerate(round_dec2) if x == True])
        print('unrounded1')
        print(vp2[num1])
        print('unrounded2')
        print(vp2[num2])

##MAKE NAME A FUNCTION OF THE SELECTION!!
Tensorboard_str = '../TensorBoard/kb_vae_dec/kb_vae_'+str(datetime.datetime.now())+'_'+'eps:'+str(epochs)+'_bs:'+str(batch_size)+'_drop:'+str(dropout_rate)
#Cut of remainder of validation set wrt batch_size
X = X[0:42000]
X_seq = X_seq[0:42000]
decX = decX[0:42000]
print('X_shape: ', X.shape)
print('X', X.shape)
print('decX', decX.shape)

if(not train_anew):
    print('Loading existing model')
    yaml_file = open('../models/experiment2/kb_vae_dec_1.yaml','r')
    kb_vae_dec_yaml = yaml_file.read()
    yaml_file.close()
    kb_vae_dec = model_from_yaml(kb_vae_dec_yaml)
    kb_vae_dec.load_weights('../models/experiment2/kb_vae_dec_1.h5')

print('Training:')
#kb_vae_dec.summary()
#kb_vae_dec.compile(optimizer=keras.optimizers.Adam(),
#    
if(train_type=='kb'):
    kb_vae.summary()
    kb_vae.compile(optimizer=keras.optimizers.Adam(),
            loss=objectives.mean_squared_error)
if(train_type=='seq'):
    seq_vae.summary()
    seq_vae.compile(optimizer=keras.optimizers.Adam(),
                loss=objectives.mean_squared_error)

print(X_seq)

print('File name: eps_',str(epochs)+'_bs_'+str(batch_size)+'_'+str(X.shape[1])+'_words_+'+str(1))
for i in range(0,epochs):
    #Ever X iter. do a run with a loss that prints
    if(i % 10 == 0):
        #Does this saving even work??????
        print("\n Saving current model")
        model_yaml = kb_vae_dec.to_yaml()
        with open("../models/experiment2/kb_vae.yaml","w") as yaml_file:
            yaml_file.write(model_yaml)
        kb_vae_dec.save_weights('../models/experiment2/kb_vae.h5')
        
    print('\n Iterations: ',i)
    if(train_type == 'kb'):
        kb_vae.fit( x=X,y=X,
                shuffle=True,
                epochs=1,
                batch_size=batch_size,
                validation_split = 0.1,
                callbacks=[TensorBoard(log_dir=Tensorboard_str,write_batch_performance=True)]
        )        
    if(train_type == 'seq'):
        seq_vae.fit( x=X_seq,y=X_seq,
                shuffle=True,
                epochs=1,
                batch_size=batch_size,
                validation_split = 0.1,
                callbacks=[TensorBoard(log_dir=Tensorboard_str,write_batch_performance=True)]
        )