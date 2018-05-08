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
from keras.layers import Dense, Activation,Input,RepeatVector,Dropout,Concatenate,concatenate
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
parser.add_argument('eps', type=int,nargs='?', default=200,const=200,
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

###MAKE A WORD-BASED ENCODEING VAE DATABASE!!!!!!

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

intermediate_dim1   = 8
intermediate_dim2   = 64
intermediate_dim3   = 128
latent_dim          = 5
epsilon_std         = 1.0

#Sampling from N-dimm gaussian, for debugging decoder
decX = np.zeros((len(tuples),latent_dim))
for i in range(decX.shape[0]):
    for j in range(latent_dim):
        decX[i][j] = random.gauss(0,1)

#Decide how many of features of kb dataset to ignore
#Currently leaves sen_len = 10
kb_strip=True
if(kb_strip):
    data_offset = sum(rel_lens[0:5])
    X = np.array([a[data_offset:] for a in X])
kb_sen_len = X.shape[1] #Length of sequence

sen_len         = X_seq.shape[1]
voc_len         = X_seq.shape[2]

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon



"""
SEQ VAE:
"""


seq_x           = Input(shape=((sen_len,voc_len)),name='Seq_inputs')
seq_h           = LSTM(intermediate_dim1, name='seq_h_layer_enc')(seq_x)
seq_z_mean      = Dense(latent_dim,name='seq_z_mean')(seq_h)
seq_z_log_sigma = Dense(latent_dim,name='seq_z_log_sigma')(seq_h)

seq_z                   = Lambda(sampling,output_shape=(latent_dim,))([seq_z_mean,seq_z_log_sigma])
"""
Path to seq:
"""
seq_decoder_h           = LSTM(intermediate_dim1,name='seq_dec_h',return_sequences=True)
seq_decoder_mean        = Dense(voc_len,name='seq_dec_x_mean',activation='softmax')

seq_repeat_z            = RepeatVector(sen_len,name='repeat_z')(seq_z)
seq_h_decoded           = seq_decoder_h(seq_repeat_z)
seq_x_decoded_mean      = seq_decoder_mean(seq_h_decoded)

##ALTER LATER FOR SEQ
seq_decoder_input       = Input(shape=(latent_dim,))
_seq_repeat_z           = RepeatVector(sen_len)(seq_decoder_input)
_seq_h_decoded          = seq_decoder_h(_seq_repeat_z)
_seq_x_decoded_mean     = seq_decoder_mean(_seq_h_decoded)
"""
Path to kb:
"""
kb_dec_h1           = Dense(intermediate_dim1,name='kb_dec_h1',activation='relu')
kb_decoder_output   = Dense(kb_sen_len,name='kb_dec_out',activation='relu')

kb_h_decoded1_seq      = kb_dec_h1(seq_z)
kb_x_decoded_mean_seq  = kb_decoder_output(kb_h_decoded1_seq)
kb_decoder_input_seq    = Input(shape=(latent_dim,))
_kb_h_decoded1_seq      = kb_dec_h1(kb_decoder_input_seq)
_kb_x_decoded_mean_seq  = kb_decoder_output(_kb_h_decoded1_seq)

"""
KB VAE: 
"""
kb_h_decoded1       = kb_dec_h1(seq_z)
kb_x_decoded_mean   = kb_decoder_output(kb_h_decoded1)
kb_decoder_input    = Input(shape=(latent_dim,))
_kb_h_decoded1      = kb_dec_h1(kb_decoder_input)
_kb_x_decoded_mean  = kb_decoder_output(_kb_h_decoded1)


print('Build model...')
#VAE model based on same input
kb_x        = Input(shape=(kb_sen_len,),name='Kb_Input')
kb_enc1     = Dense(intermediate_dim1,name='kb_enc1',activation='relu')(kb_x) ##
kb_z_mean   = Dense(latent_dim,name='kb_z_mean',activation='relu')(kb_enc1)
kb_z_log_sigma= Dense(latent_dim,name='kb_z_log_sigma',activation='sigmoid')(kb_enc1)
kb_z                = Lambda(sampling,output_shape=(latent_dim,))([kb_z_mean,kb_z_log_sigma])
"""
Path to kb output

kb_dec_h1           = Dense(intermediate_dim1,name='kb_dec_h1',activation='sigmoid')
kb_decoder_output   = Dense(kb_sen_len,name='kb_dec_out',activation='sigmoid')
"""
kb_h_decoded1       = kb_dec_h1(kb_z)
kb_x_decoded_mean   = kb_decoder_output(kb_h_decoded1)
kb_decoder_input    = Input(shape=(latent_dim,))
_kb_h_decoded1      = kb_dec_h1(kb_decoder_input)
_kb_x_decoded_mean  = kb_decoder_output(_kb_h_decoded1)
"""
Path to seq output
"""
seq_repeat_z_kb            = RepeatVector(sen_len,name='seq_repeat_z')(kb_z)
seq_h_decoded_kb          = seq_decoder_h(seq_repeat_z_kb)
seq_x_decoded_mean_kb     = seq_decoder_mean(seq_h_decoded_kb)
seq_decoder_input_kb      = Input(shape=(latent_dim,))
_seq_repeat_z_kb           = RepeatVector(sen_len)(seq_decoder_input_kb)
_seq_h_decoded_kb          = seq_decoder_h(_seq_repeat_z_kb)
_seq_x_decoded_mean_kb     = seq_decoder_mean(_seq_h_decoded_kb)

#Combi_VAE
if(False):
    """
    COMBI VAE below:
    """
    combi_kb_enc1 = Dense(intermediate_dim1,name='combi_kb_enc1',activation='sigmoid')(kb_x)
    combi_seq_enc1= LSTM(intermediate_dim1, name='combi_seq_enc1')(seq_x)
    merge_one = concatenate([combi_kb_enc1, combi_seq_enc1])

    #Or do we want splitted means and sigma's?
    #Another experiment variable!!
    #
    combi_z_mean = Dense(latent_dim,name='combi_z_mean')(merge_one) #concat inputs
    combi_z_log_sigma= Dense(latent_dim,name='combi_z_log_sigma')(merge_one)#concat inputs

    combi_z = Lambda(sampling,output_shape=(latent_dim,))([combi_z_mean,combi_z_log_sigma])

    combi_kb_dec_h1           = Dense(intermediate_dim1,name='dec_h1',activation='sigmoid')
    combi_kb_decoder_output   = Dense(kb_sen_len,name='kb_dec_out',activation='sigmoid')
    #Add layers later
    #
    combi_xkb_decoded1       = kb_dec_h1(combi_z)#combi_kb_dec_h1(combi_z)
    combi_xkb_decoded_mean   = kb_decoder_output(combi_xkb_decoded1) #combi_kb_decoder_output(combi_xkb_decoded1)

    combi_decoder_input      = Input(shape=(latent_dim,))
    _combi_xkb_decoded1      = kb_dec_h1(combi_decoder_input)#combi_kb_dec_h1(combi_decoder_input)
    _combi_xkb_decoded_mean  = kb_decoder_output(_combi_xkb_decoded1)#combi_kb_decoder_output(_combi_xkb_decoded1)

    """
    seq_z                   = Lambda(sampling,output_shape=(latent_dim,))([seq_z_mean,seq_z_log_sigma])
    seq_repeat_z            = RepeatVector(sen_len,name='repeat_z')(seq_z)
    seq_h_decoded           = seq_decoder_h(seq_repeat_z)
    seq_x_decoded_mean      = seq_decoder_mean(seq_h_decoded)
    """
    ##ALTER LATER FOR SEQ
    combi_seq_repeat_z      = RepeatVector(sen_len,name='combi_repeat_z')(combi_z)
    combi_seq_h_decoded     = seq_decoder_h(combi_seq_repeat_z)
    combi_xseq_decoded_mean = seq_decoder_mean(combi_seq_h_decoded)

    _combi_seq_repeat_z         = RepeatVector(sen_len)(combi_decoder_input)
    _combi_seq_h_decoded        = seq_decoder_h(_combi_seq_repeat_z)
    _combi_xseq_decoded_mean     = seq_decoder_mean(_combi_seq_h_decoded)
#how to redifine based on unknown true values?
def vae_loss_kb(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    #kl_loss = - 0.5 * K.mean(1 + tf.reshape(kb_z_log_sigma,[-1]) - K.square(tf.reshape(kb_z_mean,[-1])) - K.exp(tf.reshape(kb_z_log_sigma,[-1])), axis=-1)
    #fcollet autoencoder loss: https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
    #kl_loss = - 0.5 * K.mean(K.sum(1 + kb_z_log_sigma - K.square(kb_z_mean) - K.exp(kb_z_log_sigma), axis=-1))
    #kl_loss = - 0.5 * K.sum(1 + kb_z_log_sigma - K.square(kb_z_mean) - K.exp(kb_z_log_sigma), axis=-1)
    kl_loss = - 0.5 * K.mean(1 + kb_z_log_sigma - K.square(kb_z_mean) - K.exp(kb_z_log_sigma), axis=-1)
    return xent_loss + kl_loss

def vae_loss_kb_kl(x, x_decoded_mean):
    kl_loss = - 0.5 * K.mean(K.sum(1 + kb_z_log_sigma - K.square(kb_z_mean) - K.exp(kb_z_log_sigma), axis=-1))
    return  kl_loss

def vae_loss_kb_xent(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    return  xent_loss

def vae_loss_seq(x, x_decoded_mean):
    #orig xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    xent_loss = K.mean(objectives.categorical_crossentropy(x, x_decoded_mean))
    #ADD THE OTHER K.mean??????
    kl_loss = - 0.5 * K.mean(1 + seq_z_log_sigma - K.square(seq_z_mean) - K.exp(seq_z_log_sigma), axis=-1)    #Extra K.mean added to use with sequential input shape
    #orig kl_loss = - 0.5 * K.mean(K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1))
    #kl_loss = - 0.5 * K.mean(1 + tf.reshape(z_log_sigma,[-1]) - K.square(tf.reshape(z_mean,[-1])) - K.exp(tf.reshape(z_log_sigma,[-1])), axis=-1)
    #How to track the two losses in Tensorboard?
    #kl_loss weird !!!!
    return  kl_loss + xent_loss

def vae_loss_combi(x, x_decoded_mean):
    ###SORT THIS LOSS OUT STILL!
    #
    #
    xent_loss =objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + tf.reshape(kb_z_log_sigma,[-1]) - K.square(tf.reshape(kb_z_mean,[-1])) - K.exp(tf.reshape(kb_z_log_sigma,[-1])), axis=-1)
    return  kl_loss + xent_loss

seq_vae     = Model(seq_x,seq_x_decoded_mean)
seq_vae_enc = Model(seq_x,seq_z_mean)
seq_vae_dec = Model(seq_decoder_input, _seq_x_decoded_mean)

kb_vae      = Model(kb_x ,kb_x_decoded_mean)
#Models with idential  structure to determine the individual loss values
kb_vae_kl   = Model(kb_x,kb_x_decoded_mean)
kb_vae_xent = Model(kb_x,kb_x_decoded_mean)

kb_vae_enc  = Model(kb_x,kb_z_mean)
kb_vae_dec  = Model(kb_decoder_input, _kb_x_decoded_mean)

""" 
combi_vae     = Model(inputs=[kb_x,seq_x],outputs=[combi_xkb_decoded_mean,combi_xseq_decoded_mean])
combi_vae_enc = Model(inputs=[kb_x,seq_x],outputs=combi_z_mean)
combi_vae_dec = Model(inputs=combi_decoder_input, outputs=[_combi_xkb_decoded_mean,_combi_xseq_decoded_mean])
"""
ks_vae = Model(kb_x,seq_x_decoded_mean_kb)
sk_vae = Model(seq_x,kb_x_decoded_mean_seq)

print('Test summary')
sk_vae.summary()
def example_results(vae_x,vae_e,vae_d,ex_n):
    x_encoded = vae_e.predict(vae_x)
    x_decoded = vae_d.predict(x_encoded)
    print('x:\n')
    for a in vae_x[20:23]:
        print([int(b) for b in a])
    #print([[int(x) for x in y] for y in vae_x[20:23]])

    #print('enc:\n',x_encoded[20:23])
    print('dec_round:\n')
    for a in x_decoded[20:23]:
        print([round(b) for b in a])
    print('dec:\n',x_decoded[20:23])

Tensorboard_str = '../TensorBoard/vae_'+str(datetime.datetime.now())+'type_'+str()+'_eps:'+ str(epochs)+'_bs:'+str(batch_size)+'_drop:'+str(dropout_rate)
#Cut of remainder of validation set wrt batch_size
X_kb = X[0:42000]
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
if(train_type=='kb'):
    kb_vae.summary()
    kb_vae.compile(optimizer=keras.optimizers.Adagrad(),
                loss=vae_loss_kb)

    kb_vae_kl.compile(optimizer=keras.optimizers.Adam(),
            loss=vae_loss_kb_kl)
    kb_vae_xent.compile(optimizer=keras.optimizers.Adam(),
            loss=vae_loss_kb_xent)

if(train_type=='seq'):
    seq_vae.summary()
    seq_vae.compile(optimizer=keras.optimizers.Adam(),
                loss=vae_loss_seq)
if(train_type=='combi'):
    combi_vae.summary()
    combi_vae.compile(optimizer=keras.optimizers.Adam(), ###CHANGE THE LOSS!!!!!!!!!!
            loss='mean_squared_error')

#print(X_kb[0:10])
sess = tf.InteractiveSession()
bin1 = tf.constant([float(a) for b in X_kb[0:10] for a in b])
bin2 = tf.constant([float(a) for b in X_kb[20:30] for a in b])
#print(bin1.eval())
print('test1',objectives.binary_crossentropy(bin1, bin2).eval())
kb_z_log_sigma = tf.constant([[1.,2.,3.,4.],[11.,12.,13.,14.]])
kb_z_mean = tf.constant([[11.,12.,13.,14.],[21.,22.,23.,24.]])
print('test2',K.mean(- 0.5 * K.sum(1 + kb_z_log_sigma - K.square(kb_z_mean) - K.exp(kb_z_log_sigma), axis=-1)).eval())

print('VAE-kl being constructed')
#Single x,w encoders
#w_enc
jvae_w0      = Input(shape=((sen_len,voc_len)),name='JVAE_input_w0')
jvae_w0_h    = LSTM(intermediate_dim1,name='JVAE_w0_h')(jvae_w0)
jvae_w0_z_mean       = Dense(latent_dim,name='jvae_w0_z_mean')(jvae_w0_h)
jvae_w0_z_log_sigma  = Dense(latent_dim,name='jvaea_w0_z_log_sigma')(jvae_w0_h)
#x_enc
jvae_x0      = Input(shape=(kb_sen_len,),name='JVAE_input_x0')
jvae_x0_h    = Dense(intermediate_dim1,name='JVAE_layer1_x0',activation='relu')(jvae_x0) ##
jvae_x0_z_mean       = Dense(latent_dim,name='jvae_x0_z_mean')(jvae_x0_h)
jvae_x0_z_log_sigma  = Dense(latent_dim,name='jvaea_x0_z_log_sigma')(jvae_x0_h)

#Joint x,w encoders
jvae_w      = Input(shape=((sen_len,voc_len)),name='JVAE_input_w')
jvae_w_h    = LSTM(intermediate_dim1,name='JVAE_w_h')(jvae_w)

jvae_x      = Input(shape=(kb_sen_len,),name='JVAE_input_x')
jvae_x_1    = Dense(intermediate_dim1,name='JVAE_layer1_x',activation='relu')(jvae_x) ##

#Take sample from last joint encoder layer
jvae_xw_enc          = concatenate([jvae_w_h, jvae_x_1],axis=1)
jvae_xw_z_mean       = Dense(latent_dim,name='jvae_xw_z_mean')(jvae_xw_enc)
jvae_xw_z_log_sigma  = Dense(latent_dim,name='jvaea_xw_z_log_sigma')(jvae_xw_enc)
jvae_xw_z            = Lambda(sampling,output_shape=(latent_dim,))([jvae_xw_z_mean,jvae_xw_z_log_sigma])

#Reusable decoder functions
jvae_w_decoder_h     = LSTM(intermediate_dim1,name='jvae_w_decoder_h',return_sequences=True)
jvae_w_decoder_mean  = Dense(voc_len,name='jvae_w_decoder_mean',activation='softmax')
jvae_x_decoder_h     = Dense(intermediate_dim1,name='jvae_x_decoder_h',activation='relu')
jvae_x_decoder_output= Dense(kb_sen_len,name='jvae_x_decoder_output',activation='relu')

#Joint x decoder
jvae_x_decoded1       = jvae_x_decoder_h(jvae_xw_z)
jvae_x_decoded_mean   = jvae_x_decoder_output(jvae_x_decoded1)

jvae_x_decoder_input  = Input(shape=(latent_dim,))
_jvae_x_decoded       = jvae_x_decoder_h(jvae_x_decoder_input)
_jvae_x_decoded_mean  = jvae_x_decoder_output(_jvae_x_decoded)

#Joint w decoder
jvae_w_repeat_z         = RepeatVector(sen_len,name='repeat_z')(jvae_xw_z)
jvae_w_decoded          = jvae_w_decoder_h(jvae_w_repeat_z)
jvae_w_decoded_mean     = jvae_w_decoder_mean(jvae_w_decoded)
#
jvae_w_decoder_input     = Input(shape=(latent_dim,))
_jvae_w_repeat_z         = RepeatVector(sen_len)(jvae_w_decoder_input)
_jvae_w_decoded          = jvae_w_decoder_h(_jvae_w_repeat_z)
_jvae_w_decoded_mean     = jvae_w_decoder_mean(_jvae_w_decoded)


#DOES NOT WORK!!!
def jvae_kl_loss(x,x_decoded_mean):
    #[jw,jx, jw0,jx0] = x
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    #Gone to double K.mean

    kl_loss = - 0.5 * K.mean(K.mean(1 + jvae_xw_z_log_sigma - K.square(jvae_xw_z_mean) - K.exp(jvae_xw_z_log_sigma), axis=-1))
    #Correct???
    kl_loss_x_xw = - 0.5 * K.mean(K.mean( 1 + jvae_xw_z_log_sigma - jvae_x0_z_log_sigma - K.square(jvae_xw_z_mean - jvae_x0_z_mean) -K.exp(jvae_x0_z_mean),axis=-1)) #??
    kl_loss_w_xw = - 0.5 * K.mean(K.mean( 1 + jvae_xw_z_log_sigma - jvae_w0_z_log_sigma - K.square(jvae_xw_z_mean - jvae_w0_z_mean) -K.exp(jvae_w0_z_mean),axis=-1)) #??
    return xent_loss + kl_loss #- (kl_loss_w_xw + kl_loss_x_xw)

jvae_model = Model(input=[jvae_w,jvae_x,jvae_w0,jvae_x0],#output=[jvae_x_decoded_mean,jvae_w_decoded_mean,
        #jvae_w0_z_mean,jvae_w0_z_log_sigma, jvae_x0_z_mean,jvae_x0_z_log_sigma
        output = [jvae_w_decoded_mean,jvae_x_decoded_mean])

jvae_model.summary()
jvae_model.compile(optimizer=keras.optimizers.Adagrad(),
                loss=jvae_kl_loss)
#

#inputs=[jvae_w,jvae_x,jvae_x0,jvae_w0]
#outputs= []
jvae_model.fit(x = [X_seq,X_kb,X_seq,X_kb], y= [X_seq,X_kb], batch_size=batch_size, validation_split=0.1 )
print(intentionalError)

print('File name: eps_',str(epochs)+'_bs_'+str(batch_size)+'_words_'+str(X.shape[1]))
for i in range(0,epochs):
    #Ever X iter. do a run with a loss that prints
    print('\n Iterations: ',i)
    if(train_type == 'kb'):
        kb_vae.fit( x=X_kb,y=X_kb,
                shuffle=True,
                epochs=1,
                batch_size=batch_size,
                validation_split = 0.1,
                callbacks=[TensorBoard(log_dir=Tensorboard_str,write_batch_performance=True)]
        )     
        if(i % 100 == 0):
            #FIX SAVING AND LOADING FOR SPECIF NAMES AND RELOADING BUG
            print("\n Saving current model")
            model_yaml = kb_vae.to_yaml()
            with open("../models/"+str(train_type)+"_vae.yaml","w") as yaml_file:
                yaml_file.write(model_yaml)
            kb_vae.save_weights('../models/'+str(train_type)+'_vae.h5')
            example_results(X_kb,kb_vae_enc,kb_vae_dec,5)

            print('kb kl_loss',kb_vae_kl.evaluate(x =X_kb[0:500],y=X_kb[0:500],batch_size=batch_size,verbose=0))
            print('kb xent_loss',kb_vae_xent.evaluate(x =X_kb[0:500],y=X_kb[0:500],batch_size=batch_size,verbose=0))

    if(train_type == 'seq'):
        seq_vae.fit( x=X_seq,y=X_seq,
                shuffle=True,
                epochs=1,
                batch_size=batch_size,
                validation_split = 0.1,
                callbacks=[TensorBoard(log_dir=Tensorboard_str,write_batch_performance=True)]
        )    
    if(train_type == 'combi'):
        combi_vae.fit( x=[X_kb,X_seq],y=[X_kb,X_seq],
                shuffle=True,
                epochs=1,
                batch_size=batch_size,
                validation_split = 0.1,
                callbacks=[TensorBoard(log_dir=Tensorboard_str,write_batch_performance=True)]
        )
        model_yaml = combi_vae.to_yaml()
        with open("../models/"+str(train_type)+"_vae.yaml","w") as yaml_file:
            yaml_file.write(model_yaml)
        combi_vae.save_weights('../models/'+str(train_type)+'_vae.h5')

"""
#Visualize latent space
n = 15  # figure with 15x15 digits
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]]) * epsilon_std
        x_decoded = kb_vae_dec.predict(z_sample)
        #digit = x_decoded[0].reshape(digit_size, digit_size)
        print(yi,xi,':\n')
        print(x_decoded)
"""
