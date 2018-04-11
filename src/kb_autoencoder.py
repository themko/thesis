'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
from __future__ import print_function
from keras.models import Sequential,Model,load_model,model_from_json
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
parser.add_argument('batch_sz', type=int,nargs='?', default=100,const=100,
                    help='Size of the batches in training')
parser.add_argument('eps', type=int,nargs='?', default=20,const=20,
                    help='Number of epochs')
parser.add_argument('retrain', type=str,nargs='?', default='retrain',const='retrain',
                    help='Parameter indicating if currently retraining model or just running existing one')
parser.add_argument('preproc', type=str,nargs='?', default='verse',const='verse',
                    help='Type of data preprocessing')
parser.add_argument('perc', type=int,nargs='?', default=100,const=100,
                    help='Percentage of dataset in use')
parser.add_argument('dropout', type=float,nargs='?', default=0.0,const=0.0,
                    help='Percentage of dataset in use')
args = parser.parse_args()
print('Batch size: ',args.batch_sz)
batch_size  = args.batch_sz
epochs      = args.eps
ignore_names= True
train_anew  = False
set_len     = True
dropout_rate = args.dropout

if(args.retrain == 'retrain'):
    train_anew  = True
preproc     = args.preproc
assert (args.perc<= 100)
perc        = args.perc
#Print experiment set-up
print('Batch size:',batch_size, ' epochs:',epochs, 'retrain:',train_anew, ' preprocessing:',preproc, 'Used data:',perc,'%')
"""
path = "../data/English.txt"
fo1 = open(path,"r")

lines_text = fo1.read().splitlines()
lines_text = [a.lower() for a in lines_text]

print('Padding sequences')
maxlen = max([len(b) for b in lines_text])
for i,line in enumerate(lines_text):
    while len(lines_text[i])< maxlen:
        lines_text[i]+=('-')

chars = sorted(list(set(''.join([item for sublist in lines_text for item in sublist]))))
print('chars',chars)
print('total chars:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
"""

#Build the new database
if(preproc == 'kb'):
    predicate_dict = defaultdict(set)
    predicate_dict_test = defaultdict(set)

    df = pd.read_csv('../data/e2e-dataset/trainset.csv',delimiter=',')
    df_test= pd.read_csv('../data/e2e-dataset/testset.csv',delimiter=',')
    tuples = [tuple(x) for x in df.values]
    tuples_test = [tuple(x) for x in df_test.values]
    for t in tuples:
        for r in t[0].split(','):
            r_ind1 = r.index('[')
            r_ind2 = r.index(']')

            rel = r[0:r_ind1].strip()
            rel_val = r[r_ind1+1:r_ind2]
            predicate_dict[rel].add(rel_val)    
    for t in tuples_test:
        for r in t[0].split(','):
            r_ind1 = r.index('[')
            r_ind2 = r.index(']')

            rel = r[0:r_ind1].strip()
            rel_val = r[r_ind1+1:r_ind2]
            predicate_dict_test[rel].add(rel_val)
    print(predicate_dict)
    print
    rel_lens = [len(predicate_dict[p]) for p in predicate_dict.keys()]
    rel_lens_test = [len(predicate_dict_test[p]) for p in predicate_dict_test.keys()]
    print(len(tuples))
    print(rel_lens)
    print(sum(rel_lens))

#Add language labels to vectorization
print('Vectorization...')
if(preproc == 'slice'):
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
elif(preproc=='verse'):
    y = []
    print(maxlen)        
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    print(X.shape)
    for i, sentence in enumerate(sentences):
                for t, char in enumerate(sentence):
                    if(t<maxlen):
                        X[i, t, char_indices[char]] = 1
elif(preproc=='kb'):
    print('kb')   
    X = np.zeros((len(tuples), sum(rel_lens)), dtype=np.bool)
    X_test = np.zeros((len(tuples_test), sum(rel_lens)), dtype=np.bool)
    for i, tup in enumerate(tuples):
        for relation in tup[0].split(',')[1:]:
            rel_name = relation[0:relation.index('[')].strip()
            rel_value= relation[relation.index('[')+1:-1].strip()
            name_ind = predicate_dict.keys().index(rel_name)
            value_ind= list(predicate_dict[rel_name]).index(rel_value)
            j = sum(rel_lens[0:name_ind]) + value_ind
            X[i,j] = 1    
    for i, tup in enumerate(tuples_test):
        for relation in tup[0].split(',')[1:]:
            rel_name = relation[0:relation.index('[')].strip()
            rel_value= relation[relation.index('[')+1:-1].strip()
            name_ind = predicate_dict.keys().index(rel_name)
            value_ind= list(predicate_dict[rel_name]).index(rel_value)
            j = sum(rel_lens[0:name_ind]) + value_ind
            X_test[i,j] = 1
    print('Xtest crated... shape:',X_test.shape)
    print(X_test[0:2])
else:
    print(intentionalError)    

#print(X[0])
#print(tuples[0])
#print(rel_lens)
#print(predicate_dict)

intermediate_dim1 = 128
intermediate_dim2 = 64
latent_dim = 20
epsilon_std = 1.0
print('init_X')
print(rel_lens[0])
print(X[0:2])
if(ignore_names):
    #Xkb_train = np.array([line[rel_lens[0]:]for line in X])
    Xkb_train = np.array([line[rel_lens[0]:-20]for line in X])
    X = Xkb_train
    #X_test = np.array([line[rel_lens[0]:]for line in X_test])
    X_test = np.array([line[rel_lens[0]:-20]for line in X_test])
sen_len = X.shape[1] #Length of sequence
print('Build model...')


## TEST TEST
#sen_len =10

#VAE model based on same input
x = Input(shape=(sen_len,),name='Inputs')
h1= Dense(intermediate_dim1,name='enc_h1',activation='relu')(x)
h1_drop = Dropout(dropout_rate)(h1)
h2= Dense(intermediate_dim2,name='enc_h2',activation='relu')(h1_drop)
h2_drop =Dropout(dropout_rate)(h2)

h3 = Dense(32,name='enc_h3',activation='relu')(h2_drop)
z_mean = Dense(latent_dim,name='z_mean')(h3)
z_log_sigma= Dense(latent_dim,name='z_log_sigma')(h3)


#z_mean = Dense(latent_dim,name='z_mean')(h2_drop)
#z_log_sigma= Dense(latent_dim,name='z_log_sigma')(h2_drop)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = Lambda(sampling,output_shape=(latent_dim,))([z_mean,z_log_sigma])
#decoder_h = Dense(intermediate_dim,name='dec_h',activation='relu')



decoder_h2 = Dense(intermediate_dim1,name='dec_h2',activation='relu')
dec_h2_drop =Dropout(dropout_rate)
decoder_h1 = Dense(intermediate_dim2,name='dec_h1',activation='relu')
dec_h1_drop =Dropout(dropout_rate)
decoder_output = Dense(sen_len,name='dec_out',activation='relu')

dec_h3  = Dense(32,name='dec_h3',activation='relu')#s
h_dec3 = dec_h3(z)#

h_decoded2 = decoder_h2(h_dec3)#z)

h_decoded2_drop = dec_h2_drop(h_decoded2)
h_decoded1 = decoder_h1(h_decoded2_drop)
h_decoded1_drop = dec_h1_drop(h_decoded1)
x_decoded_mean = decoder_output(h_decoded1_drop)


decoder_input = Input(shape=(latent_dim,))
print(decoder_input.shape)

_h_dec3 = dec_h3(decoder_input)#
_h_decoded2 = decoder_h2(_h_dec3)#decoder_input)

_h_decoded2_drop = dec_h2_drop(_h_decoded2)

print(_h_decoded2.shape)

_h_decoded1 = decoder_h1(_h_decoded2_drop)
_h_decoded1_drop = dec_h1_drop(_h_decoded1)
_x_decoded_mean = decoder_output(_h_decoded1_drop)

def vae_loss(x, x_decoded_mean):
    #Waarom maak K.mean toevoegen val_los/acc mogelijk???????
    #Categorical xent raar???objectives

    #xent_loss = K.mean(objectives.binary_crossentropy(x,x_decoded_mean))
    #xent_loss = objectives.mean_squared_error(x,x_decoded_mean)
    #orig 
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    #orig 
    #kl_loss = - 0.5 * K.mean(K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1))
    kl_loss = - 0.5 * K.mean(1 + tf.reshape(z_log_sigma,[-1]) - K.square(tf.reshape(z_mean,[-1])) - K.exp(tf.reshape(z_log_sigma,[-1])), axis=-1)

    #kl_loss = K.print_tensor(kl_loss, message="kl_loss is: ")
    #xent_loss = K.print_tensor(xent_loss, message="xent_loss is: ")
    return  kl_loss + xent_loss


vae = Model(x,x_decoded_mean)
vae_enc = Model(x,z_mean)
vae_dec = Model(decoder_input, _x_decoded_mean)

def example_results(vae_e,vae_d,ex_n):
    vp1 = vae_e.predict(X_test[range(0,100)])
    vp2 = vae_d.predict(vp1)
    for i in range(ex_n):
        num1,num2 = [random.randint(0,99),random.randint(0,99)]
        print('\n nums: ',num1,' ',num2)
        vp_mean = ( np.array(vp1[num1])+np.array(vp1[num2]) )/2
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

if(train_anew):
    print('Training:')

#Train models
    vae.summary()
    vae.compile(optimizer=keras.optimizers.Adam(lr= 10**-6),
                loss=vae_loss)

    print('Fitting model with debugs inbetween')
    Tensorboard_str = '../TensorBoard/kb_vae_'+str(datetime.datetime.now())+'_'+'eps:'+str(epochs)+'_bs:'+str(batch_size)+'_drop:'+str(dropout_rate)

    #Cut of remainder of validation set wrt batch_size
    #remainder = X.shape[0]% latent_dim
    #remainder_test = X_test.shape[0]%batch_size
    #if(remainder!= 0):
        #X  = X[0:-remainder]
    X = X[0:42000]
    #if(remainder_test != 0):
    #    X_test = X_test[0:-remainder_test]
    print('X_shape: ',X.shape)
    for i in range(0,epochs):
        print('\n Iterations: ',i+1)
        vae.fit( X,X,
                shuffle=True,
                epochs=1,
                batch_size=batch_size,
                validation_split = 0.1,
                callbacks=[TensorBoard(log_dir=Tensorboard_str,write_batch_performance=True)]
        )
        #Ever X iter. do a run with a loss that prints
        #
        if(i %10 ==0):
            example_results(vae_enc,vae_dec,2)
        #    print(vae.get_weights())
        #print('\n final LSTM layer max weight', (np.max(decoder_mean.get_weights())) ) 
    #vae.save_weights('my_vae'+'_bs:'+str(batch_size)+'_epochs:'+str(epochs)+'_'+str(datetime.datetime.now())+'.h5')  # creates a HDF5 file 'my_model.h5'
    vae.save_weights('../models/vae_lstm_weights_eps_'+str(epochs)+'_bs_'+str(batch_size)+'_'+str(maxlen_len)+'_words_+'+str(datetime.datetime.now())+'.h5')  # creates a HDF5 file 'my_model.h5'
else:
    #CHANGE LOADOUT!
    #vae = load_model('my_vae_bs:100_epochs:11_current.h5', custom_objects={'batch_size':batch_size,'latent_dim': latent_dim, 'epsilon_std': epsilon_std, 'vae_loss': vae_loss})
    #vae = load_model('models/dfa.h5', custom_objects={'batch_size':batch_size,'latent_dim': latent_dim, 'epsilon_std': epsilon_std, 'vae_loss': vae_loss})
    vae.load_weights('../models/dfa_weights.h5')
    vae.summary()
    vae.compile(optimizer='rmsprop',loss=vae_loss,metrics=['accuracy'])

    print('Starting interpolation')
    #How to load partial models?
    vae_enc = Model(x,z_mean)
    vae_dec = Model(decoder_input, _x_decoded_mean)
    vae_enc.compile(optimizer='rmsprop',loss=vae_loss)
    vae_dec.compile(optimizer='rmsprop',loss=vae_loss)

print('Evaluated...')
vae_weights = vae.get_weights()

#Do some weight debugging
print('Weight prints:')
wes_max = []
wes_min = []
for we_list in vae_weights:
    for we in we_list:
        wes_max.append(np.max(we))
        wes_min.append(np.min(we))
