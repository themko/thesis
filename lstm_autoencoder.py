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
from keras.layers import Dense, Activation,Input,RepeatVector
from keras.layers import LSTM, Lambda
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras import backend as K
from keras import objectives
from keras.callbacks import TensorBoard
from keras.callbacks import Callback as Callback
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import regularizers
import keras
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

args = parser.parse_args()
print('Batch size: ',args.batch_sz)
batch_size  = args.batch_sz
epochs      = args.eps

train_anew  = False

set_len     = True
maxlen_len  = 5

if(args.retrain == 'retrain'):
    train_anew  = True
preproc     = args.preproc
assert (args.perc<= 100)
perc        = args.perc
#Print experiment set-up
print('Batch size:',batch_size, ' epochs:',epochs, 'retrain:',train_anew, ' preprocessing:',preproc, 'Used data:',perc,'%')
# train_anew = bool(train_anew)


#path = get_file('French.txt', origin='file:/home/thomas/Documents/Studie/Thesis/data/French.txt')
#path2 = get_file('English.txt', origin='file:/home/thomas/Documents/Studie/Thesis/data/English.txt')

#RECONSIDER?? PERHAPSE CASE SENTSITIVITY HELPS MODEL!
#path = "data/dfa.dat"
path = "data/English.txt"

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

#Build the new bilingual database

if(preproc == 'slice'):
    # cut the text in semi-redundant sequences of maxlen character
    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))
    sentences1 = []
    next_chars1 = []
    for i in range(0, len(text1) - maxlen, step):
        sentences1.append(text1[i: i + maxlen])
        next_chars1.append(text1[i + maxlen])
    print('nb sequences from corpus1:', len(sentences1)) #1409719
    #
elif(preproc == 'verse'):
    print('lt',len(lines_text))
    if(set_len == False):
        maxlen = max([len(b) for b in lines_text])
    else:
        maxlen = maxlen_len
    print('maxline',maxlen)
    sentences = lines_text

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
else:
    print(intentionalError)    

intermediate_dim = 64
latent_dim = 32
epsilon_std = 1.0


sen_len = X.shape[1] #Length of sequence
voc_len = X.shape[2] #Dimensionality of sequence
print('Build model...')

"""
inputs  = Input(shape=(sen_len,voc_len),name='inputs')
encoded = LSTM(128,name='lstm_encoder')(inputs)
decoded = RepeatVector(sen_len,name='repeat_decoder')(encoded)
decoded = LSTM(voc_len,name='lstm_decoder',return_sequences=True)(decoded)

autoencoder = Model(inputs,decoded)
encoder     = Model(inputs,encoded)
"""

#VAE model based on same input
x = Input(shape=(sen_len,voc_len))
#h = LSTM(intermediate_dim, name='h_layer',activation='relu',kernel_regularizer=regularizers.l1(0.05))(x)
h = LSTM(intermediate_dim, name='h_layer',kernel_regularizer=regularizers.l1(0.03))(x)
z_mean = Dense(latent_dim,name='z_mean')(h)
z_log_sigma= Dense(latent_dim,name='z_log_sigma')(h)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = Lambda(sampling,output_shape=(latent_dim,))([z_mean,z_log_sigma])
repeat_z = RepeatVector(sen_len,name='repeat_z')(z)
#decoder_h = Dense(intermediate_dim,name='dec_h',activation='relu')
decoder_h = Dense(intermediate_dim,name='dec_h')
decoder_mean = LSTM(voc_len,name='dec_mean',return_sequences=True,activation='softmax',kernel_regularizer=regularizers.l1(0.03))
h_decoded = decoder_h(repeat_z)
x_decoded_mean = decoder_mean(h_decoded)

decoder_input = Input(shape=(latent_dim,))
_repeat_z = RepeatVector(sen_len)(decoder_input)
_h_decoded = decoder_h(_repeat_z)
_x_decoded_mean = decoder_mean(_h_decoded)

def vae_loss(x, x_decoded_mean):
    #orig xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    xent_loss = K.mean(objectives.categorical_crossentropy(x, x_decoded_mean))
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)    #Extra K.mean added to use with sequential input shape
    #orig kl_loss = - 0.5 * K.mean(K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1))
    #kl_loss = - 0.5 * K.mean(1 + tf.reshape(z_log_sigma,[-1]) - K.square(tf.reshape(z_mean,[-1])) - K.exp(tf.reshape(z_log_sigma,[-1])), axis=-1)
    
    #How to track the two losses in Tensorboard?

    #kl_loss weird !!!!
    return  kl_loss + xent_loss


def translate_back(predictions):
    preds_num = [p.tolist().index(max(p)) for p in predictions]
    pred_trans = [indices_char[pn] for pn in preds_num]
    return pred_trans

vae = Model(x,x_decoded_mean)
vae_enc = Model(x,z_mean)
vae_dec = Model(decoder_input, _x_decoded_mean)

if(train_anew):
    print('Training:')



#Train models
    vae.summary()
    vae.compile(optimizer='rmsprop',loss=vae_loss,metrics=['accuracy'])
    train_percentage    = 0.9
    upper_train_bound   = int(np.floor(train_percentage*len(X)*perc/100))
    upper_val_bound     = int(np.floor(len(X)*perc/100))
    #Cut down ranges to avoid sampling size mismatch in last batch of epoch
    upper_train_bound   = upper_train_bound - (upper_train_bound % batch_size)
    upper_val_bound     = upper_val_bound - (upper_val_bound %batch_size)
    print('BOUNDS',upper_train_bound,upper_val_bound)
    train_range = range(0,upper_train_bound)
    val_range   = range(upper_train_bound,upper_val_bound)

    #Set up training size to be exact multitude of batch_size
    print('Data cropped: ',upper_train_bound % batch_size)
    print('X.shape',X.shape)
    """
    vae.fit(X,X,#X[train_range,:,:],X[train_range,:,:],
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data =(X[val_range,:,:],X[val_range,:,:]),
            callbacks=[TensorBoard(log_dir='TensorBoard/vae')]
            )
    """
    print('Fitting model with debugs inbetween')
    Tensorboard_str = 'TensorBoard/vae1+_'+str(datetime.datetime.now())

    for i in range(0,1):
        print('\n Iterations: ',i+1)
        vae.fit( x = X[0:upper_train_bound,:,:],y=X[0:upper_train_bound,:,:],#X[train_range,:,:],y=X[train_range,:,:],
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data =(X[val_range,:,:],X[val_range,:,:]),
        callbacks=[TensorBoard(log_dir=Tensorboard_str,write_batch_performance=True)]
        )
        #print(vae.evaluate(   x = X[0:upper_train_bound,:,:],y=X[0:upper_train_bound,:,:],
        #                batch_size=batch_size)
        #)
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
        #print(wes)
        print('minmax',min(wes_min),max(wes_max))


        #print('max_weights: ', ([np.max(x) for y in vae_weights for x in y]))#[np.max(var) for var in vae_weights])
        print('max_weight: ',max([np.max(var) for var in vae_weights]))
        print('min_weights: ',[np.min(var) for var in vae_weights])
        print('min_abs_weights: ',[np.min(abs(var)) for var in vae_weights])
        print('min_weight: ',min([np.min(var) for var in vae_weights]), min([np.min(abs(var)) for var in vae_weights]))
        #if(i %10 ==0):
        #    print(vae.get_weights())
        #print('\n final LSTM layer max weight', (np.max(decoder_mean.get_weights())) ) 
    #vae.save_weights('my_vae'+'_bs:'+str(batch_size)+'_epochs:'+str(epochs)+'_'+str(datetime.datetime.now())+'.h5')  # creates a HDF5 file 'my_model.h5'
    vae.save_weights('../models/weights_'+str(datetime.datetime.now()+'.h5')  # creates a HDF5 file 'my_model.h5'
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

#PREDICT NOT RETURNING CORRECT NUMBER OF DIMENSIONS
vp1 = vae_enc.predict(X[range(0,100),:,:])
vp2 = vae_dec.predict(vp1)

print('prediction enc')
for i in range(2):
    num1,num2 = [random.randint(0,100),random.randint(0,100)]
    print('\n nums: ',num1,' ',num2)
    vp_mean = ( np.array(vp1[num1])+np.array(vp1[num2]) )/2
    print('prediction dec')
    print('orig 0 & 1:')
    print(translate_back(X[num1]))
    print(translate_back(X[num2]))
    print('dec')
    print(translate_back(list(vp2[num1])))
    print(translate_back(list(vp2[num2])))
    print('interpolation orig decoded')

    vp_means = np.vstack((vp_mean,vp_mean,vp_mean,vp_mean,vp_mean,vp_mean,vp_mean,vp_mean,vp_mean,vp_mean))
    print('vp_means',vp_means.dtype,vp_means.shape)
    vp_means_dec = vae_dec.predict(vp_means)
    print(translate_back(list(vp_means_dec[0])))