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
from keras.datasets import imdb
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelBinarizer
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
train_anew = False
if(args.retrain == 'retrain'):
    train_anew  = True
preproc     = args.preproc
assert (args.perc<= 100)
perc        = args.perc
#Print experiment set-up
print('Batch size:',batch_size, ' epochs:',epochs, 'retrain:',train_anew, ' preprocessing:',preproc, 'Used data:',perc,'%')
# train_anew = bool(train_anew)

#MAKE THIS AN INPUT STATEMENT??
bilingual = False

path = get_file('French.txt', origin='file:/home/thomas/Documents/Studie/Thesis/data/French.txt')
path2 = get_file('English.txt', origin='file:/home/thomas/Documents/Studie/Thesis/data/English.txt')


#RECONSIDER?? PERHAPSE CASE SENTSITIVITY HELPS MODEL!
text1 = open(path).read().lower()
text2 = open(path2).read().lower()
if(bilingual):
    text  = text + text2
else:
    text =text1
#Get starting indices
fo1 = open("data/French.txt","r")
fo2 = open("data/English.txt","r")

lines_text1 = fo1.readlines()
lines_text1 = [a.lower() for a in lines_text1]
lines_text2 = fo2.readlines()
lines_text2 = [a.lower() for a in lines_text2]

if(bilingual):
    lines_text = lines_text1 + lines_text2
    #Keep track of indices of the start of each verse
    lines_index = np.array([len(x) for x in lines_text])
    lines_index = list(np.append(np.array(0),lines_index.cumsum()))
    labels = np.concatenate((np.zeros(len(lines_text1)),np.ones(len(lines_text2))),axis=0)
    lines_labels = zip(labels,lines_text)
else:
    lines_text = lines_text1
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

#Build the new bilingual database
maxlen_line = max(max([len(x) for x in lines_text1]), max([len(x) for x in lines_text2]))

if(preproc == 'slice'):
    # cut the text in semi-redundant sequences of maxlen characters
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
    maxlen = max([len(b) for b in lines_text])
    print('maxline',maxlen)
    sentences = lines_text

#Check how many of those are from the first corpus


#Add language labels to vectorization
print('Vectorization...')
if(preproc == 'slice'):
    if(bilingual):
        X = np.zeros((len(sentences), maxlen+1, len(chars)+2), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)+2), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t+1, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1
    else:
        X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1
elif(preproc=='verse'):
    y = []        
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
                for t, char in enumerate(sentence):
                    X[i, t, char_indices[char]] = 1

else:
    print(intentionalError)    
#X1 = np.concatenate((np.zeros((len(sentences),1,len(chars)+2),dtype=bool),X),axis=1)  
#
#print('Shape of X, y', X.shape) #(2789223, 40, 67)
#Set first words of the sentences to language tokens
if(bilingual):
    for i,x_emp in enumerate(X):
        if i==0:
            print(x_emp)
        if(i< len(sentences1)):
            X[i,0,-1] = 1
        else:
            X[i,0,-2] = 1
#Some checks
#print('Input shape', X.shape,X.shape[1],X.shape[2])
#printy = [list(w_vec).index(True) for w_vec in X[0]]
#print(printy)
#printy = [chars[a] for a in list(printy)]
print(text[0])
#print(chars)
#print(printy)

intermediate_dim = 22
latent_dim = 12
epsilon_std = 1.0


sen_len = X.shape[1] #Length of sequence
voc_len = X.shape[2] #Dimensionality of sequence
print('Build model...')
inputs  = Input(shape=(sen_len,voc_len),name='inputs')
encoded = LSTM(128,name='lstm_encoder')(inputs)
decoded = RepeatVector(sen_len,name='repeat_decoder')(encoded)
decoded = LSTM(voc_len,name='lstm_decoder',return_sequences=True)(decoded)

autoencoder = Model(inputs,decoded)
encoder     = Model(inputs,encoded)

#VAE model based on same input
x = Input(shape=(sen_len,voc_len))
h = LSTM(intermediate_dim, name='h_layer',activation='relu')(x)
z_mean = Dense(latent_dim,name='z_mean')(h)
z_log_sigma= Dense(latent_dim,name='z_log_sigma')(h)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    print('z_m',z_mean.shape,'z_s',z_log_sigma.shape)
    return z_mean + K.exp(z_log_sigma) * epsilon

#>>>>>>>>>>>>>>>> 
z = Lambda(sampling,output_shape=(latent_dim,))([z_mean,z_log_sigma])

repeat_z = RepeatVector(sen_len,name='repeat_z')(z)
#WHAT TYPE LAYERS, WHAT ACTIVATIONS?
decoder_h = Dense(intermediate_dim,name='dec_h',activation='relu')
#what order sen_len, voc_len?
decoder_mean = LSTM(voc_len,name='dec_mean',return_sequences=True,activation='softmax')
h_decoded = decoder_h(repeat_z)
x_decoded_mean = decoder_mean(h_decoded)

decoder_input = Input(shape=(latent_dim,))
_repeat_z = RepeatVector(sen_len)(decoder_input)
_h_decoded = decoder_h(_repeat_z)
_x_decoded_mean = decoder_mean(_h_decoded)

ex_repeat_z = RepeatVector(sen_len)(z_mean)
ex_h = Dense(intermediate_dim)(ex_repeat_z)
example_y = LSTM(voc_len,return_sequences=True)(ex_h)

example_vae  = Model(x,example_y)


def vae_loss(x, x_decoded_mean):
    #orig xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    #xent_loss = K.mean(objectives.binary_crossentropy(x, x_decoded_mean),axis=1)
    xent_loss = K.mean(objectives.categorical_crossentropy(x, x_decoded_mean))
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    #Extra K.mean added to use with sequential input shape
    #orig kl_loss = - 0.5 * K.mean(K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1))
    #kl_loss = - 0.5 * K.mean(1 + tf.reshape(z_log_sigma,[-1]) - K.square(tf.reshape(z_mean,[-1])) - K.exp(tf.reshape(z_log_sigma,[-1])), axis=-1)
    
    #How to track the two losses in Tensorboard?

    #kl_loss weird !!!!
    return  kl_loss + xent_loss


def translate_back(predictions):
    preds_num = [p.tolist().index(max(p)) for p in predictions]
    pred_trans = [indices_char[pn] for pn in preds_num]
    return pred_trans

if(train_anew):
    print('Training:')

    vae = Model(x,x_decoded_mean)
    vae_enc = Model(x,z_mean)
    vae_dec = Model(decoder_input, _x_decoded_mean)

#Train models
    #IETS MET RMSPROP>>>>>>>>>>>>>>.
    #orig. optim: rmsprop
    #vae.compile(optimizer='rmsprop',loss=vae_loss)

    #example_vae.compile(optimizer='adagrad',loss='binary_crossentropy')

    vae.summary()
    vae.compile(optimizer='rmsprop',loss=vae_loss)

    #train_range = range(0,1000000)
    train_percentage    = 0.9
    upper_train_bound   = int(np.floor(train_percentage*len(X[0])*perc/100))
    upper_val_bound     = int(np.floor(len(X[0])*perc/100))
    #Cut down ranges to avoid sampling size mismatch in last batch of epoch
    upper_train_bound   = upper_train_bound - (upper_train_bound % batch_size)
    upper_val_bound     = upper_val_bound - (upper_val_bound %batch_size)
    print('BOUNDS',upper_train_bound,upper_val_bound)
    train_range = range(0,upper_train_bound)
    val_range   = range(upper_train_bound,upper_val_bound)

    print('train_range',X[train_range].shape)
    print('val_range',X[val_range].shape)

    #vae.fit(X[train_range,:,:],X[train_range,:,:],

    #<-------------------------
    #Set up training size to be exact multitude of batch_size

    print('overfit',upper_train_bound % batch_size)
    vae.fit(X[train_range,:,:],X[train_range,:,:],
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data =(X[val_range,:,:],X[val_range,:,:]),
            callbacks=[TensorBoard(log_dir='/home/thomas/Documents/Studie/Thesis/TensorBoard/vae')]
            )

    vae.save('my_vae'+'_bs:'+str(batch_size)+'_epochs:'+str(epochs)+'_'+str(datetime.datetime.now())+'.h5')  # creates a HDF5 file 'my_model.h5'
else:
    #CHANGE LOADOUT!
    vae = load_model('my_vae_bs:100_epochs:11_current.h5', custom_objects={'batch_size':batch_size,'latent_dim': latent_dim, 'epsilon_std': epsilon_std, 'vae_loss': vae_loss})
    vae.summary()
    vae.compile(optimizer='rmsprop',loss=vae_loss)
    print('Starting interpolation')
    #How to load partial models?
    vae_enc = Model(x,z_mean)
    vae_dec = Model(decoder_input, _x_decoded_mean)
    vae_enc.compile(optimizer='rmsprop',loss=vae_loss)
    vae_dec.compile(optimizer='rmsprop',loss=vae_loss)

vp1 = vae_enc.predict(X[range(0,100),:,:])
print('prediction enc')
print(vp1[0])
print(vp1[1])
vp_mean = ( vp1[0]+vp1[2] )/2
print(vp_mean)    
vp2 = vae_dec.predict(vp1)
print('prediction dec')
print(vp2[0].shape)
print(translate_back(list(vp2[0])))