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

bilingual = False
train_anew = True

path = get_file('French.txt', origin='file:/home/thomas/Documents/Studie/Thesis/data/French.txt')
path2 = get_file('English.txt', origin='file:/home/thomas/Documents/Studie/Thesis/data/English.txt')

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
lines_text2 = fo2.readlines()
lines_text = lines_text1 + lines_text2
#Keep track of indices of the start of each verse
lines_index = np.array([len(x) for x in lines_text])
lines_index = list(np.append(np.array(0),lines_index.cumsum()))
labels = np.concatenate((np.zeros(len(lines_text1)),np.ones(len(lines_text2))),axis=0)
lines_labels = zip(labels,lines_text)
"""
test_begin = random.choice(lines_index)
words_in_text1 = (sum([len(x) for x in lines_text1]))
test_label = None
if(test_begin <words_in_text1 and test_begin  + 20 < words_in_text1):
    test_label= 0.
elif (test_begin >words_in_text1 and test_begin  + 20 >words_in_text1):
    test_label = 1.
else:
    test_label = 2.
"""
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

#Build the new bilingual database
maxlen_line = max(max([len(x) for x in lines_text1]), max([len(x) for x in lines_text2]))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

#Check how many of those are from the first corpus
sentences1 = []
next_chars1 = []
for i in range(0, len(text1) - maxlen, step):
    sentences1.append(text1[i: i + maxlen])
    next_chars1.append(text1[i + maxlen])
print('nb sequences from corpus1:', len(sentences1)) #1409719
#

#Add language labels to vectorization
print('Vectorization...')
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
# print('First word of sentence', X[0:100,0,:])
# print('example',X[500,0,60:69])
# print('example2',X[2789222,0,60:69])
print('Input shape', X.shape,X.shape[1],X.shape[2])
printy = [list(w_vec).index(True) for w_vec in X[0]]
print(printy)
printy = [chars[a] for a in list(printy)]
print(text[0])
print(chars)
print(printy)

intermediate_dim = 22
latent_dim = 12
batch_size = 64
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
z_mean = Dense(latent_dim)(h)
z_log_sigma= Dense(latent_dim)(h)

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
decoder_mean = LSTM(voc_len,name='dec_mean',return_sequences=True,activation='sigmoid')
h_decoded = decoder_h(repeat_z)
x_decoded_mean = decoder_mean(h_decoded)

decoder_input = Input(shape=(latent_dim,))
_repeat_z = RepeatVector(sen_len)(decoder_input)
_h_decoded = decoder_h(_repeat_z)
_x_decoded_mean = decoder_mean(_h_decoded)


print('vae_x.shape', x.shape)
print('vae_h.shape', h.shape)
print('zm', z_mean.shape)
print('vae_z_m.shape', z_mean.shape)
print('vae_z_s.shape', z_log_sigma.shape)
print('vae_z.shape', z.shape)
print('vae_h_dec.shape', h_decoded.shape)
print('vae_x_dec.shape',x_decoded_mean.shape )

vae = Model(x,x_decoded_mean)
vae_enc = Model(x,z_mean)
vae_dec = Model(decoder_input, _x_decoded_mean)

def vae_loss(x, x_decoded_mean):
    #   orig xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    xent_loss = K.mean(objectives.binary_crossentropy(x, x_decoded_mean),axis=1)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    #Extra K.mean added to use with sequential input shape
    #?????
    #?????
    #kl_loss = - 0.5 * K.mean(K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1))
    #kl_loss = - 0.5 * K.mean(1 + tf.reshape(z_log_sigma,[-1]) - K.square(tf.reshape(z_mean,[-1])) - K.exp(tf.reshape(z_log_sigma,[-1])), axis=-1)
    print('kl_loss',kl_loss.shape)
    print('xent_loss',xent_loss.shape)
    #InvalidArgumentError (see above for traceback): Incompatible shapes: [100,41] vs. [100]
    return xent_loss + kl_loss

#Compile encoders
#autoencoder.compile(optimizer='adagrad',loss='categorical_crossentropy')

if(train_anew):
#Train models
    #IETS MET RMSPROP>>>>>>>>>>>>>>.
    #orig. optim: rmsprop
    vae.compile(optimizer='rmsprop',loss=vae_loss)
    vae.summary()
    train_range = range(0,10000)
    val_range   = range(10000,20000)

    vae.fit(X[train_range,:,:],X[train_range,:,:],
            shuffle=True,
            epochs=1,
            batch_size=batch_size,
            validation_data =(X[val_range,:,:],X[val_range,:,:]),
            #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
            )

    vae.save('my_vae.h5')  # creates a HDF5 file 'my_model.h5'
else:
    vae = load_model('my_vae.h5')
    vae.compile(optimizer='rmsprop',loss=vae_loss)

x = np.zeros((100,len(X[0]),len(X[0][0])))
x = X[0:100]
preds = vae.predict(x,batch_size=batch_size, verbose=0)
print('Preds',preds)
preds = vae.predict(x)[0]
print(preds)
print(X[0:1])
index_from = 3
def translate_back(sentence):
    #Code from https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset#44635045
    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k:(v+index_from) for k,v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    id_to_word = {value:key for key,value in word_to_id.items()}
    return (' '.join(id_to_word[id] for id in sentence ))