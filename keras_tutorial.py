#Standar (non-var) Sequence-to-Sequence encoder 

from __future__ import print_function
from keras.models import Sequential,Model
from keras.layers import Dense, Activation,Input,RepeatVector
from keras.layers import LSTM, Lambda
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.datasets import imdb
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelBinarizer
import keras
import numpy as np
import random
import sys
import itertools


# a = Input(shape=(20,))
# b = Dense(10)(a)

# model = Model(a,b)
# print(a.shape)
# print(b.shape)

# batch_size = 100
# epsilon_std = 1.0
# x = Input(shape=(41,))
# #x = Input(shape=(sen_len,))
# h = Dense(22, name='h_layer',activation='relu')(x)

# print('vae_x.shape', x.shape)
# print('vae_h.shape', h.shape)
# z_mean = Dense(12)(h)
# z_log_sigma= Dense(12)(h)
# print('zm',z_mean.shape)

# def sampling(args):
#     z_mean, z_log_sigma = args
#     epsilon = K.random_normal(shape=(batch_size, 12),
#                               mean=0., stddev=epsilon_std)
#     return z_mean + K.exp(z_log_sigma) * epsilon

# z = Lambda(sampling,output_shape=(12,))([z_mean,z_log_sigma])

# print('vae_z_m.shape', z_mean.shape)
# print('vae_z_s.shape', z_log_sigma.shape)
# print('vae_z.shape', z.shape)

# #WHAT TYPE LAYERS, WHAT ACTIVATIONS?
# decoder_h = Dense(22,name='dec_h',activation='relu')
# #decoder_h = LSTM(intermediate_dim,name='dec_h')
# #what order sen_len, voc_len?
# decoder_mean = Dense(41,name='dec_mean',activation='sigmoid')
# h_decoded = decoder_h(z)

# print('vae_h_dec.shape', h_decoded.shape)
# x_decoded_mean = decoder_mean(h_decoded)

# print('vae_x_dec.shape',x_decoded_mean.shape )

# vae = Model(x,z_mean)




"""
import keras
from keras.preprocessing import sequence
from keras.datasets import imdb
from sklearn.preprocessing import LabelBinarizer
from keras.datasets import imdb
"""

"""
#Load data
top_words = 600 #5000
index_from = 3
num_epochs = 3
batch_size=64

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+index_from) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
id_to_word = {value:key for key,value in word_to_id.items()}

def translate_back(sentence,id_to_w):
    #Code from https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset#44635045
    
    return (' '.join(id_to_word[id] for id in sentence ))

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=None)
print('Data loaded')
#Pad sequences
max_review_length = 500 #500
print X_train.shape, (X_train[0])
print(translate_back(X_train[0],id_to_word))


print('translate 100')
imdb_list_train = [translate_back(x,id_to_word) for x in X_train[0:5000]]
print('writing file')
thefile = open('imdb_train.txt', 'w')
for item in imdb_list_train:
  thefile.write("%s\n" % item.encode('utf8'))

"""
