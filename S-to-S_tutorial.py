# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 13:22:36 2017

@author: Tom Koenen
"""
#S-to-S classifier
import numpy as np
import keras
from keras import backend as K
from keras import metrics
from keras import objectives
from keras.datasets import imdb
from keras.models import Sequential,Model
from keras.layers import Input,Dense, LSTM, RepeatVector, Lambda,Layer
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
#fix random seed
np.random.seed(7)
#Load data
top_words = 600 #5000
index_from = 3
num_epochs = 3
batch_size=64
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
print('Data loaded')
#Pad sequences
max_review_length = 50 #500
print X_train.shape, (X_train[0])
X_train =  sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

print('x_test',X_test.shape)
def translate_back(sentence):
    #Code from https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset#44635045
    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k:(v+index_from) for k,v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    id_to_word = {value:key for key,value in word_to_id.items()}
    print(' '.join(id_to_word[id] for id in sentence ))

translate_back(X_train[3])
#

"""
#create the Sequence classifier model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words,embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=3,batch_size=64)
"""

print('Creating autoencoder')
#create S-to-S autoencoder
input_dim = top_words
embedding_vector_length = 32
timesteps = max_review_length
latent_dim=10
intermediate_dim = 256
batch_size=20
epochs = 1
epsilon_std = 1.0

x = Input(shape=(max_review_length,),name='main_input')
embeds = Embedding(output_dim=512,input_dim=top_words,input_length=max_review_length)(x)
print(embeds)
h = LSTM(intermediate_dim, activation='relu',name='h')(embeds)
#h = Dense(intermediate_dim, activation='relu',name='embedding_layer')(x)
print(h)
z_mean = Dense(latent_dim,name='z_mean')(h)
z_log_sigma = Dense(latent_dim,name='z_log_sigma')(h)
print(z_mean,z_log_sigma)
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,),name='z')([z_mean, z_log_sigma])
print(z)

repeat_decoder = RepeatVector(max_review_length)
decoder_h = Dense(intermediate_dim,activation='relu',name='decoder_h')


#decoder_mean=Dense(input_dim,activation='sigmoid', name='decoder_mean')
#Why did this change work out?
decoder_mean=Dense(timesteps,activation='sigmoid', name='decoder_mean')

#h_repeat = repeat_decoder(z)
#h_decoded = decoder_h(h_repeat)
h_decoded= decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator =  Model(decoder_input,_x_decoded_mean)

"""
# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        #HERE~
        xent_loss = input_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x,y)
"""

def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.categorical_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    #print('kl_loss: ',get_value(kl_loss),' xent_loss: ',xent_loss)
    return kl_loss + xent_loss

vae = Model(x,x_decoded_mean)
encoder = Model(x,z_mean)

vae.compile(optimizer='rmsprop', loss=vae_loss)
X_train_small = X_train[0:2500]
vae.fit(X_train_small,X_train_small,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data =(X_test,X_test))

ys = vae.predict(X_test)
print('ys: \n',ys)
#sentence = X_train[0]
#print('inp,',sentence.dtype) #shape=(50,) #dtype=int32

