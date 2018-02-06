import keras
from keras import backend as K

from keras.datasets import imdb,mnist
from keras.preprocessing import sequence

from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives

#Parameters
max_features = 25000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)

input_dim = 80

batch_size = 32
timesteps = 3
inter_dim = 128
latent_dim = 256


print("Building model...")

inputs = Input(max_features,timesteps,input_dim)  
#LSTM-encoding
h = LSTM(inter_dim, activation = 'relu')(inputs)

#VAE z-layer
z_mu = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)

def sampling(args):
	z_mu, z_log_sigma = args
	epsilon = K.random_normal(	shape=(batch_size,latent_dim), 
								mean=0, stddev = 1.)
	return z_mu + K.exp(z_log_sigma) * 1.

z =  Lambda(sampling, output_shape = (latent_dim,))([z_mu,z_log_sigma])

#decoded LSTM layer
decoder_h = LSTM(inter_dim,return_sequences=True)
decoder_mu = LSTM(input_dim,return_sequences=True)

h_decoded = RepeatVector(timesteps)(z)
h_decoded = decoder_h(h_decoded)

#decoded layer
inputs_decoded_mu = decoder_mu(h_decoded)

vae = Model(inputs,inputs_decoded_mu)

##
#Other encoders and decoders
##

def vae_loss(inputs,inputs_decoded_mu):
	cross_ent_loss = objectives.mse(inputs,inputs_decoded_mu)
	kl_loss = -.5 * K.mean(1+ z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma))
	loss = cross_ent_loss + kl_loss
	print('kl_loss = ',kl_loss)
	return loss

print('Compiling model...')
vae.compile(optimizer='rmsprop', loss=vae_loss)

print("Building inputs...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)

vae.fit(x_train,x_train,
		shuffle=True,
		epochs = 10,
		batch_size = batch_size,
		validation_data=(x_test,x_test))