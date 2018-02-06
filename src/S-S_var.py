#Standar (non-var) Sequence-to-Sequence encoder 
import numpy as np
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential,Model
from keras.layers import Input,Dense, LSTM, RepeatVector,Layer,TimeDistributed,Lambda
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer


#Load data
top_words = 60 #5000
max_review_length = 50 #500
index_from = 3
num_epochs = 2
batch_size=64
intermediate_dim = 64
latent_dim = 16
epsilon_std = 1.

#Creating autoencoder
x = Input(shape=(max_review_length,top_words),name='main_input')
dense_in = TimeDistributed(Dense(16,name='dense_in'))(x)
encoded = LSTM( intermediate_dim,input_shape=(intermediate_dim,max_review_length), 
                activation='relu',name='encoded')(dense_in)
#right value for k??
k= max_review_length
#encoded_repeat = RepeatVector(k,name='encoded_repeat')(encoded)
#Add sampling step here
z_mean = Dense(latent_dim)(encoded)
z_log_sigma = Dense(latent_dim)(encoded)

print('test zm',z_mean.shape)
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
##

decoded = LSTM(16,name='lstm_decoded',return_sequences=True)(encoded_repeat)
dense_out = TimeDistributed(Dense(top_words,activation='softmax'),name='time_dist_dense_out')(decoded)

print(x)
print(dense_in)
print(encoded)
print(encoded_repeat)
print(decoded)
print(decoded_repeat)
print(dense_out)


(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
print('Data loaded')

X_train =  sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
print('Sequences padded')
enc =LabelBinarizer()
enc.fit(range(top_words)) #Possible problem, imdb.load_data ? starts at index 1

X_enc_in = ([enc.transform(x).tolist() for x in X_train])
X_enc_out = ([enc.transform(x).tolist() for x in X_test])
#Transpose matrix
#X_enc_in= map(list, zip(*X_enc_in))
#X_enc_out= map(list, zip(*X_enc_out))
print('One-hot encoded')


print('Commpiling model')
SS = Model(x,dense_out)
S_encoder = Model(x,encoded)
SS.compile(optimizer='adagrad',loss='categorical_crossentropy')
print('Fitting model')
SS.fit(X_enc_in,X_enc_in,
        shuffle=True,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data =(X_enc_out,X_enc_out))