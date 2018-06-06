import keras

import keras.backend as K
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import \
    Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten, Input, Reshape, LSTM, Embedding, RepeatVector,\
    TimeDistributed, Bidirectional, Concatenate, Lambda, SpatialDropout1D
from keras.optimizers import Adam
from tensorflow.python.client import device_lib

from keras.utils import multi_gpu_model

import tensorflow as tf

from sklearn import datasets

from tqdm import tqdm
import math, sys, os
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from argparse import ArgumentParser

from keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Reshape, MaxPooling2D, UpSampling2D, Flatten, Cropping2D
from keras.models import Model, Sequential
from keras.engine.topology import Layer

#import util

INDEX_FROM = 3
CHECK = 5

def anneal(step, total, k = 1.0, anneal_function='logistic'):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-total/2))))
        elif anneal_function == 'linear':
            return min(1, step/total)

class KLLayer(Layer):

    """
    Identity transform layer that adds KL divergence
    to the final model loss.
    http://tiao.io/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/
    """

    def __init__(self, weight = K.variable(1.0), *args, **kwargs):
        self.is_placeholder = True
        self.weight = weight
        super().__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

class Sample(Layer):

    """
    Performs sampling step
    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super().__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        eps = Input(tensor=K.random_normal(shape=K.shape(mu) ))

        z = K.exp(.5 * log_var) * eps + mu

        return z

    def compute_output_shape(self, input_shape):
        shape_mu, _ = input_shape
        return shape_mu


def decode_imdb(seq):

    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + INDEX_FROM) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    id_to_word = {value: key for key, value in word_to_id.items()}

    return ' '.join(id_to_word[id] for id in seq)

def sparse_loss(y_true, y_pred):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)
def go(options):
    slength = options.max_length
    top_words = options.top_words
    lstm_hidden = options.lstm_capacity

    print('devices', device_lib.list_local_devices())

    if options.task == 'europarl':

        dir = options.data_dir
        x, x_vocab_len, x_word_to_ix, x_ix_to_word, _, _, _, _ = \
            util.load_data(dir+os.sep+'europarl-v8.fi-en.en', dir+os.sep+'europarl-v8.fi-en.fi', vocab_size=top_words)

        # Finding the length of the longest sequence
        x_max_len = max([len(sentence) for sentence in x])

        print('max sequence length ', x_max_len)
        print(len(x_ix_to_word), 'distinct words')

        x = util.batch_pad(x, options.batch)

        # Padding zeros to make all sequences have a same length with the longest one
        # x = sequence.pad_sequences(x, maxlen=slength, dtype='int32', padding='post', truncating='post')
        # y = sequence.pad_sequences(y, maxlen=slength, dtype='int32', padding='post', truncating='post')

        def decode(seq):
            print(seq)
            return ' '.join(x_ix_to_word[id] for id in seq)

    else:
        # Load only training sequences
        (x, _), _ = imdb.load_data(num_words=top_words)

        # rm start symbol
        x = [l[1:] for l in x]

        # x = sequence.pad_sequences(x, maxlen=slength+1, padding='post', truncating='post')
        # x = x[:, 1:] # rm start symbol

        #x = util.batch_pad(x, options.batch) #CHANGED!

        decode = decode_imdb

    print('Data Loaded.')

    #print(sum([b.shape[0] for b in x]), ' sentences loaded') CHANGED!!

    # for i in range(3):
    #     print(x[i, :])
    #     print(decode(x[i, :]))


    ## Define model
    input = Input(shape=(None, ))

    embedding = Embedding(top_words, options.embedding_size, input_length=None)

    embedded = embedding(input)
    h = Bidirectional(LSTM(lstm_hidden))(embedded)

    zmean = Dense(options.hidden)(h)
    zlsigma = Dense(options.hidden)(h)

    kl = KLLayer()
    h = kl([zmean, zlsigma]) # computes the KL loss and stores it for later

    z = Sample()(h)  # implements the reparam trick

    z_exp = Dense(lstm_hidden)(z)

    input_shifted = Input(shape=(None, ))
    embedded_shifted = embedding(input_shifted)

    embedded_shifted = SpatialDropout1D(rate=options.dropout)(embedded_shifted)

    # zrep = RepeatVector(slength + 1)(z)
    # catted = Concatenate(axis=2)( [zrep, embedded_shifted] )

    h = TimeDistributed(Dense(lstm_hidden))(embedded_shifted)
    h = LSTM(lstm_hidden, return_sequences=True)(h, initial_state=[z_exp, z_exp])

    out = TimeDistributed(Dense(top_words))(h)

    auto = Model([input, input_shifted], out)

    if options.num_gpu is not None:
        auto = multi_gpu_model(auto, gpus=options.num_gpu)

    opt = keras.optimizers.Adam(lr=options.lr)

    auto.compile(opt, keras.losses.sparse_categorical_crossentropy)
    auto.summary()
    epochs = 0
    while epochs < options.epochs:

        print('Set KL weight to ', anneal(epochs, options.epochs))
        K.set_value(kl.weight, anneal(epochs, options.epochs))

        for batch in tqdm(x):
            n = batch.shape[0]
            batch_shifted = np.concatenate([np.ones((n, 1)), batch], axis=1)  # prepend start symbol
            batch_out = np.concatenate([batch, np.zeros((n, 1))], axis=1)[:, :, None]  # append pad symbol

            auto.train_on_batch([batch, batch_shifted], batch_out)

        epochs += options.out_every

        # show reconstructions for some sentences from batch 90
        b = x[90]
        n = b.shape[0]
        b_shifted = np.concatenate([np.ones((n, 1)), b], axis=1)  # prepend start symbol

        out = auto.predict([b, b_shifted])
        y = np.argmax(out, axis=-1)

        for i in range(b.shape[0]):
            print('in   ',  decode(b[i, :]))
            print('out   ', decode(y[i, :]))
            print()

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=150, type=int)

    parser.add_argument("-E", "--embedding-size",
                        dest="embedding_size",
                        help="Size of the word embeddings on the input layer.",
                        default=300, type=int)

    parser.add_argument("-o", "--output-every",
                        dest="out_every",
                        help="Output every n epochs.",
                        default=1, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.001, type=float)

    parser.add_argument("-b", "--batch-size",
                        dest="batch",
                        help="Batch size",
                        default=32, type=int)

    parser.add_argument("-t", "--task",
                        dest="task",
                        help="Task",
                        default='imdb', type=str)


    parser.add_argument("-D", "--data-directory",
                        dest="data_dir",
                        help="Data directory",
                        default='./data', type=str)

    parser.add_argument("-d", "--dropout-rate",
                        dest="dropout",
                        help="The word dropout rate used when training the decoder",
                        default=0.5, type=float)

    parser.add_argument("-H", "--hidden-size",
                        dest="hidden",
                        help="Latent vector size",
                        default=16, type=int)

    parser.add_argument("-L", "--lstm-hidden-size",
                        dest="lstm_capacity",
                        help="LSTM capacity",
                        default=256, type=int)


    parser.add_argument("-g", "--num-gpu",
                        dest="num_gpu",
                        help="How many GPUs to use",
                        default=None, type=int)

    parser.add_argument("-m", "--max_length",
                        dest="max_length",
                        help="Max length",
                        default=None, type=int)

    parser.add_argument("-w", "--top_words",
                        dest="top_words",
                        help="Top words",
                        default=10000, type=int)

    options = parser.parse_args()

    print('OPTIONS', options)

go(options)