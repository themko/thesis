import time
#import pandas as pd
import numpy as np
import keras
import argparse
from keras import backend as K
from keras.models import Sequential,Model,load_model,model_from_json,model_from_yaml
from keras.layers import *
from collections import defaultdict, OrderedDict, Counter
#from ptb import PTB
from keras import objectives
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from tqdm import tqdm
import datetime


splits = ['train', 'valid']
min_occ = 1
max_sequence_length = 60
create_data = 'store_true'
data_dir = 'data'

def anneal(step, total, k = 1.0, anneal_function='logistic'):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-total/2))))
        elif anneal_function == 'linear':
            return min(1, step/total)

def read_words(filename):
    with open(filename) as f:
        return f.read().replace('\n', '<eos>').split()

def build_vocab(filename):
    data = read_words(filename)
    counter = Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    #Remove special tags from word list. Added manually to start of word dictonaries later
    count_pairs = [cp if cp[0]!='<unk>' and cp[0]!='<eos>' else None for cp in count_pairs]
    count_pairs = filter(lambda a: a != None, count_pairs)

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((i, v) for v, i in word_to_id.items())
    return word_to_id, id_to_word

def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word !='<unk>'and word != '<eos>']


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

def go(args):
    vocab_size = args.vocab_size
    seq_len = args.sequence_length
    batch_size = args.batch_size
    lr = args.learning_rate
    embed_dim = args.embed_dim
    latent_dim = args.latent_dim
    lstm_dim = args.lstm_dim
    num_layers = 1 #???
    h_factor = 1#(2 if Bidir else 1)*num_layers
    kl_incr = 0.0025
    epsilon_std = 1.0
    dropout_rate = args.dropout_rate
    epochs = args.epochs

    hidden_factor = 1 * num_layers
    Bidir = True
    if Bidir== True:
        hidden_factor = 2 * num_layers

    word_to_id, id_to_word = build_vocab('data/ptb.train.txt')

    #Add additional tags to dictionaries
    #COULD MAYBE BE SWITCHED AROUND IN ORDER IF NECESSARY
    index_from = 4 
    word_to_id = {k: (v + index_from) for k, v in word_to_id.items()}
    word_to_id['<pad>'] = 0
    word_to_id['<sos>'] = 1
    word_to_id['<unk>'] = 2
    word_to_id['<eos>'] = 3
    id_to_word = dict((i, v) for v, i in word_to_id.items())

    file = file_to_word_ids('data/ptb.train.txt',word_to_id)

    eos_id = word_to_id['<eos>']
    unk_id = word_to_id['<unk>']

    X_input = []
    line_ind = 0
    #Cut file into lines
    for idx, number in enumerate(file):
        if number >= vocab_size:
            file[idx] = unk_id
        if number == eos_id or idx - line_ind >= seq_len-1:
            X_input.append(file[line_ind:idx+1])
            line_ind=idx+1

    print()
    X_input.sort(key=len)

    X_batches = []
    for i in range(0,len(X_input)-len(X_input)%batch_size,batch_size):
        X_batch = X_input[i:i+batch_size]
        bmlen = max([len(a)for a in X_batch])
        batch_input = sequence.pad_sequences(X_batch,maxlen=bmlen, dtype='int32',padding='post',truncating='post')
        X_batches.append(np.array(batch_input))

    input = Input(shape=(None,))
    embedding = Embedding(vocab_size,embed_dim,input_length=None)
    embed = embedding(input)

    h =Bidirectional(LSTM(lstm_dim))(embed)
    z_mean  = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    kl = KLLayer()
    h = kl([z_mean,z_log_sigma])

    def vae_loss(x, x_decoded_mean):
    #orig xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        xent_loss = K.mean(objectives.categorical_crossentropy(x, x_decoded_mean)) #*X_seq.shape[1] 
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)    #Extra K.mean added to use with sequential input shape
        return  kl_loss + xent_loss

    sample_z = Sample()(h)
    exp_z = Dense(lstm_dim*hidden_factor)(sample_z)
    #Do proper DROPOUT!!
    shift_input = Input(shape=(None,))
    shift_embed = embedding(shift_input)
    decoder_input = SpatialDropout1D(rate=dropout_rate)(shift_embed)

    #GET STATES BACK IN THERE!!
    decoded = LSTM(lstm_dim*hidden_factor,return_sequences=True)(decoder_input,initial_state=[exp_z,exp_z])
    output  = TimeDistributed(Dense(vocab_size))(decoded)

    model_vae = Model(input, sample_z)
    model_vae.compile(optimizer='adam',loss='mse')
    #model_vae.summary()

    #model_vae.fit(input_array,output_array,batch_size=batch_size)
    model_test = Model([input,shift_input],output)
    opt = keras.optimizers.Adam(lr=lr)
    model_test.compile(optimizer=opt,loss=keras.losses.sparse_categorical_crossentropy)
    model_test.summary()

    print('xs',len(X_batches))

    #For debugging:
    #X_batches = X_batches[0:3]
    
    model_string = 'Bowman'+str(datetime.datetime.now())+'_eps:'+str(epochs)+'_bs:'+str(batch_size)+'_dims:'+str([(vocab_size,seq_len),embed_dim,lstm_dim,latent_dim])
    print(model_string)
    #For n epochs
    d_eval = 1
    for i in range(epochs):
        print('Epoch: '+str(i+1))
        print('Set KL weight to ', anneal(i, epochs))
        K.set_value(kl.weight, anneal(i, epochs))
        #

        if(i%d_eval ==0):
            b = X_batches[1][0:10]
            n = b.shape[0]
            b_shift = np.concatenate([np.ones((n, 1)), b], axis=1) 

            out = model_test.predict([b, b_shift])
            y = np.argmax(out, axis=-1)

            for i in range(b.shape[0]):
                in_values = [' '.join([id_to_word[aa] for aa in a]) for a in b]
                out_values= [' '.join([id_to_word[aa] for aa in a]) for a in y]
                for i in range(len(in_values)):
                    print('in   ', in_values[i] )
                    print('out   ', out_values[i] ,'\n')
        for batch in tqdm(X_batches):
        #Batches
        #i is the starting index of the batch
            n = batch.shape[0]
            batch_shift = np.concatenate([np.ones((n,1)),batch],axis=1) #Add <sos> tag to start-of-sequence
            batch_out   = np.concatenate([batch,np.zeros((n,1))],axis=1)  #Why this padding here? 
            #Alter the batches aprropriately!!
            model_test.train_on_batch([batch,batch_shift],batch_out[:,:,None])
        model_yaml = model_test.to_yaml()
        with open("../models/"+model_string+".yaml","w") as yaml_file:
            yaml_file.write(model_yaml)
        model_test.save_weights('../models/'+model_string+'.h5')



if __name__ == '__main__':
    parser = argparse.ArgumentParser("VAE model arguments")
    parser.add_argument("--vocab_size", type=int, default=10000, help="How many words in vocab")
    parser.add_argument("--sequence_length", type=int, default=100, help="how long the maximum seq ")
    parser.add_argument("--learning_rate", type=int, default=0.001, help="how high the lr ")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--rnn_size", type=int, default=128, help="how many LSTM units per layr")
    parser.add_argument("--lstm_layers", type=int, default=3, help="how many LSTM layers")
    parser.add_argument("--dropout_rate", type=int, default=0.5, help="dropout rate for a ")
    parser.add_argument("--lstm_dim", type=int, default=256, help="size of LSTM dim")
    parser.add_argument("--latent_dim", type=int, default=16, help="size of latent/hidden dim")
    parser.add_argument("--embed_dim", type=int, default=300, help="Embedding size"
                                                                      "droupout layer inserted "
                                                                      "after every LSTM layer")
    parser.add_argument("--bidirectional", action="store_true",
                        help="Whether to use bidirectional LSTM. If true, inserts a backwards LSTM"
                        " layer after every normal layer.", default=True)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--validation_steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()
    go(args)
