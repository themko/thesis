'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

from keras.datasets import imdb
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelBinarizer
import keras
import numpy as np
import random
import sys
import itertools

word_num = False
split_lines  = True

if(word_num):   
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=None)
    text = []
    [text.append(x) for y in X_train[0:50] for x in y]
else:
    #path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    #path = get_file('imdb_train.txt', origin='file:/home/thomas/Documents/Studie/Thesis/imdb_train.txt')
    path = get_file('French.txt', origin='file:/home/thomas/Documents/Studie/Thesis/data/French.txt')
    #path2 = get_file('English.txt', origin='file:/home/thomas/Documents/Studie/Thesis/data/English.txt')
    text = open(path).read().lower()

    #Take portion of total
    divider = 10
    text = text[0:int(round(len(text)/divider))]

    #text2 = text2[0:int(round(len(text2)/divider))]


print('corpus length:', len(text))

chars = sorted(list(set(text)))
#print('chars',chars)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
    #print('testing')
    #print(sentences,'\n',next_chars)
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

"""
print('Loading imdb data')
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=maxlen)
print('Sequences padded')
X_imdb_train =  sequence.pad_sequences(X_train, maxlen=maxlen)
X_imdb_test = sequence.pad_sequences(X_test, maxlen=maxlen)
enc =LabelBinarizer()
top_words = 50
enc.fit(range(top_words)) #Possible problem, imdb.load_data ? starts at index 1
print('One-hot encoding imdb')
X_enc_train = ([enc.transform(x).tolist() for x in X_imdb_train])
X_enc_test = ([enc.transform(x).tolist() for x in X_imdb_test])

print(X_enc_train[0])
print(X[0])
"""

#Print imdb data into Nietzsche format




# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
    
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

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=128,
              epochs=1)

    #Make start index differnt for word_num: always start at the start of the review
    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        sentence = text[start_index: start_index + maxlen]
        if(not word_num):
            generated = ''
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)            
            for i in range(400):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()  
        else:
            generated = []
            [generated.append(x) for x in sentence]
            print('----- Generating with seed: "' , translate_back(sentence) , '"\n')
            #print(generated)            
            for i in range(400):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]
                generated.append(next_char)
                del sentence[0]
                sentence.append(next_char)
                #print(next_char)
            #print(generated)
            print(translate_back(generated))