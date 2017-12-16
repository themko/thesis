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
from keras.layers import LSTM, La
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
    path2 = get_file('English.txt', origin='file:/home/thomas/Documents/Studie/Thesis/data/English.txt')

    text1 = open(path).read().lower()
    text2 = open(path2).read().lower()
    text  = text1 + text2
    #Get starting indices
    fo1 = open("data/French.txt","r")
    fo2 = open("data/English.txt","r")

    lines_text1 = fo1.readlines()
    lines_text2 = fo2.readlines()
    lines_text = lines_text1 + lines_text2
    #Keep track of indices of the start of each verse
    #lines_index = np.array([len(x) for x in lines_text1])
    lines_index = np.array([len(x) for x in lines_text])
    lines_index = list(np.append(np.array(0),lines_index.cumsum()))
    labels = np.concatenate((np.zeros(len(lines_text1)),np.ones(len(lines_text2))),axis=0)
    lines_labels = zip(labels,lines_text)
    print('ll', (lines_labels[31100:31110]))
    test_begin = random.choice(lines_index)
    #33102 lines in  the bible
    print( text[test_begin:test_begin+20] )
    print('textnum', test_begin)
    words_in_text1 = (sum([len(x) for x in lines_text1]))
    test_label = None
    if(test_begin <words_in_text1 and test_begin  + 20 < words_in_text1):
        test_label= 0.
    elif (test_begin >words_in_text1 and test_begin  + 20 >words_in_text1):
        test_label = 1.
    else:
        test_label = 2.
    print('label', test_label)

    #Take portion of total
    divider = 10
    mini_text = text[0:int(round(len(text)/divider))]



print('corpus length:', len(text))

chars = sorted(list(set(text)))
#print('chars',chars)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

#Build the new bilingual database
#
maxlen_line = max(max([len(x) for x in lines_text1]), max([len(x) for x in lines_text2]))
#print('minlen',minlen) F: 15, E:12

#Cut the text into slices, starting at the start of each verse
#PERFORMS TERRIBLY
"""
print('Slicing bilingual sentences')
maxlen = 11
step = 12
sentences = []
next_chars = []
for line in lines_text1:
    for i in range(0, len(line) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))
"""

"""
lines_text = np.zeros((len(lines_text1)+len(lines_text2),maxlen_line))
for i in range(0,len(lines_text1)):
    for j in range(0,maxlen_line):
        if j <= len(lines_text1[i]):
            lines_text[i][j] = lines_text1[i][j]
        """

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

#TAKE SLICE OF SENTENCES AND CHARACTERS FOR COMPUATABILITY!!



#Add language labels to vectorization
print('Vectorization...')
X = np.zeros((len(sentences), maxlen+1, len(chars)+2), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)+2), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t+1, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

#
print('Shape of X, y', X.shape) #(2789223, 40, 67)
#Set first words of the sentences to language tokens
for i,x_emp in enumerate(X):
    if i==0:
        print(x_emp)
    if(i< len(sentences1)):
        X[i,0,-1] = 1
    else:
        X[i,0,-2] = 1
print('First word of sentence', X[0:100,0,:])


print('example',X[500,0,60:69])
#print('example2',X[2789222,0,60:69])


print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen+1, len(chars)+2)))
model.add(Dense(len(chars)+2))
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
    start_index = random.choice(lines_index)
    #start_index = random.randint(0, len(text) - maxlen - 1)

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
                x = np.zeros((1, maxlen+1, len(chars)+2))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)

                #Ignore errors caused by index of added 2 in vector space
                if(next_index > len(chars) -1):
                    break

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
                # x = np.zeros((1, maxlen, len(chars)))
                x = np.zeros((1, maxlen+1, len(chars)+2))
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
        model.save('lstm_bible_gen.h5py')
    model.save('lstm_bible_gen_final.h5py')

