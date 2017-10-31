#Standar (non-var) Sequence-to-Sequence encoder 
import numpy as np
import pickle
import keras
from keras.preprocessing import sequence
from keras.datasets import imdb
from sklearn.preprocessing import LabelBinarizer
from keras.datasets import imdb


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

#X_train =  sequence.pad_sequences(X_train, maxlen=max_review_length)
#X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)



print('translate 100')
imdb_list_train = [translate_back(x,id_to_word) for x in X_train[0:5000]]
print('writing file')
thefile = open('imdb_train.txt', 'w')
for item in imdb_list_train:
  thefile.write("%s\n" % item.encode('utf8'))
#enc =LabelBinarizer()
#enc.fit(range(top_words +1))
#enc.fit(range(top_words))

#X_enc_in = list([enc.transform(x).tolist() for x in X_train])
#X_enc_out = list([enc.transform(x) for x in X_test])
#print('x_enc',X_enc_in[13])

#data = [[1,2],[3,4]]
#pickle.dump(data,open('test.p','wb'))

