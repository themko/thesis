#Standar (non-var) Sequence-to-Sequence encoder 
import numpy as np
import time
import pickle
import keras
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential,Model,load_model
from keras.layers import Input,Dense, LSTM, RepeatVector,Layer,TimeDistributed
from sklearn.preprocessing import LabelBinarizer

#Build Data
top_words = 200 #5000
index_from = 3
num_epochs = 20
batch_size=256
intermediate_dim = 64
max_review_length = 50 #500

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
print('Data loaded')


X_train =  sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
print('Sequences padded')
enc =LabelBinarizer()
enc.fit(range(top_words)) #Possible problem, imdb.load_data ? starts at index 1

X_enc_in = ([enc.transform(x).tolist() for x in X_train])
X_enc_out = ([enc.transform(x).tolist() for x in X_test])
print('One-hot encoded')
#save data
#data = [X_enc_in,X_enc_out]
#pickle.dump(data,open('data.p','wb'))

#Build model?
if(False):
	#Creating autoencoder
	x = Input(shape=(max_review_length,top_words),name='main_input')
	dense_in = TimeDistributed(Dense(16,name='dense_in'),name='time_dist_dense_in')(x)
	encoded = LSTM( intermediate_dim,input_shape=(intermediate_dim,max_review_length), 
	                activation='relu',name='lstm_encoded')(dense_in)
	#right value for k??
	k= max_review_length
	encoded_repeat = RepeatVector(k,name='encoded_repeat')(encoded)
	decoded = LSTM(16,name='lstm_decoded',return_sequences=True)(encoded_repeat)
	dense_out = TimeDistributed(Dense(top_words,activation='softmax',input_shape=(50,16)),name='time_dist_dense_out')(decoded)
	#dense_out = Dense(top_words,activation='softmax',input_shape=(50,16),name='time_dist_dense_out')(decoded)

	print(x)
	print(dense_in)
	print(encoded)
	print(encoded_repeat)
	print(decoded)
	print(dense_out)
	print('Compiling model')
	SS = Model(x,dense_out)

	S_encoder = Model(x,encoded)

	#Decoder used duped autoenoder layers (REPLACE IF ORIGINAL REPLACED)
	decoder_input = Input(shape=(intermediate_dim,))
	_encoded_repeat = RepeatVector(k,name='encoded_repeat')(decoder_input)
	_decoded = LSTM(16,name='lstm_decoded',return_sequences=True)(_encoded_repeat)
	_dense_out = TimeDistributed(Dense(top_words,activation='softmax',input_shape=(50,16)),name='time_dist_dense_out')(_decoded)
	#_dense_out = Dense(top_words,activation='softmax',input_shape=(50,16),name='time_dist_dense_out')(_decoded)


	S_decoder = Model(decoder_input,_dense_out)
	S_decoder_lstm = Model(decoder_input,_decoded)

	SS.compile(optimizer='adagrad',loss='categorical_crossentropy')
	print('Fitting model')
	SS.fit(X_enc_in,X_enc_in,
	        shuffle=True,
	        epochs=num_epochs,
	        batch_size=batch_size,
	        validation_data =(X_enc_out,X_enc_out))
	#Save model
	timestr = time.strftime("%Y%m%d")
	SS.save('S-to-S_1.h5py')
	S_encoder.save('S_enc_1.h5py')
	S_decoder.save('S_dec_1.h5py')

	"""
	SS.save('S-to-S_model_'+timestr+'.h5py')
	S_encoder.save('S_enc_'+timestr+'.h5py')
	S_decoder.save('S_dec_'+timestr+'.h5py')
	"""
else:
	#Load model
	SS = load_model('S-to-S_1.h5py')
	S_encoder = load_model('S_enc_1.h5py')
	S_decoder = load_model('S_dec_1.h5py')
	SS.summary()

def onehot_to_num(database):
	#2500,50,200
	sentences = []
	for sen in database:
		words = []
		for word in sen:
			w = word.tolist()
			num = w.index(max(w))
			words.append(num)
		sentences.append(words)
	return sentences

def translate_back(sentence):
    #Code from https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset#44635045
    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k:(v+index_from) for k,v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    id_to_word = {value:key for key,value in word_to_id.items()}
    return (' '.join(id_to_word[id] for id in sentence ))


#test interpolation on first 2 training data
enc_outputs = S_encoder.predict(X_enc_out)
print('enc_outputs',enc_outputs[0],enc_outputs[1])
enc_interpol = (enc_outputs[0]-enc_outputs[1])/2.
print('interpol', enc_interpol)

test_Ss  =S_decoder.predict(np.array([enc_interpol,enc_outputs[0],enc_outputs[1]]))
test_Ss_lstm = S_decoder_lstm.predict(np.array([enc_interpol,enc_outputs[0],enc_outputs[1]]))
print('decoded 1,2,interpol',test_Ss)
print('trans_to_nums', onehot_to_num(test_Ss))
print('transl_decoded', [translate_back(x) for x in onehot_to_num(test_Ss)])

test_out = SS.predict(X_enc_out)
test_out = onehot_to_num(test_out) #to.list !! flattened probably
print('test_out 0:10',len(test_out), len(test_out[0]),'#\'s:',test_out[0:10])
print('Translation [0:10]',[translate_back(x) for x in test_out[0:10]])
print('Lstm outputs: ',test_Ss_lstm[0:10])