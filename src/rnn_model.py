import numpy as np
import time
import keras
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential,Model,load_model
from keras.layers import Input,Dense, LSTM, RepeatVector,Layer,TimeDistributed
from sklearn.preprocessing import LabelBinarizer