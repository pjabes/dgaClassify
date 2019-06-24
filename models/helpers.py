import pandas as pd
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import sklearn
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


def generate_char_dictionary(dataset):
    """Determines the set of characters within the data """
    
    chars_dict = dict() # Create an empty dict
    unique_chars = enumerate(set(''.join(str(dataset))))
    for i, x in unique_chars:
        chars_dict[x] = i + 1
    
    return chars_dict

def tokenizeString(domain, chars_dict):
    """Neural Networks require data to be tokenized as integers to work."""

    tokenList = []

    for char in domain:
        tokenList.append(chars_dict[char])
  
    return tokenList

def padToken(dataset, maxlen):

    newList = [0] * maxlen
    return newList + dataset

def build_model(max_features_num, maxlen):
    """Build LSTM model"""
    model = Sequential()
    model.add(Embedding(max_features_num, 64, input_length=maxlen))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_crossentropy','acc'])

    return model