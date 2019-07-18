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

def train(df):

    chars_dict = generate_char_dictionary(df)
    max_features = len(chars_dict) + 1

    df['tokenList'] = df.apply(lambda row: tokenizeString(row['domain'], chars_dict), axis=1)
    df['tokenListLen'] = df['tokenList'].apply(lambda x: len(x))

    # Calculate the largest length within the DataFrame
    maxlen = df['tokenListLen'].max()

    model = build_model(max_features, maxlen)

    # model.summary()

    test = df['tokenList'].values.tolist()


    tokenList = sequence.pad_sequences(test, maxlen)

    new_X = df['tokenList'].values

    new_Y = df['class'].values

    cb = []

    cb.append(EarlyStopping(monitor='val_loss', 
                            min_delta=0, #an absolute change of less than min_delta, will count as no improvement
                            patience=5, #number of epochs with no improvement after which training will be stopped
                            verbose=0, 
                            mode='auto', 
                            baseline=None, 
                            restore_best_weights=False))

    history = model.fit(x=tokenList, y=new_Y, 
                        batch_size=128, 
                        epochs=30, 
                        verbose=1, 
                        callbacks=cb, 
                        validation_split=0.2, #
                        validation_data=None, 
                        shuffle=True, 
                        class_weight=None, 
                        sample_weight=None, 
                        initial_epoch=0,
                        steps_per_epoch=None, 
                        validation_steps=None)
    
    return [model, chars_dict, maxlen]

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