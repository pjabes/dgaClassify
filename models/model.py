import random

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

# from helpers import generate_char_dictionary as generate_char_dictionary
# from helpers import tokenizeString as tokenizeString
# from helpers import padToken as padToken
# from helpers import build_model
# from helpers import train as train

import pandas as pd
import numpy as np
import json 
import datetime

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import model_from_json
from keras import backend as K

import sklearn
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

from flask import Flask   
from flask import request

app = Flask(__name__)


global model
global maxlen
global chars_dict


first_run = True
maxlen = None
chars_dict = {}


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

@app.route('/api/build-model/<size>')
def build_model_route(size):

    K.clear_session()

    size = int(size)

    if size < 1000:
        return('Model dataset too small.  Please increase size to at least 1000.')

    # Load Training Data
    benignDomains_df = pd.read_csv('./datasets/benign.csv', usecols=[1], names=['domain'])
    benignDomains_df['class'] = 0
    maliciousDomains_df = pd.read_csv('./datasets/dga.txt', sep='\t', usecols=[0], names=['domain'])
    maliciousDomains_df['class'] = 1

    # Randomise training sets and select a **sized** sample
    benignDomains_df = benignDomains_df.sample(frac=0.25)
    print(benignDomains_df)
    maliciousDomains_df = maliciousDomains_df.sample(frac=0.25)
    domains_df = pd.concat([benignDomains_df[0:size], maliciousDomains_df[0:size]], sort=False)

    df = domains_df

    # Train the model with the training data.
    chars_dict = generate_char_dictionary(df)
    max_features = len(chars_dict) + 1

    df['tokenList'] = df.apply(lambda row: tokenizeString(row['domain'], chars_dict), axis=1)
    df['tokenListLen'] = df['tokenList'].apply(lambda x: len(x))

    # Calculate the largest length within the DataFrame
    maxlen = df['tokenListLen'].max()

    model = build_model(max_features, maxlen)

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
                        epochs=50, 
                        verbose=1, 
                        callbacks=cb, 
                        validation_split=0.4, #
                        validation_data=None, 
                        shuffle=True,   
                        class_weight=None, 
                        sample_weight=None, 
                        initial_epoch=0,
                        steps_per_epoch=None, 
                        validation_steps=None)










    max_len = maxlen

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    with open("chars_dict.json", "w") as json_file:
        json_file.write(json.dumps(chars_dict))

    with open("max_length.json", "w") as json_file:
        json_file.write(str(max_len))

    model.save_weights("model.h5")

    with open("version.json", "w") as json_file:
        json_file.write(str(datetime.datetime.now()))

    return('Completed Building Model with size of: ' + str(size))


@app.route('/api/predict/<domain>')
def predict(domain):

    K.clear_session()

    loaded_model_json = open('model.json', 'r').read()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")

    maxlen = int(open('max_length.json', 'r').read())
    chars_dict = json.loads(open('chars_dict.json', 'r').read())

    version = str(open('version.json', 'r').read())
    domainToken = []
    
    for char in domain:
        domainToken.append(chars_dict[char])

    tokenList = sequence.pad_sequences([domainToken], maxlen)
    
    result = model.predict_classes(tokenList)[0]
    
    if result == 0:
        return domain + ' - ' + "benign" + ' using model ' + version
    else:
        return domain + ' - ' + "malicious" + ' using model ' + version

if __name__ == "__main__":
    app.run()