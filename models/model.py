import random

import pandas as pd
import numpy as np
import validators

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import sklearn
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

from helpers import tokenizeString as tokenizeString

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


@app.route('/api/build-model')
def build_model_route():
    """API route to build and train a model object and save to disk

    Arguments:
        size {integer} -- represents the size of training datasets.  Symmetric between benign and malicious classes.

    Returns:
        string -- returns string with the result of the model construction. 
    """

    K.clear_session()

    # Parameters
    size = request.args.get('size', default=5000, type=int)
    patience = request.args.get('patience', default=5, type=int)
    epochs = request.args.get('epochs', default=10, type=int)

    if size < 1000:
        return('Model datasets are too small.  Suggested size is at least 1000.')
    
    if epochs < 5:
        return('Model epochs is too small.  Suggested size is at least 30.')

    # Load Training Data
    # TODO: Ensure that input data is appropriately sanitized as per RFC 3986
    benignDomains_df = pd.read_csv('./datasets/benign.csv', usecols=[0])
    benignDomains_df['domain'] = benignDomains_df['domain'].str.lower()
    benignDomains_df['class'] = 0

    maliciousDomains_df = pd.read_csv('./datasets/dga.csv', usecols=[0])
    maliciousDomains_df['domain'] = maliciousDomains_df['domain'].str.lower()
    maliciousDomains_df['class'] = 1

    # Randomise training sets and select a **sized** sample
    benignDomains_df = benignDomains_df.sample(frac=0.5)
    maliciousDomains_df = maliciousDomains_df.sample(frac=0.5)
    domains_df = pd.concat([benignDomains_df[0:size], maliciousDomains_df[0:size]], sort=False)

    # Character Dictionary based upon RFC3986
    max_features = 40

    # Training the Model
    domains_df['tokenList'] = domains_df.apply(lambda row: tokenizeString(row['domain']), axis=1)
    domains_df['tokenListLen'] = domains_df['tokenList'].str.len()

    max_len = domains_df['tokenListLen'].max()
    domainsList = domains_df['tokenList'].values.tolist()
    tokenList = sequence.pad_sequences(domainsList, max_len)
    classList = domains_df['class'].values

    # Build Model
    model = Sequential()
    model.add(Embedding(40, 64, input_length=max_len))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['binary_crossentropy','acc'])

    cb = []
    cb.append(EarlyStopping(monitor='val_loss', 
                            min_delta=0, #an absolute change of less than min_delta, will count as no improvement
                            patience=patience, #number of epochs with no improvement after which training will be stopped
                            verbose=0, 
                            mode='auto', 
                            baseline=None, 
                            restore_best_weights=False))

    history = model.fit(x=tokenList, y=classList, 
                        batch_size=128, 
                        epochs=epochs, 
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
    

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    with open("max_length.json", "w") as json_file:
        json_file.write(str(max_len))

    model.save_weights("model.h5")

    with open("version.json", "w") as json_file:
        json_file.write(str(datetime.datetime.now()))

    return('Completed Building Model with size of: ' + str(size))


@app.route('/api/predict/<domain>')
def predict_route(domain):
    """API route to predict a domain based on the latest model object
    
    Arguments:
        domain {string} -- the suspicious domain represented as a string without protocol or www
    
    Returns:
        string -- results of the prediction or an error
    """


    if validators.domain(domain):

        K.clear_session()

        loaded_model_json = open('model.json', 'r').read()
        model = model_from_json(loaded_model_json)
        model.load_weights("model.h5")
        maxlen = int(open('max_length.json', 'r').read())
        version = str(open('version.json', 'r').read())
        
        tokenList = tokenizeString(domain)
        tokenList = sequence.pad_sequences([tokenList], maxlen)
        print(model.predict_classes(tokenList))
        result = model.predict_classes(tokenList)[0]
        
        if result == 0:
            return domain + ' - ' + "benign" + ' using model ' + version
        else:
            return domain + ' - ' + "malicious" + ' using model ' + version
            
    else:
        return(domain + ' is not a valid domain according to RFCs.  Please input a valid domain.')

if __name__ == "__main__":
    app.run(debug=True)