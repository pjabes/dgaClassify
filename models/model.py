import random
from helpers import generate_char_dictionary as generate_char_dictionary
from helpers import tokenizeString as tokenizeString
from helpers import padToken as padToken
from helpers import build_model
from helpers import train as train

import pandas as pd
import numpy as np
import json 

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import model_from_json

import sklearn
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

from flask import Flask   
from flask import request

app = Flask(__name__)

first_run = True

@app.route('/api/build-model/<size>')
def my_route(size):
    size = int(size)

    if size < 1000:
        return('Model dataset too small.  Please increase size to at least 1000.')

    # Load Training Data
    benignDomains_df = pd.read_csv('./datasets/benign.csv', usecols=[1], names=['domain'])
    benignDomains_df['class'] = 0
    maliciousDomains_df = pd.read_csv('./datasets/dga.txt', sep='\t', usecols=[0], names=['domain'])
    maliciousDomains_df['class'] = 1

    # Randomise training sets and select a **sized** sample
    benignDomains_df = benignDomains_df.sample()
    maliciousDomains_df = maliciousDomains_df.sample()
    domains_df = pd.concat([benignDomains_df[0:size], maliciousDomains_df[0:size]], sort=False)

    # Train the model with the training data.
    model = train(domains_df)

    chars_dict = model[1]

    chars_dict = model[1]
    max_len = model[2]
    model = model[0]

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    with open("chars_dict.json", "w") as json_file:
        json_file.write(json.dumps(chars_dict))

    with open("max_length.json", "w") as json_file:
        json_file.write(str(max_len))

    model.save_weights("model.h5")

    return('Completed Building Model with size of: ' + str(size))


@app.route('/api/predict/<domain>')
def predict(domain):
    

    if first_run:
        # Load the Model
        loaded_model_json = open('model.json', 'r').read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("model.h5")

        maxlen = int(open('max_length.json', 'r').read())
        chars_dict = json.loads(open('chars_dict.json', 'r').read())

        domainToken = []
        
        for char in domain:
            domainToken.append(chars_dict[char])

        tokenList = sequence.pad_sequences([domainToken], maxlen)
        
        result = model.predict_classes(tokenList)[0]
        
        if result == 0:
            return domain + ' - ' + "benign"
        else:
            return domain + ' - ' + "malicious"

        first_run = False
    else:
        domainToken = []
        
        for char in domain:
            domainToken.append(chars_dict[char])

        tokenList = sequence.pad_sequences([domainToken], maxlen)
        
        result = model.predict_classes(tokenList)[0]
        
        if result == 0:
            return domain + ' - ' + "benign"
        else:
            return domain + ' - ' + "malicious"


if __name__ == "__main__":
    app.run()