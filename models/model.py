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

from helpers import generate_char_dictionary as generate_char_dictionary
from helpers import tokenizeString as tokenizeString
from helpers import padToken as padToken
from helpers import build_model
from helpers import train as train

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


@app.route('/api/build-model/<size>')
def build_model_route(size):

    K.clear_session()

    size = int(size)

    if size < 1000:
        return('Model datasets are too small.  Suggested size is 1000.')
    
    # Load Training Data
    benignDomains_df = pd.read_csv('./datasets/benign.csv', usecols=[1], names=['domain'])
    benignDomains_df['class'] = 0
    maliciousDomains_df = pd.read_csv('./datasets/dga.txt', sep='\t', usecols=[0], names=['domain'])
    maliciousDomains_df['class'] = 1

    # Randomise training sets and select a **sized** sample
    benignDomains_df = benignDomains_df.sample(frac=1)
    maliciousDomains_df = maliciousDomains_df.sample(frac=1)
    domains_df = pd.concat([benignDomains_df[0:size], maliciousDomains_df[0:size]], sort=False)
    print(domains_df.size)

    model = train(domains_df)

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

    with open("version.json", "w") as json_file:
        json_file.write(str(datetime.datetime.now()))

    return('Completed Building Model with size of: ' + str(size))


@app.route('/api/predict/<domain>')
def predict_route(domain):

    if validators.domain(domain):

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
        
        K.clear_session()

        if result == 0:
            return domain + ' - ' + "benign" + ' using model ' + version
        else:
            return domain + ' - ' + "malicious" + ' using model ' + version
            
    else:
        return(domain + ' is not a valid domain according to RFCs.  Please input a valid domain.')

if __name__ == "__main__":
    print(validators.domain('google.com'))
    app.run()