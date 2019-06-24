import random
from helpers import generate_char_dictionary as generate_char_dictionary
from helpers import tokenizeString as tokenizeString
from helpers import padToken as padToken
from helpers import build_model

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
        

def predict(domain, model, chars_dict, maxlen):
    """Predict whether a given domain is malicious or benign."""

    domainToken = []
    
    for char in domain:
        domainToken.append(chars_dict[char])

    tokenList = sequence.pad_sequences([domainToken], maxlen)
    
    result = model.predict_classes(tokenList)[0]
    
    if result == 0:
        return domain + ' - ' + "benign"
    else:
        return domain + ' - ' + "malicious"



# Ingesting Benign Domains
benignDomains_df = pd.read_csv('./datasets/CISCO-top-1m.csv')
benignDomains_df.rename(columns={'netflix.com': 'domain'}, inplace=True)
benignDomains_df['class'] = 0
benignDomains_df['src'] = 'cisco'
print(benignDomains_df.head())
# # Ingesting Malicious Domains
maliciousDomains_df = pd.read_csv('./datasets/dga-domains.txt', sep='\t', usecols=[0,1], names=['src', 'domain'])
maliciousDomains_df['class'] = 1

# # Merging Datasets
domains_df = pd.concat([maliciousDomains_df[0:1000], benignDomains_df[0:1000]], sort=False)
domains_df = domains_df.sample(frac=0.5)

model = train(domains_df)
print(predict('google.com', model[0], model[1], model[2]))

def hello_world():
    print('helo world')