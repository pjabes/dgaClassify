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

def tokenizeString(domain):
    """Neural Networks require data to be tokenized as integers to work."""
    chars_dict = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "0": 10,
                  "a": 11, "b": 12, "c": 13, "d": 14, "e": 15, "f": 16, "g": 17, "h": 18, "i": 19,
                  "j": 20, "k": 21, "l": 22, "m": 23, "n": 24, "o": 25, "p": 26, "q": 27, "r": 28,
                  "s": 29, "t": 30, "u": 31, "v": 32, "w": 33, "x": 34, "y": 35, "z": 36, "-": 37,
                  "_": 38, ".": 39, "~": 40}


    tokenList = []

    for char in domain:
        tokenList.append(chars_dict[char])
        
    return tokenList

def padToken(dataset, maxlen):

    newList = [0] * maxlen

    return newList + dataset

