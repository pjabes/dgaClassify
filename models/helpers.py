from keras.preprocessing import sequence

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

def padSequence(tokenList, max_len):

    return sequence.pad_sequences([tokenList], max_len)
