import process_text as pt
from tensorflow.keras import layers
import os
import numpy as np

TEXT_PATH = "./texts/"
DATA_PATH = "./data/"

class Tokenizer():
    def __init__(self, ids_from_words, words_from_ids, text_from_ids):
        self.ids_from_words = ids_from_words
        self.words_from_ids = words_from_ids
        self.text_from_ids = text_from_ids

def generateData():
    # Code
    tokens = pt.clean_and_tokenize_docs(TEXT_PATH)
    print('Total Tokens: %d' % len(tokens))
    print('Unique Tokens: %d' % len(set(tokens)))

    vocab = sorted(set(tokens))

    ## generate ids from words
    ids_from_words = layers.StringLookup(vocabulary=list(vocab), mask_token=None)

    ## generate words from ids
    words_from_ids = layers.StringLookup(vocabulary=ids_from_words.get_vocabulary(),
            invert=True)

    ## generate text from ids
    def text_from_ids(ids):
            return tf.strings.reduce_join(words_from_ids(ids), axis=-1, separator=" ")

    all_ids = ids_from_words(tokens)
    all_ids_string = [str(x) for x in np.array(all_ids)]

    os.mkdir(DATA_PATH)
    with open(DATA_PATH+"vocab.txt","w") as fv, open(DATA_PATH+"tokens.txt","w") as ft:
        fv.write(' '.join(vocab))
        fv.close()
        ft.write(' '.join(all_ids_string))
        ft.close()

def loadVocab() :
    with open(DATA_PATH+"vocab.txt","r") as f :
        vocab = sorted(set(f.read().split(' ')))
        f.close()
    return vocab

def loadTokens() :
    with open(DATA_PATH+"tokens.txt") as f:
        tokenString = f.read()
        # Split by space and convert to token int
        all_ids = [int(x) for x in tokenString.split(' ')]
        f.close()
    return all_ids

# Exported

def getTokenizer():
    vocab = getVocab()

    ## generate ids from words
    ids_from_words = layers.StringLookup(vocabulary=list(vocab), mask_token=None)

    ## generate words from ids
    words_from_ids = layers.StringLookup(vocabulary=ids_from_words.get_vocabulary(),
            invert=True)

    ## generate text from ids
    def text_from_ids(ids):
            return tf.strings.reduce_join(words_from_ids(ids), axis=-1, separator=" ")

    return Tokenizer(ids_from_words,words_from_ids, text_from_ids)

def getVocab():
    if(not os.path.exists(DATA_PATH+"vocab.txt")) :
        generateData()

    return loadVocab()

def getTokens():
    if(not os.path.exists(DATA_PATH+"tokens.txt")) :
        generateData()

    return loadTokens()

#print(getTokenizer().ids_from_words(['the']))
