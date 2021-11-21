#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import tensorflow as tf
from tensorflow.keras import layers

class RNNTextModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, ids_from_words, words_from_ids):
        super().__init__(self)
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.gru = layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = layers.Dense(vocab_size)
        self.ids_from_words = ids_from_words
        self.words_from_ids = words_from_ids

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

def read_files():
    '''
    Arguments: filepath (string)
    Returns: array of strings
    '''
    docs = []
    path = './texts/'
    filelist = os.listdir(path)
    for f in filelist:
        file = open(os.path.join(path + f), 'r')
        docs.append(file.read())
    return docs

def clean_and_tokenize_doc(doc):
    '''
    tokenizes a single document it according to the decisions we make

    arguments: doc (string)
    returns: array of strings

   '''
   # remove lines that begin with markup (e.g., metadata)
   # this will require splitting the doc by lines, possibly by paragraphs?

   # filter formatting tags (e.g., <strong></strong>) but not <quote></quote>

   # normalize quotes (make same ascii character)

   # handle special characters

   # tokenize document

    tokenized_doc = word_tokenize(doc)
    return tokenized_doc

def tokenize_documents(documents):
    return [word for doc in documents for word in clean_doc(doc)]

docs = read_text()
tokens = tokenize_documents(docs)

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
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

# Constants (to be moved to top)
SEQ_LENGTH = 20
BATCH_SIZE = 64
BUFFER_SIZE = 10000 # might be too high for word processing?

EMBEDDING_DIM = 16
RNN_UNITS = 64
EPOCHS = 1

def split_input_target(sequence):
        return sequence[:-1], sequence[1:]

sequences = ids_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)
dataset = sequences.map(split_input_target)

# https://stackoverflow.com/questions/41175401/what-is-a-batch-in-tensorflow
dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        )

model = RNNTextModel(
        vocab_size = len(ids_from_words.get_vocabulary()),
        embedding_dim = EMBEDDING_DIM,
        rnn_units = RNN_UNITS,
        ids_from_words = ids_from_words,
        words_from_ids = words_from_ids
        )

def generate_one_step(word, states):
    word_tensor = tf.constant(word, shape=(1,1))
    id = ids_from_words(word_tensor)
    pred, states = model(id, states, return_state=True)
    pred = pred[:, -1, :]

    sampled_indices = tf.random.categorical(pred, num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

    next_word = text_from_ids(sampled_indices)
    return next_word, states

def generate_sentence(seed=['The'], length=10):
    next_word = tf.constant(seed, shape=(len(seed), 1))
    states = None
    result = [next_word]

    for i in range(length):
        print('The %s next_word is %s with shape %s' % (i, next_word, next_word.shape))
        print('Result: %s' % (result))
        print('states: %s' % (states))
        next_word, states = generate_one_step(next_word, states)
        result.append(next_word)
    result = tf.strings.join(result, " ")[0]
    return result


result = generate_sentence()


