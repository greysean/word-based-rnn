#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import tensorflow as tf
from tensorflow.keras import layers
from nltk import word_tokenize

# Constants

PATH = "./texts/"
EMBEDDING_DIM = 16
RNN_UNITS = 64
EPOCHS = 1

SEQ_LENGTH = 20
BATCH_SIZE = 64
BUFFER_SIZE = 10000 # might be too high for word processing?

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

class OneStep(tf.keras.Model):
    def __init__(self, model, words_from_ids, ids_from_words, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.words_from_ids = words_from_ids
        self.ids_from_words = ids_from_words

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_words(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_words.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

#    @tf.function
    def generate_one_step(self, word, states=None):
        # Create proper shape for model and convert to id
        word_tensor = tf.constant(word, shape=(1,1))
        ids = self.ids_from_words(word_tensor)

        # Predict and select last token prediction
        predicted_logits, states = self.model(inputs=ids, states=states, return_state=True)
        predicted_logits = predicted_logits[:, -1, :]

        # Sample based on logit probability
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Return the corresponsing word
        predicted_words = words_from_ids(predicted_ids)
        return predicted_words, states

    def generate_sentence(self, seed=['The'], length=10):
        next_word = tf.constant(seed, shape=(len(seed)))
        states = None
        result = [next_word]

        for i in range(length):
            next_word, states = self.generate_one_step(next_word, states)
            result.append(next_word)

        # join generated sequence and return as string
        result = tf.strings.reduce_join(result, separator=" ")
        return result.numpy().decode('utf-8')


def read_files():
    '''
    Arguments: filepath (string)
    Returns: array of strings
    '''
    docs = []
    path = PATH
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
   # e.g. one result was this: tf.Tensor([b'The opt operates us\xe2\x80\x94holy 2011\xe2\x80\x94 Collaboration facilities Logistic 1000 gained parallel'], shape=(1,), dtype=string)

   # tokenize document

    tokenized_doc = word_tokenize(doc)
    return tokenized_doc

def tokenize_documents(documents):
    return [word for doc in documents for word in clean_and_tokenize_doc(doc)]

docs = read_files()
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


one_step_model = OneStep(model, words_from_ids, ids_from_words)
result = one_step_model.generate_sentence()

print(result)
