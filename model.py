#!/usr/bin/env python
# coding: utf-8

import load_vocab

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import time
import re

PATH = "./texts/"
EMBEDDING_DIM = 16
RNN_UNITS = 64
EPOCHS = 1

SEQ_LENGTH = 20
BATCH_SIZE = 64
BUFFER_SIZE = 10000 # might be too high for word processing?

class RNNTextModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.gru = layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = layers.Dense(vocab_size)

        #self.ids_from_words = ids_from_words
        #self.words_from_ids = words_from_ids

        #self.dataset = dataset

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
# Creates an RNNTextModel object with parameters specified by file constants
# Implicitly builds model using dataset variable taken from global scope, expects this to be defined.
def build_model():

    model = RNNTextModel(
        vocab_size = len(load_vocab.getTokenizer().ids_from_words.get_vocabulary()),
        embedding_dim = EMBEDDING_DIM,
        rnn_units = RNN_UNITS,
        )
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    model.build((64,100))
    #for input_example_batch, target_example_batch in dataset.take(1):
    #    example_batch_predictions = model(input_example_batch)
    return model


# Utility class for generating text with an RNNTextModel
class OneStep():
    def __init__(self, model, temperature=1.0):
        super().__init__()
        model.temperature = temperature # Could be bad form to change an object's properties from within a class that contains it.
        self.model = model

        tokenizer = load_vocab.getTokenizer()

        self.words_from_ids = tokenizer.words_from_ids
        self.ids_from_words = tokenizer.ids_from_words

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
        predicted_words = self.words_from_ids(predicted_ids)
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

CHECKPOINT_PATH = "./checkpoints/"

def latest_checkpoint():
    checkpoints = [f.name for f in os.scandir(CHECKPOINT_PATH)]
    if len(checkpoints)>0:
        # Extract checkpoint number as int and find max
        latest_checkpoint = max([int(re.search('ckpt_(\\d+)\\.hdf5', c).group(1)) for c in checkpoints])
    else:
        latest_checkpoint = 0

    return latest_checkpoint


def get_latest_model():
    checkpoint = latest_checkpoint()

    print('latest checkpoint found:',checkpoint)

    # Load model if saved version exists, else build.
    model = None
    try:
        print("Attempting to load model...")
        model = build_model()

        path = f"./checkpoints/ckpt_{checkpoint}.hdf5"
        print("Loading from", path)
        model.load_weights(path)
        print("Model exists! loading model..." )
    except Exception as e:
        print(e)
        print("Model does not exist! Building now...")
        model = build_model()

    return model

print(get_latest_model())
