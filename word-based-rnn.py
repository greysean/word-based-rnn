#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import os
import time

path_to_file = 'titles.txt'

print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with open(path_to_file, 'r') as f: 
    text = f.readlines()

# strip newlines, remove empty strings, add <EOS> at end of each string
text = [line.strip() for line in text if len(line) > 0]
text = " ".join([s + " <EOS>" for s in text])

vocab = sorted(set(text.split(" ")))

# helper functions 

## generate ids from words 
ids_from_words = layers.StringLookup(
     vocabulary=list(vocab), mask_token=None)

## generate words from ids 
words_from_ids = layers.StringLookup(
     vocabulary=ids_from_words.get_vocabulary(), invert=True, mask_token=None)

## generate text from ids 
def text_from_ids(ids):
    return tf.strings.reduce_join(words_from_ids(ids), axis=-1, separator=" ")

data = tf.strings.split(text, sep=" ") 

all_ids = ids_from_words(data)

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 20
# examples_per_epoch = len(text)//(seq_length+1)

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))


# Length of the vocabulary in chars
# vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 16

# Number of RNN units
rnn_units = 64

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
    
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

model = MyModel(
    # Be sure the vocabulary size matches the `StringLookup` layers.
    vocab_size=len(ids_from_words.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

print(model.summary())


loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 1

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

