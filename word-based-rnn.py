#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers
print('tensorflow version:', tf.__version__)

import numpy as np
import time

class RNNBookTitles(tf.keras.Model):
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

class OneStep(tf.keras.Model):
  def __init__(self, model, words_from_ids, ids_from_words, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.words_from_ids = words_from_ids
    self.ids_from_ = ids_from_words

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_words(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_words.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_words = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_words(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_words = self.words_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_words, states

def build_model(epochs = 1):
    path_to_file = 'titles.txt'

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

    # The embedding dimension
    embedding_dim = 16

    # Number of RNN units
    rnn_units = 64

    model = RNNBookTitles (
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

    history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])

    one_step_model = OneStep(model, words_from_ids, ids_from_words)

    tf.saved_model.save(one_step_model, "rnn_book_titles")
    return one_step_model

build_model()
