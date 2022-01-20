#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import tensorflow as tf
from tensorflow.keras import layers
import process_text as pt
import time

# TODO timestamp checkpoints / saved weights
now = time.strftime("%Y-%m-%d")

# Constants

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
        self.ids_from_words = ids_from_words
        self.words_from_ids = words_from_ids
        self.dataset = dataset

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
        vocab_size = len(ids_from_words.get_vocabulary()),
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
        self.words_from_ids = words_from_ids
        self.ids_from_words = ids_from_words

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

def split_input_target(sequence):
        return sequence[:-1], sequence[1:]


# Code

tokens = pt.clean_and_tokenize_docs(PATH)
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

#Generate training data
sequences = ids_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)
dataset = sequences.map(split_input_target)
# https://stackoverflow.com/questions/41175401/what-is-a-batch-in-tensorflow
#dataset = (
#        dataset
#        .shuffle(BUFFER_SIZE, seed=666)
#        .batch(BATCH_SIZE)
#        )

# split into train, validation, test
def get_dataset_partitions(ds, train_split=0.8, val_split=0.1, test_split=0.1, batch_size=64, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    ds_size = dataset.cardinality().numpy()
    print("dataset size: {}".format(ds_size))

    # Specify seed to always have the same split distribution between runs
    ds = ds.shuffle(shuffle_size, seed=666)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size).batch(BATCH_SIZE)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

train, validate, test = get_dataset_partitions(dataset)

print("train.take: {}".format(train.take(1)))

# Find latest checkpoint file
checkpoints = [f.name for f in os.scandir('./checkpoints')]
if len(checkpoints)>0:
    # Extract checkpoint number as int and find max
    latest_checkpoint = max([int(print(c) or re.search('ckpt_(\\d+)\\.hdf5', c).group(1)) for c in checkpoints])
else:
    latest_checkpoint = 0

print('latest checkpoint found:',latest_checkpoint)


# Load model if saved version exists, else build.
model = None
try:
    print("Attempting to load model...")
    model = build_model()

    path = f"./checkpoints/ckpt_{latest_checkpoint}.hdf5"
    print("Loading from", path)
    model.load_weights(path)
    print("Model exists! loading model..." )
except Exception as e:
    print(e)
    print("Model does not exist! Building now...")
    model = build_model()
    latest_checkpoint = 0

# Train and save model
checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.hdf5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only = True
    )

history = model.fit(
        train,
        #validation_data = validate, # Not working
        epochs=EPOCHS + latest_checkpoint,
        callbacks=[checkpoint_callback],
        initial_epoch = latest_checkpoint)
model.save_weights('./saved/rnn')

one_step_model = OneStep(model)
result = one_step_model.generate_sentence()

print(result)
