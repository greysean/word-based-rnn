#!/usr/bin/env python
# coding: utf-8

import load_vocab
import model as load_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import tensorflow as tf
from tensorflow.keras import layers
import process_text as pt
import time
import math

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

def split_input_target(sequence):
        return sequence[:-1], sequence[1:]


# split into train, validation, test
def get_dataset_partitions(ds, train_split=0.8, val_split=0.1, test_split=0.1, batch_size=64, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    ds_size = dataset.cardinality().numpy()
    print("dataset size: {}".format(ds_size))

    # Specify seed to always have the same split distribution between runs
    ds = ds.shuffle(shuffle_size, seed=666)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    # Make these sizes multiples of batch size to minimize the data dropped
    train_size = math.ceil(train_size / batch_size) * batch_size
    val_size = math.ceil(val_size / batch_size) * batch_size

    train_ds = ds.take(train_size).batch(batch_size)
    val_ds = ds.skip(train_size).take(val_size).batch(batch_size)
    test_ds = ds.skip(train_size + val_size).take(ds_size-train_size-val_size).batch(batch_size, drop_remainder = True)

    return train_ds, val_ds, test_ds


# Code
all_ids = load_vocab.getTokens()
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
train, validate, test = get_dataset_partitions(dataset)

print("train.take: {}".format(train.take(1)))

model, latest_checkpoint = load_model.get_latest_model()
# Train and save model
checkpoint_dir = load_model.CHECKPOINT_PATH
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.hdf5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only = True
    )

history = model.fit(
        train,
        validation_data = validate,
        epochs=EPOCHS + latest_checkpoint,
        callbacks=[checkpoint_callback],
        initial_epoch = latest_checkpoint)
model.save_weights('./saved/rnn')
