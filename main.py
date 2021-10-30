import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time

import tensorflow as tf
from tensorflow.keras import layers
print('tensorflow version:', tf.__version__)

from model import build_model

try: 
    one_step_model = tf.saved_model.load("./saved_models/rnn_book_titles")
    print("Model exists! loading model..." )
except: 
    print("Model does not exist! Building now...")
    one_step_model = build_model()

start = time.time()
states = None
next_word = tf.constant(['The'])
result = [next_word]

for n in range(1000):
  next_word, states = one_step_model.generate_one_step(next_word, states=states)
  result.append(next_word)

result = tf.strings.join(result, separator = " ")
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)
