# word-based-rnn

This program should train an rnn model on thousands of book titles to generate new titles with words as a unit. is is adapted from https://www.tensorflow.org/text/tutorials/text_generation.

## to do

I re-wrote the model with new texts in exp_model.py. To run:

```
source ~/.env/tensorflow/bin/activate
python exp_model.py.
```

note that the model is currently not being trained, but it doesn't need to be trained in order to get all of this functionality working.

### text processing considerations

- [ ] add code for text cleaning
  - see exp_model.py function ```clean_and_tokenize_doc()```

notes:

I thought about using keras' built-in Tokenizer but it doesn't allow for flexibility to e.g. use regex to filter tokens, which is needed to account for things like paragraphs, markup tags, etc.

I decided to use nltk because tokenizing is hard and they have libraries devoted to it

texts need to be cleaned according to decisions we make. I've included notes in exp_model.py to get us started; here are other thoughts:

  - lines beginning with # (i.e. metadata) need to be removed
  - decisions about how to tokenize quotes and other markup
    - do we want to keep quote tags?
    - we definitely want to remove formatting, e.g. ```<strong></strong>, <em></em>, <br/>```
    - newlines: on the one hand they are necessary for tracking paragraphs, but on the other it might be simpler to ignore them for now

### refactoring

  - [ ] turn generate_sentence / generate_one_step into a class
    - i tried, but something strange seems to happen to tensor objects depending on whether they are instantiated in a function or a class or not. I tried to debug some of it in notes, but i'm still confused.
  - [ ] convert result to a string
	- the documentation is wrong. it says: ```print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)``` but this doesn't work for me. Something about tf.Tensor object vs. tf.EagerTensor object.

