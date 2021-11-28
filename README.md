# word-based-rnn

This program should train an rnn model on thousands of book titles to generate new titles with words as a unit. is is adapted from https://www.tensorflow.org/text/tutorials/text_generation.

## to do

To run:

```
source ~/.env/tensorflow/bin/activate
python main.py.
```

### cannot generate text from loading saved model
I cannot for the life of me figure out how to make this work.

### text processing considerations

- [X] add code for text cleaning: DONE
  - remove metadata markup
  - remove formatting tags
  - remove muse links
  - remove footnotes
  - replace non-ascii smartquotes with plain
  -
#### tokenizer

- tokenizing is hard. keras' built-in Tokenizer doesn't have enough flexibility. doesn't allow for flexibility to e.g. use regex to filter tokens. nltk has a great library.

#### headers

a lot of french philosophy texts have headers that break the text into numbered or named sections. i think we should try to keep this format since it's so distinct to the style, and i could see it easily fitting into a tweet.

#### tags

- We should remove formatting tags, e.g. ``<strong></strong> <em></em> <br/>``, but keep quotes and verses.

#### paragraphs
at some point it might make sense to keep track of these. ignoring them for now.

#### special characters

- to see all unique tags use ```grep -rohi "<[^>]*>" texts/ | sort --unique```
- to see all non-ascii characters use ``grep -rohi -P '[^\x00-\x7F]' texts | sort --unique ``

    - newlines: on the one hand they are necessary for tracking paragraphs, but on the other it might be simpler to ignore them for now

### refactoring

  - [ ] turn generate_sentence / generate_one_step into a class
    - i tried, but something strange seems to happen to tensor objects depending on whether they are instantiated in a function or a class or not. I tried to debug some of it in notes, but i'm still confused.
  - [ ] convert result to a string
	- the documentation is wrong. it says: ```print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)``` but this doesn't work for me. Something about tf.Tensor object vs. tf.EagerTensor object.

