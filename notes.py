##d_take = dataset.take(1) 
##print(d_take) 
## this returns some weird DataTake object that I don't understand 
## I don't understand why it's different than 
## for input_example_batch, target_example_batch in dataset.take(1) 
#
#for input_example_batch, target_example_batch in dataset.take(1):
#    example_batch_predictions = model(input_example_batch)
#    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
#
## all of this works, whew 
##test = tf.constant([[1,2,3]])
##print(test) 
#
##test2 = model(test) 
##print(test2) 
##something i noticed when trying this is that the nested structure was necessary. let's see if that helps? 
#
#
## this works 
#word = tf.constant([["The"]])
#id_from_word = ids_from_words(word) 
#print(id_from_word)
#print(word.shape)

# this works fine, but it breaks later down the line at 
#word = tf.constant(['The']) 
#print(word.shape) 
#id = ids_from_words(word) 
#print(id) 

#this works here but not down the line 
#word = tf.constant(['The'])
#split_word = tf.strings.split(word) 
#id_from_word = ids_from_words(split_word) 
#print(id_from_word) 

#this also works!!!! 
#word_reconstruction = words_from_ids(id_from_word)
#print(word_reconstruction)

#pred = model(id_from_word) 
#pred = pred[:, -1, :]
#print(pred)
#
##I'm not sure why having it be nested is necessary?? the error is: 
## ValueError: Input 0 of layer gru is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (1, 16)
#
## nesting the word (i.e. tf.constant([["The]]) works fine... but WHY 
#
##pred_string = words_from_ids(pred) 
##print(pred_string) 
## This works, but it returns a bunch of unknowns?!?! 
##it's also still bytes, let's try to convert it to a string... 
## update: it's unknowns b/c you pass in teh sampled indices and not the logits lol 
#
#sampled_indices = tf.random.categorical(pred, num_samples=1) 
#print(sampled_indices) 
#sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy() 
#print(sampled_indices) 
#
## woo ok got it! 
#
## this below is not working. weird error. let's investiate! 
## word_reconstruction = words_from_ids(text_from_ids(sampled_indices).numpy())
#
## it's already numpy so wtf?!?!
##word_as_numpy = sampled_indices.numpy() 
##print(word_as_numpy) 
#
## lol once again i'm an idiot 
#word_reconstruction = text_from_ids(sampled_indices).numpy() 
#print(word_reconstruction) 
#
## woot! so this should enable me to build the class! 
#
## wait i have an idea about why this might not be working... in the char model it does a 'split_unicode'so maybe that's why? 
## update: yep! creating a string with just one nested word and then using tf.strings.split did the trick! 
## update2: it's true that this puts the data in a nested format but apparently the raggedtensor format doesn't work...

########### ABOVE GOT MESSY SO STARTING OVER ###############

#print("word1") 
#word = tf.constant(["The"])
#id = ids_from_words(word) 
#print(word.shape)
#print(id.shape) 
#pred = model(id) #- doesn't work
#
#
#print("word2") 
#word2 = tf.constant([['The']])
#id2 = ids_from_words(word2) 
#print(word2.shape) 
#print(id2.shape) 
#pred = model(id2) # works 
#
#print("word3") 
#word3 = tf.constant(['The'], shape=(1,1))
#id3 = ids_from_words(word3) 
#print(word3.shape) 
#print(id3.shape) 
#pred = model(id3) # works??  
#

#states = None
#seed = ['The']
#word4 = tf.constant(seed, shape=(1,1))
#id4 = ids_from_words(word4) 
#pred, states = model(id4, states, return_state=True) # works
#print(pred, states) 
# the only difference i can see is that this is an eager tensor and the other is just tensor. 
#print("word5") 
#seed = ['The', 'gun']
#word5 = tf.constant(seed, shape=(len(seed),1))
#id5 = ids_from_words(word5) 
#print(word5.shape) 
#print(id5.shape) 
#pred = model(id5) #this worked, but returned two words? I really didn't expect that 

#pred = pred[:, -1, :]
#
#sampled_indices = tf.random.categorical(pred, num_samples=1) 
#sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy() 
#
#word_reconstruction = text_from_ids(sampled_indices).numpy() 
#print(word_reconstruction) 

#class OneStep(tf.keras.Model):
#    def __init__(self, model, words_from_ids, ids_from_words, temperature=1.0):
#        super().__init__()
#        self.temperature = temperature
#        self.model = model
#        self.words_from_ids = words_from_ids
#        self.ids_from_words = ids_from_words
#
##    @tf.function
#    def generate_one_step(self, word, states=None): 
#        id = self.ids_from_words(word)
#        print(word.shape) 
#        print(id.shape) 
#        print(type(id)) 
#        predicted_logits, states = self.model(inputs=id, states=states, return_state=true) 
#        print(predicted_logits, states) 
#        predicted_logits = predicted_logits[:, -1, :] 
#        print(predicted_logits) 
#        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1) 
#        predicted_ids = tf.squeeze(predicted_ids, axis=-1) 
#
#        return predicted_words, states
#
#one_step = OneStep(model, model.words_from_ids, model.ids_from_words)
#
#word = tf.constant(['The'], shape=(1,1) )
#result = one_step.generate_one_step(word)
#print(result) 
