from __future__ import absolute_import

from keras import backend as K
import numpy as np
from keras.utils.np_utils import accuracy
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, merge
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.objectives import mse

import global_settings as G
from sentences_generator import Sentences
import vocab_generator as V_gen

k = G.window_size # context windows size
context_size = 2*k

# Creating a sentence generator from demo file
sentences = Sentences("test_file.txt")
vocabulary = dict()
V_gen.build_vocabulary(vocabulary, sentences)
V_gen.filter_vocabulary_based_on(vocabulary, G.min_count)
reverse_vocabulary = V_gen.generate_inverse_vocabulary_lookup(vocabulary, "vocab.txt")

# generate embedding matrix with all values between -1/2d, 1/2d
embedding = np.random.uniform(-1.0/2.0/G.embedding_dimension, 1.0/2.0/G.embedding_dimension, (G.vocab_size+3, G.embedding_dimension))

# Creating CBOW model
word_index = Input(shape=(1,))
context = Input(shape=(context_size,))
negative_samples = Input(shape=(G.negative,))
shared_embedding_layer = Embedding(input_dim=(G.vocab_size+3), output_dim=G.embedding_dimension, weights=[embedding])

word_embedding = shared_embedding_layer(word_index)
context_embeddings = shared_embedding_layer(context)
negative_words_embedding = shared_embedding_layer(negative_samples)
cbow = Lambda(lambda x: K.mean(x, axis=1), output_shape=(G.embedding_dimension,))(context_embeddings)

word_context_product = merge([word_embedding, cbow], mode='dot')
negative_context_product = merge([negative_words_embedding, cbow], mode='dot', concat_axis=-1)

model = Model(input=[word_index, context, negative_samples], output=[word_context_product, negative_context_product])

model.compile(optimizer='rmsprop', loss='binary_crossentropy')
print model.summary()

model.fit_generator(V_gen.pretraining_batch_generator(sentences, vocabulary, reverse_vocabulary), samples_per_epoch=G.train_words, nb_epoch=1)

# input_context = np.random.randint(10, size=(1, context_size))
# input_word = np.random.randint(10, size=(1,))
# input_negative = np.random.randint(10, size=(1, G.negative))

# print "word, context, negative samples"
# print input_word.shape, input_word
# print input_context.shape, input_context
# print input_negative.shape, input_negative

# output_dot_product, output_negative_product = model.predict([input_word, input_context, input_negative])
# print "word cbow dot product"
# print output_dot_product.shape, output_dot_product
# print "cbow negative dot product"
# print output_negative_product.shape, output_negative_product
