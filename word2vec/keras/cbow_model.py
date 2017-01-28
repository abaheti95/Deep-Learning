from __future__ import absolute_import

from keras import backend as K
import numpy as np
from keras.utils.np_utils import accuracy
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, merge
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.objectives import mse

k = 3 # context windows size
context_size = 2*k
neg = 5 # number of negative samples
# generate weight matrix for embeddings
embedding = []
for i in range(10):
	embedding.append(np.full(100, i))
embedding = np.array(embedding)
print embedding

# Creating CBOW model
word_index = Input(shape=(1,))
context = Input(shape=(context_size,))
negative_samples = Input(shape=(neg,))
shared_embedding_layer = Embedding(input_dim=10, output_dim=100, weights=[embedding])

word_embedding = shared_embedding_layer(word_index)
context_embeddings = shared_embedding_layer(context)
negative_words_embedding = shared_embedding_layer(negative_samples)
cbow = Lambda(lambda x: K.mean(x, axis=1), output_shape=(100,))(context_embeddings)

word_context_product = merge([word_embedding, cbow], mode='dot')
negative_context_product = merge([negative_words_embedding, cbow], mode='dot', concat_axis=-1)

model = Model(input=[word_index, context, negative_samples], output=[word_context_product, negative_context_product])

model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

input_context = np.random.randint(10, size=(1, context_size))
input_word = np.random.randint(10, size=(1,))
input_negative = np.random.randint(10, size=(1, neg))

print "word, context, negative samples"
print input_word.shape, input_word
print input_context.shape, input_context
print input_negative.shape, input_negative

output_dot_product, output_negative_product = model.predict([input_word, input_context, input_negative])
print "word cbow dot product"
print output_dot_product.shape, output_dot_product
print "cbow negative dot product"
print output_negative_product.shape, output_negative_product
