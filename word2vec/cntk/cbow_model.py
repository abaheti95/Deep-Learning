from __future__ import print_function
import numpy as np
import cntk as C
from cntk.learner import sgd, learning_rate_schedule, UnitType
from cntk.utils import ProgressPrinter
from cntk.layers import Embedding, GlobalAveragePooling
from cntk.models import Sequential
from cntk.initializer import uniform

import global_settings as G

context_size = G.window_size * 2
##################################################
################### Inputs #######################
##################################################
word_one_hot = C.input_variable((G.vocab_size), np.float32)
context_one_hots = C.input_variable((context_size, G.vocab_size), np.float32)
negative_one_hots = C.input_variable((G.negative, G.vocab_size), np.float32)
 
# shared_embedding_layer = Embedding(G.embedding_dimension, uniform(scale=1.0/2.0/G.embedding_dimension))
shared_embedding_layer = Embedding(G.embedding_dimension)

word_embedding = shared_embedding_layer(word_one_hot)
context_embeddings = shared_embedding_layer(context_one_hots)
negative_embeddings = shared_embedding_layer(negative_one_hots)

print(word_embedding.shape)
word_embedding_reshaped = C.reshape(word_embedding, shape=(1, G.embedding_dimension))
print(word_embedding_reshaped.shape)
print(context_embeddings.shape)
print(negative_embeddings.shape)

cbow = C.reshape(C.reduce_mean(context_embeddings, 0), shape=(G.embedding_dimension))
print(cbow.shape)

# word_context_product = C.times_transpose(word_embedding_reshaped, cbow)
word_context_product = C.times_transpose(word_embedding, cbow)
print(word_context_product.shape)
negative_context_product = C.times_transpose(negative_embeddings, cbow)
print(negative_context_product.shape)
