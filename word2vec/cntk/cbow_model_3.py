from __future__ import print_function
import os
import codecs
import numpy as np
import cntk as C
from cntk import Trainer
from _cntk_py import set_computation_network_trace_level
set_computation_network_trace_level(1)

from cntk.learner import adam_sgd, UnitType, sgd, learning_rate_schedule, momentum_as_time_constant_schedule
from cntk.utils import ProgressPrinter
from cntk.layers import Embedding, GlobalAveragePooling
from cntk.models import Sequential
from cntk.initializer import uniform
from cntk.io import CTFDeserializer, MinibatchSource, INFINITELY_REPEAT, FULL_DATA_SWEEP

import global_settings as G
# from sentences_generator import Sentences
import LID_sentence_generator as Sentences
import vocab_generator as V_gen
import save_embeddings as S
import input_to_CTF as CTF_gen

import time

start = time.time()

k = G.window_size # context windows size
context_size = 2*k

# Creating a sentence generator from demo file
sentences = Sentences.get_sentence_generator()
# sentences = Sentences("test_file.txt")


vocabulary = dict()
def load_vocabulary_and_reverse_vocabulary_from_file(filename):
	reverse_vocabulary = dict()
	vocabulary = dict()
	total_words = 0
	with codecs.open(filename, "r", "utf-8") as rf:
		idx = 1
		for line in rf:
			if not line:
				continue
			word, freq = line.split()[0], int(line.split()[1])
			vocabulary[word] = freq
			total_words += freq
			reverse_vocabulary[word] = idx
			idx += 1
	G.train_words = total_words
	return vocabulary, reverse_vocabulary

if not os.path.isfile(G.vocab_file) or G.regen_vocab:
	print("Generating vocabulary")
	# generate the vocabular and save in the vocab file
	V_gen.build_vocabulary(vocabulary, sentences)
	V_gen.filter_vocabulary_based_on(vocabulary, G.min_count)
	reverse_vocabulary = V_gen.generate_inverse_vocabulary_lookup(vocabulary, G.vocab_file)
else:
	print("Loading vocabulary from file:", G.vocab_file)
	# if vocab file already present. Load the vocabulary from that file
	vocabulary, reverse_vocabulary = load_vocabulary_and_reverse_vocabulary_from_file(G.vocab_file)

# 2 buffer vectors are kept for vocabulary
G.embedding_vocab_size = G.vocab_size + 2

end = time.time()
print("Time taken for preprocessing %.2fsecs" % (end - start))

total_training_instances = 0
if not os.path.isfile(G.CTF_input_file) or G.regen_input_file:
	print("Generating CTF input:", G.CTF_input_file)
	start = time.time()
	# if the file doesn't exists then generate the CTF input file
	total_training_instances = CTF_gen.create_CTF_format_input(sentences, vocabulary, reverse_vocabulary)
	end = time.time()
	print("Time taken for generating CTF format input file = %.2fsecs" % (end - start))
else:
	# Load the total number of training instances from the config file
	with open(G.CTF_config_file, "r") as rf:
		total_training_instances = int(next(rf))

# def cntk_pretraining_batch_generator(sentences, vocabulary, reverse_vocabulary):
# 	inputs, labels = pretraining_batch_generator(sentences, vocabulary, reverse_vocabulary)
# 	yield inputs

def create_word2vec_cbow_model(word_one_hot, context_one_hots, negative_one_hots):
	# shared_embedding_layer = Embedding(G.embedding_dimension, uniform(scale=1.0/2.0/G.embedding_dimension))
	shared_embedding_layer = Embedding(G.embedding_dimension)

	word_embedding = shared_embedding_layer(word_one_hot)
	context_embeddings = [shared_embedding_layer(x) for x in context_one_hots]
	negative_embeddings = [shared_embedding_layer(x) for x in negative_one_hots]

	print(word_embedding.shape)
	word_embedding_reshaped = C.reshape(word_embedding, shape=(1, G.embedding_dimension))
	print(word_embedding_reshaped.shape)

	context_embeddings_all = C.reshape(C.splice(*context_embeddings), shape=(context_size, G.embedding_dimension))
	negative_embeddings_all = C.reshape(C.splice(*negative_embeddings), shape=(G.negative, G.embedding_dimension))
	print(context_embeddings_all.shape)
	print(negative_embeddings_all.shape)
	cbow = C.reshape(C.reduce_mean(context_embeddings_all, 0), shape=(G.embedding_dimension))
	print(cbow.shape)

	# word_context_product = C.times_transpose(word_embedding_reshaped, cbow)
	word_context_product = C.times_transpose(word_embedding, cbow)
	print(word_context_product.shape)
	negative_context_product = C.reshape(C.times_transpose(negative_embeddings_all, cbow), shape=(G.negative))
	print(negative_context_product.shape)

	word_negative_context_product = C.splice(word_context_product, negative_context_product)
	print(word_negative_context_product.shape)
	# return model and shared embedding layer
	return word_negative_context_product, shared_embedding_layer

def create_trainer():
	# Will take the model and the batch generator to create a Trainer
	# Will return the input variables, trainer variable, model and the embedding layer
	##################################################
	################### Inputs #######################
	##################################################
	word_one_hot = C.input_variable((G.embedding_vocab_size), np.float32, is_sparse=True, name=G.word_input_field)
	context_one_hots = [C.input_variable((G.embedding_vocab_size), np.float32, is_sparse=True, name=G.context_input_field.format(i)) for i in range(context_size)]
	negative_one_hots = [C.input_variable((G.embedding_vocab_size), np.float32, is_sparse=True, name=G.negative_input_field.format(i)) for i in range(G.negative)]

	# The target labels should have first as 1 and rest as 0
	targets = C.input_variable((G.negative + 1), np.float32, name=G.target_input_field)

	word_negative_context_product, embedding_layer = create_word2vec_cbow_model(word_one_hot, context_one_hots, negative_one_hots)
	loss = C.binary_cross_entropy(word_negative_context_product, targets)
	eval_loss = C.binary_cross_entropy(word_negative_context_product, targets)
	print("loss functions done")
	lr_schedule = learning_rate_schedule(G.learning_rate, UnitType.minibatch)

	learner = adam_sgd(word_negative_context_product.parameters, lr = lr_schedule, momentum = momentum_as_time_constant_schedule(700))
	print("learner done")
	trainer = Trainer(word_negative_context_product, (loss, eval_loss), learner)
	
	return word_one_hot, context_one_hots, negative_one_hots, targets, trainer, word_negative_context_product, embedding_layer

def save_embeddings(word_negative_context_product, vocabulary):
	print("Saving Embeddings")
	# save the embeddings
	for k in word_negative_context_product.parameters:
		print(k.name, k.shape)
		# print(k.value)
		S.save_embeddings("embeddings.txt", k.value, vocabulary)

def train():
	global sentences, vocabulary, reverse_vocabulary
	# function will create the trainer and train it for specified number of epochs
	# Print loss 50 times while training
	print_freqency = 50
	pp = ProgressPrinter(print_freqency)

	# get the trainer
	word_one_hot, context_one_hots, negative_one_hots, targets, trainer, word_negative_context_product, embedding_layer = create_trainer()
	
	# Create a CTF reader which reads the sparse inputs
	print("reader started")
	reader = CTFDeserializer(G.CTF_input_file)
	reader.map_input(G.word_input_field, dim=G.embedding_vocab_size, format="sparse")
	# context inputs
	for i in range(context_size):
		reader.map_input(G.context_input_field.format(i), dim=G.embedding_vocab_size, format="sparse")
	# negative inputs
	for i in range(G.negative):
		reader.map_input(G.negative_input_field.format(i), dim=G.embedding_vocab_size, format="sparse")
	# targets
	reader.map_input(G.target_input_field, dim=(G.negative + 1), format="dense")
	print("reader done")

	# Get minibatch source from reader
	is_training = True
	minibatch_source = MinibatchSource(reader, randomize=is_training, epoch_size=INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)
	minibatch_source.streams[targets] = minibatch_source.streams[G.target_input_field]
	del minibatch_source.streams[G.target_input_field]
	print("minibatch source done")
	
	total_minibatches = total_training_instances // G.minibatch_size
	print("traning started")
	print("Total minibatches to train =", total_minibatches)
	for i in range(total_minibatches):
		# Collect minibatch
		# start_batch_collection = time.time()
		mb = minibatch_source.next_minibatch(G.minibatch_size, input_map=minibatch_source.streams)
		# end_batch_collection = time.time()
		# print("Batch collection time = %.6fsecs" % (end_batch_collection - start_batch_collection))
		# print("Time taken to collect one training_instance = %.6fsecs" % ((end_batch_collection - start_batch_collection)/G.minibatch_size))
		# Train minibatch
		# start_train = time.time()
		trainer.train_minibatch(mb)
		# end_train = time.time()
		# print("minibatch train time = %.6fsecs" % (end_train - start_train))
		# print("Time per training instance = %.6fsecs" % ((end_train - start_train)/G.minibatch_size))
		# Update progress printer
		pp.update_with_trainer(trainer)

		# start_batch_collection = time.time()
	print("Total training instances =", total_training_instances)
	return word_negative_context_product

start = time.time()

word_negative_context_product = train()
# save the embeddings
# save_embeddings(word_negative_context_product, vocabulary)

end = time.time()
print("Time taken to create and train the model = %.2fsecs" % (end - start))
# print(embedding_layer.shape)
# print(embedding_layer.values())
