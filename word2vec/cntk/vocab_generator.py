"""
I'm trying to implement a cntk version of cbow word2vec with negative sampling
This file will generate vocabulary from the sentences generator
"""
from __future__ import print_function

import collections
import math
import os
import random
import sys
import math
import random
import codecs
import numpy as np

# local imports
# Global settings for the project
import global_settings as G

# vocabulary will be a dictionary of words and their corresponding counts
def build_vocabulary(vocabulary, sentences):
	# global G.train_words
	print("Generating Vocabulary from the sentences")
	# Count the total number of training words
	G.train_words = 0
	for sentence in sentences:
		for word in sentence.strip().split():
			vocabulary.setdefault(word, 0)
			vocabulary[word] += 1
			G.train_words += 1;
	print("Vocabulary size = %d" % len(vocabulary))
	print("Total words to be trained = %d" % G.train_words)

def filter_vocabulary_based_on(vocabulary, min_count):
	# global G.vocab_size
	print("Deleting the words which occur less than %d times" % min_count)
	# find the words to be deleted
	delete_word_list = [word for word, count in vocabulary.items() if count < min_count]
	# All the words which will be deleted from the corpus will become unknown words
	# Therefore counting the number of unkown words in the corpus
	unk_count = 0
	for word in delete_word_list:
		# delete the low occurance word from the vocabulary and add to the unknown counts
		unk_count += vocabulary.pop(word, 0)
	vocabulary[G.UNKNOWN_WORD] = unk_count
	G.vocab_size = len(vocabulary)
	print("Vocabulary size after filtering words = %d" % G.vocab_size)

def generate_inverse_vocabulary_lookup(vocabulary, save_filepath):
	# It is assumed the the vocabulary here has the UNKNOWN_WORD
	# sorting the words in vocabulary in descending order
	sorted_words = reversed(sorted(vocabulary, key=lambda word: vocabulary[word]))
	# creating a reverse index lookup
	reverse_vocabulary = dict()
	# index 0 is reserved for padding
	# UNKNOWN_WORD will have the index 1
	reverse_vocabulary[G.UNKNOWN_WORD] = 1
	index = 2
	with codecs.open(save_filepath, "w", "utf-8") as wf:
		# first save the UNKNOWN_WORD
		wf.write(G.UNKNOWN_WORD + "\t" + str(vocabulary[G.UNKNOWN_WORD]) + "\n")
		for word in sorted_words:
			if word == G.UNKNOWN_WORD:
				continue
			reverse_vocabulary[word] = index
			# also write the word in the save file
			wf.write(word + "\t" + str(vocabulary[word]) + "\n")
			index += 1
	return reverse_vocabulary

def subsample_sentence(sentence, vocabulary):
	subsampled_sentence = list()
	# replace words with unknown word is not found in vocabulary
	sentence = [word if word in vocabulary else G.UNKNOWN_WORD for word in sentence]
	if G.sample <= 0:
		# If sampling is set to zero then don't do the sampling
		return sentence
	for word in sentence:
		# If the word is occuring frequently then the probablity of retaining that word is less
		prob = (math.sqrt(vocabulary[word] / (G.sample * G.train_words)) + 1) * (G.sample * G.train_words) / vocabulary[word]
		rand = random.random()
		if prob < rand:
			continue
		else:
			subsampled_sentence.append(word)
	# print("subsampled length = %d/%d" % (len(subsampled_sentence), len(sentence)))
	# if len(subsampled_sentence) == len(sentence):
	# 	print("Dafuq didn't subsample %d" % len(sentence))
	return subsampled_sentence

def get_negative_samples(current_word_index):
	# Generate random negative samples
	negative_samples = random.sample(range(G.vocab_size), G.negative)
	while current_word_index in negative_samples:
		negative_samples = random.sample(range(G.vocab_size), G.negative)
	return negative_samples

def pretraining_batch_generator(sentences, vocabulary, reverse_vocabulary):
	# Read one sentence from the file into memory
	# For each sentence generate word index sequence
	# Now spit out batches for training
	# Each batch will emit the following things
	# 1 - current word index
	# 2 - context word indexes
	# 3 - negative sampled word indexes
	for sentence in sentences:
		# split the sentence on whitespace
		sentence = sentence.split()
		# Now we have to perform subsampling of the sentence to remove frequent words
		# This will improve the speed
		sentence = subsample_sentence(sentence, vocabulary)
		if len(sentence) < G.MIN_SENTENCE_LENGTH:
			continue
		sent_seq = [reverse_vocabulary[word] for word in sentence]

		# Create current batch
		sentence_length = len(sent_seq)
		for i in range(sentence_length):
			current_word_index = None
			context_word_indexes = list()
			for j in range(-G.window_size, G.window_size + 1):
				# j will be of indices -G.window_size to G.window_size
				if j == 0:
					# current word
					current_word_index = [sent_seq[i]]
				else:
					# context word
					if (i+j) < 0 or (i+j) >= sentence_length:
						# pad with zeros
						context_word_indexes.append(0)
					else:
						context_word_indexes.append(sent_seq[(i+j)]) 
			# get negative samples
			negative_samples = get_negative_samples(current_word_index)
			# yield a batch here
			# batch should be a tuple of inputs and targets
			# print([current_word_index.shape, context_word_indexes.shape, negative_samples.shape], [np.array([1.0]).shape, np.zeros((1,G.negative)).shape])
			# print([current_word_index, context_word_indexes, negative_samples], [np.array([1.0]), np.zeros((1,G.negative))])
			yield (current_word_index, context_word_indexes, negative_samples)

def sentences_to_index_sequences(sentences, vocabulary, save_vocab_filepath, save_index_filepath):
	# generate inverse vocabulary lookup
	reverse_vocabulary = generate_inverse_vocabulary_lookup(vocabulary, save_vocab_filepath)

	with open(save_index_filepath, "w") as wf:
		for sentence in sentences:
			# creating a list of word numbers for current sentence
			sentence_indexes = list()
			for word in sentence.split():
				if word in reverse_vocabulary:
					sentence_indexes.append(reverse_vocabulary[word])
				else:
					# UNKNOWN_WORD index
					sentence_indexes.append(0)
			# Save the sentence indexes in the save_filepath
			write_indexes_to_file(wf, sentence_indexes)
	return reverse_vocabulary


