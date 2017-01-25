"""
I'm trying to implement a keras version of cbow word2vec with negative sampling
This file will generate vocabulary from the sentences generator
"""

import collections
import math
import os
import random
import sys

# local imports
# Global settings for the project
import golbal_settings as G

# vocabulary will be a dictionary of words and their corresponding counts
def build_vocabulary(vocabulary, sentences):
	print "Generating Vocabulary from the sentences"
	for sentence in sentences:
		for word in sentence.strip().split():
			vocabulary.setdefault(word, 0)
			vocabulary[word] += 1
	print "Vocabulary size = %d" % len(vocabulary)

def filter_vocabulary_based_on(vocabulary, min_count):
	print "Deleting the words which occur less than %d times" % min_count
	# find the words to be deleted
	delete_word_list = [word for word, count in vocabulary.iteritems() if count < min_count]
	# All the words which will be deleted from the corpus will become unknown words
	# Therefore counting the number of unkown words in the corpus
	unk_count = 0
	for word in delete_word_list:
		# delete the low occurance word from the vocabulary and add to the unknown counts
		unk_count += vocabulary.pop(word, 0)
	vocabulary[G.UNKNOWN_WORD] = unk_count
	print "Vocabulary size after filtering words = %d" % len(vocabulary)

def generate_inverse_vocabulary_lookup(vocabulary):
	# It is assumed the the vocabulary here has the UNKNOWN_WORD
	# sorting the words in vocabulary in descending order
	sorted_words = reversed(sorted(vocabulary, key=lambda word: vocabulary[word]))
	# creating a reverse index lookup
	reverse_vocabulary = dict()
	# UNKOWN_WORD will have the index 0
	reverse_vocabulary[G.UNKNOWN_WORD] = 0
	index = 1
	for word in sorted_words:
		if word != G.UNKNOWN_WORD:
			reverse_vocabulary[word] = index
			index += 1
	return reverse_vocabulary

def sentences_to_index_sequences(sentences, vocabulary, save_filepath):
	# generate inverse vocabulary lookup
	reverse_vocabulary = generate_inverse_vocabulary_lookup(vocabulary)
	for sentence in sentences:
		# creating a list of word numbers for current sentence
		sentence_indexes = list()
		for word in sentence.split():
			if word in reverse_vocabulary:
				sentence_indexes.append(reverse_vocabulary[word])
			else:
				# UNKNOWN_WORD index
				sentence_indexes.append(0)
		# TODO: save the sentence indexes in the save_filepath



