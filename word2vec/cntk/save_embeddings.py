import codecs
import global_settings as G

def write_array_to_file(wf, array):
	for i in range(len(array)):
		wf.write(str(array.item(i)) + " ")
	wf.write("\n")

def save_embeddings(save_filepath, weights, vocabulary):
	with codecs.open(save_filepath, "w", "utf-8") as wf:
		# First line is vocabulary size and embedding dimension
		wf.write(str(len(vocabulary)) + " " + str(weights.shape[1]) + "\n")
		# Now each line is word "\t" and embedding
		# First word is UNKNOWN_WORD by our convention
		index = 1
		wf.write(G.UNKNOWN_WORD + "\t")
		write_array_to_file(wf, weights[index])
		index += 1
		# Now emit embedding for each word based on their sorted counts
		sorted_words = reversed(sorted(vocabulary, key=lambda word: vocabulary[word]))
		for word in sorted_words:
			if word == G.UNKNOWN_WORD:
				continue
			wf.write(word + "\t")
			write_array_to_file(wf, weights[index])
			index += 1
