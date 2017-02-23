import global_settings as G
import vocab_generator as V_gen

"""
This file will take the training instance generator and create a CTF format file
which will be consumed by the CNTK reader during the training
"""

def create_CTF_format_input(sentences, vocabulary, reverse_vocabulary):
	with open(G.CTF_input_file, 'w') as wf:
		training_instances = 0
		for inputs in V_gen.pretraining_batch_generator(sentences, vocabulary, reverse_vocabulary):
			word_index = inputs[0]
			context_indexes = inputs[1]
			negative_indexes = inputs[2]
			# write the input to file
			wf.write("|" + G.word_input_field + " " + str(word_index[0]) + ":1 ")
			for idx, context_word_index in enumerate(context_indexes):
				wf.write("|" + G.context_input_field.format(idx) + " " + str(context_word_index) + ":1 ")
			zeros_string = ""
			for idx, negative_word_index in enumerate(negative_indexes):
				wf.write("|" + G.negative_input_field.format(idx) + " " + str(negative_word_index) + ":1 ")
				zeros_string += "0 "
			# Create targets
			wf.write("|" + G.target_input_field + " 1 " + zeros_string)
			#new line
			wf.write("\n")

			training_instances += 1
			if training_instances%1000000 == 0:
				print(training_instances, "entries created")
	with open(G.CTF_config_file, 'w') as wf:
		wf.write(str(training_instances) + "\n")
	return training_instances





