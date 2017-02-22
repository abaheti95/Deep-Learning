
UNKNOWN_WORD = "<unk>"

embedding_dimension = 100
min_count = 2
window_size = 3
sample = 1e-3
negative = 5
vocab_size = 1000000
train_words = None

# CTF input creation settings
CTF_input_file = "ctf_input_" + str(min_count) + "_" + str(window_size) + "_" + str(negative) + ".txt"
Input_Config_file = "ctf_config_" + str(min_count) + "_" + str(window_size) + "_" + str(negative) + ".txt"
word_input_field = "word"
context_input_field = "context{}"
negative_input_field = "negative{}"
target_input_field = "targets"

# Special parameters
MIN_SENTENCE_LENGTH = 3

# Training parameters
learning_rate = 0.0025
num_minibatches = None
minibatch_size = 2
