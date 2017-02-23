
UNKNOWN_WORD = "<unk>"

embedding_dimension = 100
min_count = 3
window_size = 3
sample = 1e-3
negative = 5
vocab_size = 1000000
train_words = None

# CTF input creation settings
regen_vocab = False
vocab_file = "7lang_vocab.txt"
regen_input_file = False
CTF_input_file = "ctf_7lang_input_" + str(min_count) + "_" + str(window_size) + "_" + str(negative) + ".txt"
CTF_config_file = "ctf_7lang_config_" + str(min_count) + "_" + str(window_size) + "_" + str(negative) + ".txt"
word_input_field = "word"
context_input_field = "context{}"
negative_input_field = "negative{}"
target_input_field = "targets"

# Special parameters
MIN_SENTENCE_LENGTH = 3

# Training parameters
learning_rate = 0.0025 * 1000
num_minibatches = None
minibatch_size = 4096
