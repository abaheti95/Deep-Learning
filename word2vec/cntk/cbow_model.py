from __future__ import print_function
import numpy as np
import cntk as C
from cntk.learner import sgd, learning_rate_schedule, UnitType
from cntk.utils import ProgressPrinter
from cntk.layers import Embedding
from cntk.models import Sequential

import global_settings as G

context_size = G.window_size * 2
##################################################
################### Inputs #######################
##################################################
word_one_hot = C.input_variable((G.vocab_size), np.float32)
context_one_hots = C.input_variable((context_size, G.vocab_size), np.float32)
negative_one_hots = C.input_variable((G.negative, G.vocab_size), np.float32)

 

