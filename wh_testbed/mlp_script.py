import numpy as np
import tensorflow as tf

from tframe import FLAGS
from tframe import console

from models.neural_net import NeuralNet

from signals.utils.dataset import load_wiener_hammerstein, DataSet
from signals.utils.figure import Figure, Subplot

import wh_model_lib

# =============================================================================
# Global configuration
WH_PATH = '../data/wiener_hammerstein/whb.tfd'
VAL_SIZE = 20000

MEMORY_DEPTH = 80
D = MEMORY_DEPTH
NN_EPOCH = 500
NN_HID_DIMS = [2 * D] * 4
MODEL_MARK = 'mlp_D{}_{}_{}'.format(MEMORY_DEPTH, NN_HID_DIMS, 'lrelu')
# FLAGS.epoch_tol = 10

NN_LEARNING_RATE = 0.001

FLAGS.smart_train = True

FLAGS.train = True
# FLAGS.train = False

FLAGS.overwrite = True
# FLAGS.overwrite = False

FLAGS.save_best = False
# FLAGS.save_best = True

# Turn off overwrite while in save best mode
FLAGS.overwrite = FLAGS.overwrite and not FLAGS.save_best and FLAGS.train

EVALUATION = not FLAGS.train
PLOT = EVALUATION
# =============================================================================
# Load data set
train_set, val_set, test_set = load_wiener_hammerstein(
  WH_PATH, depth=MEMORY_DEPTH)
assert isinstance(train_set, DataSet)
assert isinstance(val_set, DataSet)
assert isinstance(test_set, DataSet)

model = wh_model_lib.mlp_00(
  MEMORY_DEPTH, MODEL_MARK, NN_HID_DIMS, NN_LEARNING_RATE)

# Define model and identify
if FLAGS.train: model.identify(
  train_set, val_set, batch_size=10, print_cycle=100, epoch=NN_EPOCH)

# Evaluation
if EVALUATION:
  model.evaluate(train_set, start_at=1000, plot=False)
  model.evaluate(test_set, start_at=1000, plot=PLOT)



