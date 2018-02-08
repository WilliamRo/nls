from tframe import FLAGS
from tframe import console

from signals.utils.dataset import load_wiener_hammerstein, DataSet

import wh_model_lib

# =============================================================================
# Global configuration
WH_PATH = '../data/wiener_hammerstein/whb.tfd'
VAL_SIZE = 20000

MEMORY_DEPTH = 80
D = MEMORY_DEPTH
NN_EPOCH = 500
NN_LEARNING_RATE = 0.001

FLAGS.train = True
# FLAGS.train = False
FLAGS.overwrite = True
# FLAGS.overwrite = False
FLAGS.save_best = False
# FLAGS.save_best = True

FLAGS.smart_train = True
FLAGS.epoch_tol = 20

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

model = wh_model_lib.test_00(MEMORY_DEPTH, NN_LEARNING_RATE)

# Define model and identify
if FLAGS.train: model.identify(
  train_set, val_set, batch_size=10, print_cycle=100, epoch=NN_EPOCH)

# Evaluation
if EVALUATION:
  model.evaluate(train_set, start_at=1000, plot=False)
  model.evaluate(test_set, start_at=1000, plot=PLOT)
