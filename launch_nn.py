import tensorflow as tf

from tframe import FLAGS
from tframe import console

from signals.utils.dataset import load_wiener_hammerstein, DataSet
import nn_models


def main(_):
  # =============================================================================
  # Global configuration
  WH_PATH = './data/wiener_hammerstein/whb.tfd'

  NN_LEARNING_RATE = 0.001
  BATCH_SIZE = 32

  MEMORY_DEPTH = 40
  NN_EPOCH = 500
  PRINT_CYCLE = 10

  FLAGS.train = True
  # FLAGS.train = False
  FLAGS.overwrite = True
  # FLAGS.overwrite = False
  FLAGS.save_best = False
  # FLAGS.save_best = True

  FLAGS.smart_train = False
  FLAGS.epoch_tol = 20

  # Turn off overwrite while in save best mode
  FLAGS.overwrite = FLAGS.overwrite and not FLAGS.save_best and FLAGS.train
  # =============================================================================

  console.start('NN Demo')

  # Load data set
  train_set, val_set, test_set = load_wiener_hammerstein(
    WH_PATH, depth=MEMORY_DEPTH)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)

  model = nn_models.mlp_00(MEMORY_DEPTH, NN_LEARNING_RATE)

  # Define model and identify
  if FLAGS.train: model.identify(
    train_set, val_set, batch_size=BATCH_SIZE,
    print_cycle=PRINT_CYCLE, epoch=NN_EPOCH)

  console.end()


if __name__ == "__main__":
  tf.app.run()
