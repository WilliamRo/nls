import tensorflow as tf
import os
import sys

abspath = os.path.abspath(__file__)
dn = os.path.dirname
nls_root = dn(dn(dn(abspath)))
sys.path.insert(0, nls_root)   # nls
sys.path.insert(0, dn(dn(abspath)))       # wh_gcp
sys.path.append(dn)
del dn

from tframe import FLAGS
from tframe import console

from signals.utils.dataset import load_wiener_hammerstein, DataSet
import trainer.model_lib as model_lib


def main(_):
  console.start('trainer.task')

  EPOCH = 1000
  # FLAGS.train = False
  FLAGS.overwrite = True
  # FLAGS.save_best = True
  FLAGS.smart_train = True

  # Hyper parameters
  LEARNING_RATE = 0.001
  LAYER_NUM = 4
  BATCH_SIZE = 32
  MEMORY_DEPTH = 80
  LAYER_DIM = MEMORY_DEPTH * 2
  ACTIVATION = 'relu'

  # Set default flags
  FLAGS.progress_bar = True

  FLAGS.save_model = True
  FLAGS.summary = False
  FLAGS.snapshot = False

  PRINT_CYCLE = 100

  WH_PATH = os.path.join(nls_root, 'data/wiener_hammerstein/whb.tfd')
  MARK = 'mlp00'

  # Get model
  model = model_lib.mlp_00(
    MARK, MEMORY_DEPTH, LAYER_DIM, LAYER_NUM, LEARNING_RATE,
    activation=ACTIVATION)

  # Load data set
  train_set, val_set, test_set = load_wiener_hammerstein(
    WH_PATH, depth=MEMORY_DEPTH)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)

  # Train or evaluate
  if FLAGS.train:
    model.identify(train_set, val_set, batch_size=BATCH_SIZE,
                   print_cycle=PRINT_CYCLE, epoch=EPOCH)
  else:
    model.evaluate(train_set, start_at=1000, plot=False)
    model.evaluate(test_set, start_at=1000, plot=False)

  console.end()


if __name__ == "__main__":
  tf.app.run()
