import os, sys
abspath = os.path.abspath(__file__)
dn = os.path.dirname
nls_root = dn(dn(dn(abspath)))
sys.path.insert(0, nls_root)   # nls
sys.path.insert(0, dn(dn(abspath)))       # wh_gcp
del dn

import tensorflow as tf
from tframe import FLAGS
from tframe import console

from signals.utils.dataset import load_wiener_hammerstein, DataSet
import trainer.model_lib as model_lib


def main(_):
  console.start('trainer.task')

  # Set default flags
  if FLAGS.use_default:
    FLAGS.train = True
    FLAGS.overwrite = True
    FLAGS.smart_train = False
    FLAGS.save_best = False

  WH_PATH = os.path.join(nls_root, 'data/wiener_hammerstein/whb.tfd')
  MARK = 'mlp00'
  MEMORY_DEPTH = 40
  LAYER_DIM = MEMORY_DEPTH * 2
  LAYER_NUM = 4
  LEARNING_RATE = 0.001
  BATCH_SIZE = 32
  PRINT_CYCLE = 100
  EPOCH = 10

  # Get model
  model = model_lib.mlp_00(
    MARK, MEMORY_DEPTH, LAYER_DIM, LAYER_NUM, LEARNING_RATE)

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
    pass

  console.end()


if __name__ == "__main__":
  tf.app.run()
