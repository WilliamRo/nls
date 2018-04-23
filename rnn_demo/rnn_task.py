import os, sys
import tensorflow as tf

abspath = os.path.abspath(__file__)
dn = os.path.dirname
nls_root = dn(dn(abspath))
sys.path.insert(0, nls_root)
sys.path.insert(0, dn(abspath))
del dn, abspath

from tframe import FLAGS
from tframe import console

from signals.utils.dataset import load_wiener_hammerstein, DataSet
import rnn_models


def main(_):
  # Configuration
  # FLAGS.train = False
  # FLAGS.smart_train = True

  FLAGS.overwrite = True
  FLAGS.summary = True
  FLAGS.save_model = False
  FLAGS.snapshot = False

  MEMORY_DEPTH = 1
  EPOCH = 2

  # Start
  console.start('rnn_task')

  # Initiate model
  model = rnn_models.vanilla_RNN('rnn00')

  # Load data
  train_set, val_set, test_set = load_wiener_hammerstein(
    r'../data/wiener_hammerstein/whb.tfd', depth=MEMORY_DEPTH)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)

  # Train or evaluate
  if FLAGS.train:
    pass
  else:
    console.show_status('Evaluating ...')

  # End
  console.end()


if __name__ == "__main__":
  tf.app.run()
