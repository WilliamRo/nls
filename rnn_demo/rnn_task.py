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
  console.start('rnn_task')

  console.end()


if __name__ == "__main__":
  tf.app.run()
