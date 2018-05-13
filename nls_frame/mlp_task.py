import tensorflow as tf

import sys, os
abspath = os.path.abspath(__file__)
dn = os.path.dirname
nls_root = dn(dn(abspath))
sys.path.insert(0, nls_root)
sys.path.append(dn)

from tframe import hub
from tframe import console
from tframe import SaveMode

from models.neural_net import NlsHub
from signals.utils.dataset import DataSet
from data_util import load_data
import nlsf_model_lib

hub.data_dir = os.path.join(nls_root, 'nls_frame/wrap_data/hello_nls_data.tfd')


def main(_):
  console.start('mlp task')

  # Configurations
  th = NlsHub(as_global=True)
  th.memory_depth = 6
  th.num_blocks = 2
  th.multiplier = 2
  th.hidden_dim = th.memory_depth * th.multiplier
  # th.actype1 = 'lrelu'   # Default: relu

  th.epoch = 10
  th.batch_size = 32
  th.learning_rate = 1e-4
  th.validation_per_round = 5
  th.print_cycle = 100

  th.train = True
  # th.smart_train = True
  # th.max_bad_apples = 4
  # th.lr_decay = 0.6

  th.early_stop = True
  th.idle_tol = 20
  th.save_mode = SaveMode.NAIVE
  # th.warm_up_thres = 1
  # th.at_most_save_once_per_round = True

  th.overwrite = True
  th.export_note = True
  th.summary = False
  # th.monitor = True
  th.save_model = True

  th.allow_growth = False
  th.gpu_memory_fraction = 0.40

  description = '0'
  th.mark = 'mlp-{}x({}x{})-{}'.format(
    th.num_blocks, th.memory_depth, th.multiplier, description)
  # Get model
  model = nlsf_model_lib.mlp_00(th)
  # Load data
  train_set, val_set, test_set = load_data(
    th.data_dir, depth=th.memory_depth)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)

  # Train or evaluate
  if th.train:
    model.nn.train(train_set, validation_set=val_set, trainer_hub=th)
  else:
    console.show_status('Evaluating ...')
    model.evaluate(train_set, start_at=th.memory_depth)
    model.evaluate(val_set, start_at=th.memory_depth)
    model.evaluate(test_set, start_at=th.memory_depth, plot=True)

  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()

