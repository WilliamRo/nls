import tensorflow as tf
import os, sys

dn = os.path.dirname
abs_path = os.path.abspath(__file__)
sys.path.append(dn(abs_path))
sys.path.append(dn(dn(abs_path)))
del dn, abs_path

from tframe import console
from tframe import SaveMode
from tframe.trainers import SmartTrainerHub

from signals.utils.dataset import load_wiener_hammerstein, DataSet

import lott_lib


def main(_):
  console.start('Lottery')

  # Configurations
  MEMORY_DEPTH = 80
  HIDDEN_DIM = MEMORY_DEPTH * 16
  FIX_PRE_WEIGHT = True
  BRANCH_INDEX = 0

  th = SmartTrainerHub(as_global=True)
  th.mark = 'mlp00'
  th.epoch = 1000
  th.batch_size = 64
  th.learning_rate = 0.001
  # th.print_cycle = 80
  # th.validate_cycle = 300
  th.validation_per_round = 40
  th.idle_tol = 20
  th.max_bad_apples = 6

  th.train = True
  # th.smart_train = True
  th.overwrite = True and BRANCH_INDEX == 0
  th.export_note = True
  th.summary = False
  th.snapshot = False
  # th.save_model = False
  th.save_mode = SaveMode.ON_RECORD
  th.warm_up_rounds = 10

  # Load data
  train_set, val_set, test_set = load_wiener_hammerstein(
    r'../data/wiener_hammerstein/whb.tfd', depth=MEMORY_DEPTH)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)

  # Get model
  model = lott_lib.mlp00(th.mark, MEMORY_DEPTH, HIDDEN_DIM, th.learning_rate)

  # Train or evaluate
  if th.train:
    model.nn.train(train_set, validation_set=val_set, trainer_hub=th,
                   branch_index=BRANCH_INDEX, freeze=FIX_PRE_WEIGHT)
  else:
    # BRANCH_INDEX = 1
    model.evaluate(train_set, start_at=MEMORY_DEPTH, branch_index=BRANCH_INDEX)
    model.evaluate(val_set, start_at=MEMORY_DEPTH, branch_index=BRANCH_INDEX)
    model.evaluate(test_set, start_at=MEMORY_DEPTH, branch_index=BRANCH_INDEX)

  console.end()


if __name__ == '__main__':
  tf.app.run()
