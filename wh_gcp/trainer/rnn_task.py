import tensorflow as tf

try:
  from tframe import hub
  from tframe import console
  from tframe import SaveMode
  from tframe.trainers import SmartTrainer, SmartTrainerHub

  from models.neural_net import NlsHub
  from signals.utils.dataset import load_wiener_hammerstein, DataSet
  import trainer.model_lib as model_lib
  hub.data_dir = '../../data/wiener_hammerstein/whb.tfd'
except:
  import sys, os
  abspath = os.path.abspath(__file__)
  dn = os.path.dirname
  nls_root = dn(dn(dn(abspath)))
  sys.path.insert(0, nls_root)   # nls
  sys.path.insert(0, dn(dn(abspath)))       # wh_gcp
  sys.path.append(dn)
  del dn

  from tframe import hub
  from tframe import console
  from tframe import SaveMode
  from tframe.trainers import SmartTrainer, SmartTrainerHub

  from models.neural_net import NlsHub
  from signals.utils.dataset import load_wiener_hammerstein, DataSet
  import trainer.model_lib as model_lib

  hub.data_dir = os.path.join(nls_root, 'data/wiener_hammerstein/whb.tfd')


def main(_):
  console.start('RNN task')

  description = '0'
  # Configurations
  th = NlsHub(as_global=True)
  th.memory_depth = 2
  th.num_blocks = 1
  th.multiplier = 10
  th.hidden_dim = th.memory_depth * th.multiplier
  th.num_steps = 8

  th.epoch = 2000000
  th.batch_size = 32
  th.learning_rate = 1e-3
  th.validation_per_round = 20
  th.print_cycle = 0

  th.train = True
  th.smart_train = False
  th.idle_tol = 20
  th.max_bad_apples = 4
  th.lr_decay = 0.6
  th.early_stop = True
  th.save_mode = SaveMode.ON_RECORD
  th.warm_up_rounds = 10
  th.overwrite = True                        # Default: False
  th.export_note = True
  th.summary = False
  th.activation_sum = False
  th.save_model = False

  th.mark = 'rnn-{}x{}x({}x{})-{}'.format(
    th.num_blocks, th.num_steps, th.memory_depth, th.multiplier, description)

  # Get model
  model = model_lib.rnn0(th)
  # Load data
  train_set, val_set, test_set = load_wiener_hammerstein(
    th.data_dir, depth=th.memory_depth, validation_size=2000)
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
    model.evaluate(test_set, start_at=th.memory_depth)

  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()

