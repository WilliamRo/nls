import tensorflow as tf
import misc

from tframe import hub
from tframe import console
from tframe import SaveMode

from models.neural_net import NlsHub
from signals.utils.dataset import load_wiener_hammerstein, DataSet
import trainer.model_lib as model_lib

hub.data_dir = misc.from_root('data/wiener_hammerstein/whb.tfd')


def main(_):
  console.start('RNN task')

  # Configurations
  th = NlsHub(as_global=True)
  th.memory_depth = 10
  th.num_blocks = 1
  th.multiplier = 8
  th.hidden_dim = th.memory_depth * th.multiplier
  th.num_steps = 32

  # th.input_gate = False
  # th.forget_gate = False
  # th.output_gate = False

  th.epoch = 100000
  th.batch_size = 32
  th.learning_rate = 1e-4
  th.validation_per_round = 20
  th.print_cycle = 0

  # th.train = False
  th.smart_train = True
  th.max_bad_apples = 4
  th.lr_decay = 0.6

  th.early_stop = True
  th.idle_tol = 20
  th.save_mode = SaveMode.ON_RECORD
  th.warm_up_thres = 1
  th.at_most_save_once_per_round = True

  th.overwrite = True                        # Default: False
  th.export_note = True
  th.summary = True
  th.monitor_preact = False
  th.save_model = True

  th.allow_growth = False
  th.gpu_memory_fraction = 0.15

  description = '0'
  th.mark = 'lstm[{}]{}x({}x{})-{}steps-{}'.format(
    ('i' if th.input_gate else '') + ('f' if th.forget_gate else '') +
    ('o' if th.output_gate else '') + 'g',
    th.num_blocks, th.memory_depth, th.multiplier, th.num_steps, description)
  # Get model
  model = model_lib.lstm0(th)
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

