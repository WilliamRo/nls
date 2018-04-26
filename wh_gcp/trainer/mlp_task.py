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
  console.start('mlp task')

  # Configurations
  th = NlsHub(as_global=True)
  th.memory_depth = 80
  th.hidden_num = 1
  th.multiplier = 8
  th.hidden_dim = th.memory_depth * th.multiplier

  th.mark = 'mlp00'
  th.epoch = 50000
  th.batch_size = 64
  th.learning_rate = 0.0001
  th.validation_per_round = 30

  th.train = True
  th.smart_train = True
  th.idle_tol = 20
  th.early_stop = True
  th.save_mode = SaveMode.ON_RECORD
  th.warm_up_rounds = 50
  th.overwrite = True
  th.export_note = True
  th.summary = False
  th.save_model = True

  # Get model
  model = model_lib.mlp_00(th.mark, th.memory_depth, th.hidden_dim,
                           th.hidden_num, th.learning_rate)
  # Load data
  train_set, val_set, test_set = load_wiener_hammerstein(
    th.data_dir, depth=th.memory_depth)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)

  # Train or evaluate
  if th.train:
    model.nn.train(train_set, validation_set=val_set, trainer_hub=th)
  else:
    model.evaluate(train_set, start_at=th.memory_depth)
    model.evaluate(val_set, start_at=th.memory_depth)
    model.evaluate(test_set, start_at=th.memory_depth)

  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()

