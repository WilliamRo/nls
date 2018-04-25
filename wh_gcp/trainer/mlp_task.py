import tensorflow as tf

try:
  from tframe import hub
  from tframe import console
  from tframe import SaveMode
  from tframe.trainers import SmartTrainer, SmartTrainerHub

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

  from signals.utils.dataset import load_wiener_hammerstein, DataSet
  import trainer.model_lib as model_lib

  hub.data_dir = os.path.join(nls_root, 'data/wiener_hammerstein/whb.tfd')


def main(_):
  console.start('mlp task')

  # Configurations
  MEMORY_DEPTH = 80
  HIDDEN_DIM = MEMORY_DEPTH * 16
  HIDDEN_LAYER_NUM = 1

  th = SmartTrainerHub(as_global=True)
  th.mark = 'mlp00'
  th.epoch = 5
  th.batch_size = 64
  th.learning_rate = 0.001
  th.validation_per_round = 20

  th.train = True
  th.early_stop = True
  th.save_mode = SaveMode.ON_RECORD
  th.warm_up_rounds = 10
  th.overwrite = True
  # th.export_note = True
  th.summary = False

  # Get model
  model = model_lib.mlp_00(th.mark, MEMORY_DEPTH, HIDDEN_DIM, HIDDEN_LAYER_NUM,
                           th.learning_rate)

  # Load data
  train_set, val_set, test_set = load_wiener_hammerstein(
    th.data_dir, depth=MEMORY_DEPTH)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)

  # Train or evaluate
  if th.train:
    model.nn.train(train_set, validation_set=val_set, trainer_hub=th)
  else:
    model.evaluate(train_set, start_at=MEMORY_DEPTH)
    model.evaluate(val_set, start_at=MEMORY_DEPTH)
    model.evaluate(test_set, start_at=MEMORY_DEPTH)

  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()

