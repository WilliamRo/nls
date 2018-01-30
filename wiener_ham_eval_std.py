import numpy as np
import tensorflow as tf

from tframe import FLAGS
from tframe import console

from models.neural_net import NeuralNet

from signals.utils.dataset import load_wiener_hammerstein, DataSet
from signals.utils.figure import Figure, Subplot

# =============================================================================
# Global configuration
WH_PATH = 'data/wiener_hammerstein/whb.tfd'
VAL_SIZE = 20000
MODEL_MARK = 'wh_00'

NN_MEMORY_DEPTH = 5
NN_DEGREE = 3
NN_LEARNING_RATE = 0.001
NN_EPOCH = 3

FLAGS.train = False
PLOT = False
EVALUATION = True
# =============================================================================

# Load data set
train_set, val_set, test_set = load_wiener_hammerstein(
  WH_PATH, depth=NN_MEMORY_DEPTH)
assert isinstance(train_set, DataSet)
assert isinstance(val_set, DataSet)
assert isinstance(test_set, DataSet)

# Define model and identify
model = NeuralNet(NN_MEMORY_DEPTH, MODEL_MARK, NN_DEGREE)
model.nn.build(metric='ratio', metric_name='Err %',
               optimizer=tf.train.AdamOptimizer(NN_LEARNING_RATE))
if FLAGS.train:
  model.identify(train_set, val_set,
                 batch_size=50, print_cycle=100, epoch=NN_EPOCH)

# Evaluation
def evaluate(u, y):
  system_output = y[1000:]
  model_output = model(u)[1000:]
  delta = system_output - model_output

  console.supplement('E[err] = {:.4f}'.format(delta.average))
  console.supplement('STD[err] = {:.4f}'.format(float(np.std(delta))))
  console.supplement('RMS[err] = {:.4f}'.format(
    float(np.sqrt(np.mean(delta * delta)))))
  console.supplement('RMS[truth] = {:.4f}'.format(
    float(np.sqrt(np.mean(system_output * system_output)))))

  return system_output, model_output, delta

if EVALUATION:
  console.show_status('Evaluating estimation data ...')
  evaluate(train_set.signls[0], train_set.responses[0])
  console.show_status('Evaluating test data ...')
  system_output, model_output, delta = evaluate(
    test_set.signls[0], test_set.responses[0])

  if PLOT:
    fig = Figure('Simulation Error')
    # Add ground truth
    prefix = 'System Output, $||y|| = {:.4f}$'.format(system_output.norm)
    fig.add(Subplot.PowerSpectrum(system_output, prefix=prefix))
    # Add model output
    prefix = 'Model Output, $||\Delta|| = {:.4f}$'.format(delta.norm)
    fig.add(Subplot.PowerSpectrum(model_output, prefix=prefix, Delta=delta))
    # Plot
    fig.plot(ylim=True)



