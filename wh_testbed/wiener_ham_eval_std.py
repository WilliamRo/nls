import numpy as np
import tensorflow as tf

from tframe import FLAGS
from tframe import console

from models.neural_net import NeuralNet

from signals.utils.dataset import load_wiener_hammerstein, DataSet
from signals.utils.figure import Figure, Subplot

import model_lib

# =============================================================================
# Global configuration
WH_PATH = '../data/wiener_hammerstein/whb.tfd'
VAL_SIZE = 20000
MEMORY_DEPTH = 50
NN_EPOCH = 20
MODEL_MARK = 'mlp_D{}_4x2D'.format(MEMORY_DEPTH)
NN_TOL_EPC = 10
# NN_DEGREE = 4
# NN_LEARNING_RATE = 0.0001
# NN_MAX_VOL_ORD = 99

FLAGS.train = True
FLAGS.overwrite = True
FLAGS.save_best = True

# Turn off overwrite while in save best mode
FLAGS.overwrite = FLAGS.overwrite and not FLAGS.save_best

EVALUATION = False
PLOT = False

model = model_lib.mlp_00(MEMORY_DEPTH, MODEL_MARK)
# =============================================================================

# Load data set
train_set, val_set, test_set = load_wiener_hammerstein(
  WH_PATH, depth=MEMORY_DEPTH)
assert isinstance(train_set, DataSet)
assert isinstance(val_set, DataSet)
assert isinstance(test_set, DataSet)

# Define model and identify
if FLAGS.train: model.identify(
  train_set, val_set, batch_size=10, print_cycle=100, epoch=NN_EPOCH,
  tol_epoch=NN_TOL_EPC)

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



