# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# [P]erceptron [E]xpansion [T]echnique demo
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import numpy as np

from tframe import FLAGS
from tframe import console

from signals.utils.dataset import DataSet
from signals.utils.figure import Figure, Subplot

import svn_models
import data_utils

# =============================================================================
# Global configuration
MEMORY_DEPTH = 26
NN_HID_DIM = 6  # 6 by default
NN_ORDER = 4
NN_EPOCH = 100
MODEL_MARK = 'pet_h{}_o{}'.format(NN_HID_DIM, NN_ORDER)
FLAGS.epoch_tol = 20

NN_LEARN_RATE = 0.001
# NN_LEARN_RATE = 0.0001

FLAGS.train = True
# FLAGS.train = False

FLAGS.overwrite = True
# FLAGS.overwrite = False

FLAGS.save_best = False
# FLAGS.save_best = True

# Turn off overwrite while in save best mode
FLAGS.overwrite = FLAGS.overwrite and not FLAGS.save_best

EVALUATION = not FLAGS.train
PLOT = EVALUATION

# Initiate model
model = svn_models.pet(MEMORY_DEPTH, NN_HID_DIM, NN_ORDER,
                       NN_LEARN_RATE, MODEL_MARK)
# =============================================================================

# Load data set
train_set, val_set = data_utils.load_svn1997(MEMORY_DEPTH)
assert isinstance(train_set, DataSet)
assert isinstance(val_set, DataSet)

# Define model and identify
if FLAGS.train: model.identify(
  train_set, val_set, batch_size=1000, print_cycle=5, epoch=NN_EPOCH)

# Evaluation
def evaluate(u, y):
  system_output = y[5:]
  model_output = model(u)[5:]
  delta = system_output - model_output
  rms_truth = float(np.sqrt(np.mean(system_output * system_output)))

  val = delta.average
  pct = val / rms_truth * 100
  console.supplement('E[err] = {:.4f} ({:.3f}%)'.format(val, pct))

  val = float(np.std(delta))
  pct = val / rms_truth * 100
  console.supplement('STD[err] = {:.4f} ({:.3f}%)'.format(val, pct))

  val = float(np.sqrt(np.mean(delta * delta)))
  pct = val / rms_truth * 100
  console.supplement('RMS[err] = {:.4f} ({:.3f}%)'.format(val, pct))

  console.supplement('RMS[truth] = {:.4f}'.format(rms_truth))

  return system_output, model_output, delta

if EVALUATION:
  console.show_status('Evaluating validation data ...')
  system_output, model_output, delta = evaluate(
    val_set.signls[0], val_set.responses[0])

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


