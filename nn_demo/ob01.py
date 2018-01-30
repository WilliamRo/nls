from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

import tensorflow as tf

from tframe import FLAGS
from tframe import console
from tframe.layers import Linear, Activation

from models import Volterra, Wiener, NeuralNet
from signals.generator import gaussian_white_noise, multi_tone
from signals.utils import Figure, Subplot, DataSet


# =============================================================================
# Command Center
# =============================================================================
TRAIN = True
OVERWRITE = True
WIENER_ON = True
PLOT = False
# System parameters
SYS_DEGREE = 4
SYS_MEM_DEPTH = 5
SYS_LOCK_ORDERS = (1, 2, 3)
WEAK_NONLINEARITY = False
# Wiener parameters
WN_DEGREE = SYS_DEGREE if SYS_LOCK_ORDERS is None else max(SYS_LOCK_ORDERS)
WN_MEN_DEPTH = SYS_MEM_DEPTH
# Data parameters
NOISE_NUM = 5
NOISE_LEN = 100000
TEST_FREQS = [3160, 7080]
# Neural net parameters
HIDDEN_DIMS = [[40] * 4, [200] * 4]
NN_MEM_DEPTH = 5
NN_DEGREE = 3
NN_ORDERS = (1, 2, 3)
NN_HOMO_STRS = np.arange(1) * 0.05
NN_MAX_VOL_ORD = 3
POSTFIX = '{}'.format(NN_ORDERS)
# Train parameters
EPOCH = 1


def define_system():
  system = Volterra(SYS_DEGREE, SYS_MEM_DEPTH)
  if WEAK_NONLINEARITY:
    system.set_kernel((0,), 1.0)
    system.set_kernel((1,), -0.16)
    system.set_kernel((2,), 0.8)
    system.set_kernel((3,), -0.02)
    system.set_kernel((4,), 0.01)
    system.set_kernel((0, 0), 0.26)
    system.set_kernel((1, 0), -0.2)
    system.set_kernel((1, 1), 0.01)
    system.set_kernel((2, 0), -0.2)
    system.set_kernel((2, 1), 0.1)
    system.set_kernel((2, 2), -0.02)
    system.set_kernel((3, 0), 0.18)
    system.set_kernel((3, 2), -0.1)
    system.set_kernel((4, 1), 0.1)
    system.set_kernel((4, 2), 0.14)
    system.set_kernel((4, 2), -0.0)
    system.set_kernel((0, 0, 0), 0.06)
    system.set_kernel((1, 0, 0), 0.04)
    system.set_kernel((1, 1, 0), 0.03)
    system.set_kernel((1, 1, 1), 0.02)
    system.set_kernel((2, 0, 0), -0.03)
    system.set_kernel((2, 1, 0), 0.02)
    system.set_kernel((2, 1, 1), 0.0)
    system.set_kernel((2, 2, 0), 0.01)
    system.set_kernel((2, 2, 1), 0.02)
    system.set_kernel((2, 2, 2), 0.0)
    system.set_kernel((0, 0, 0, 0), 0.0)
    system.set_kernel((0, 0, 0, 0, 0), 0.0)
  else:
    system.set_kernel((0,), -1.0)
    system.set_kernel((1,), 2.4)
    system.set_kernel((2,), 0.5)
    system.set_kernel((3,), -3.5)
    system.set_kernel((4,), 1.6)
    system.set_kernel((0, 0), 0.1)
    system.set_kernel((1, 0), 2.0)
    system.set_kernel((1, 1), 1.2)
    system.set_kernel((2, 0), -0.4)
    system.set_kernel((2, 1), 3.1)
    system.set_kernel((2, 2), 1.7)
    system.set_kernel((3, 0), 2.1)
    system.set_kernel((3, 2), -1.2)
    system.set_kernel((4, 1), 1.5)
    system.set_kernel((4, 2), 1.3)
    system.set_kernel((4, 2), -2.0)
    system.set_kernel((0, 0, 0), 0.2)
    system.set_kernel((1, 0, 0), 1.2)
    system.set_kernel((1, 1, 0), 0.5)
    system.set_kernel((1, 1, 1), 0.2)
    system.set_kernel((2, 0, 0), 2.3)
    system.set_kernel((2, 1, 0), 0.3)
    system.set_kernel((2, 1, 1), 2.0)
    system.set_kernel((2, 2, 0), 0.1)
    system.set_kernel((2, 2, 1), 1.2)
    system.set_kernel((2, 2, 2), 3.0)
    system.set_kernel((0, 0, 0, 0), 0.0)
    system.set_kernel((0, 0, 0, 0, 0), 0.0)
  system.lock_orders(SYS_LOCK_ORDERS)
  return system

def generate_data(system):
  # Training set
  A = 1
  noises, noise_responses = [], []
  for _ in range(NOISE_NUM):
    noise = gaussian_white_noise(A, NOISE_LEN, NOISE_LEN)
    noises.append(noise)
    noise_responses.append(system(noise))
  training_set = DataSet(noises, noise_responses,
                         memory_depth=NN_MEM_DEPTH, intensity=A)
  # Validation set
  fs = 20000
  freq_pool = [[8000], [1200, 7690], [560, 1400, 8000],
               [3200, 4550, 6710, 8190], [1100, 3200, 5210, 7019, 7200]]
  duration = 2
  noise_power = 1e-3
  val_signals, val_responses = [], []
  for freqs in freq_pool:
    val_signal = multi_tone(freqs, fs, duration, noise_power=noise_power)
    val_signals.append(val_signal)
    val_responses.append(system(val_signal))
  validation_set = DataSet(val_signals, val_responses,
                           memory_depth=NN_MEM_DEPTH)
  # Test set
  test_signal = multi_tone(TEST_FREQS, fs, duration, noise_power=noise_power)
  test_response = system(test_signal)
  test_set = DataSet(test_signal, test_response, memory_depth=NN_MEM_DEPTH)
  return training_set, validation_set, test_set

def init_vn(mark, homo_str):
  D = NN_MEM_DEPTH
  hidden_dims = HIDDEN_DIMS

  degree = NN_DEGREE
  if degree is None: degree = len(hidden_dims) + 1
  elif degree < 1: raise ValueError('!! Degree must be greater than 1')

  activation = lambda: Activation('relu')
  learning_rate = 0.001
  reg = None

  # Initiate model
  model = NeuralNet(D, mark, degree=degree, orders=NN_ORDERS)

  for order in range(NN_MAX_VOL_ORD + 1, degree + 1):
    if order not in NN_ORDERS: continue
    dims = hidden_dims[order - NN_MAX_VOL_ORD - 1]
    for dim in dims:
      model.nn.add(order, Linear(dim, weight_regularizer='l2', strength=reg))
      model.nn.add(order, activation())
    model.nn.add(order, Linear(1, weight_regularizer='l2', strength=reg))

  # Build model
  model.nn.build(loss='euclid', metric='ratio', metric_name='Err %',
                 homo_strength=homo_str,
                 optimizer=tf.train.AdamOptimizer(learning_rate))
  return model

def verify(vns, wn, system, test_set):
  # Show linear coefficients
  console.show_status(
    'System linear coefs = {}'.format(system.kernels.linear_coefs))
  console.show_status('VN linear coefs:')
  for strength, vn in vns.items():
    console.supplement('vn_{:.2f}: {}'.format(strength, vn.nn.linear_coefs))
  # Generate system output
  signal, system_output = test_set.signls[0], test_set.responses[0]
  # Wiener error ratio
  wiener_output, wiener_delta = None, None
  if WIENER_ON:
    wiener_output = wn(signal)
    wiener_delta = system_output - wiener_output
    wiener_ratio = wiener_delta.norm / system_output.norm * 100
    console.show_status('Wiener err ratio = {:.2f} %'.format(wiener_ratio))
  # VN error ratio
  console.show_status('VN error ratio:')
  best_str, best_ratio, best_delta, best_output = None, 9999, None, None
  for strength, vn in vns.items():
    vn_output = vn(signal)
    vn_delta = system_output - vn_output
    vn_ratio = vn_delta.norm / system_output.norm * 100
    if strength == 0 or vn_ratio < best_ratio:
      best_str, best_ratio = strength, vn_ratio
      best_delta, best_output = vn_delta, vn_output
    console.supplement('VN_{:.2f} err ratio = {:.2f} %'.format(
      strength, vn_ratio))
  # Plot
  if PLOT: plot(system_output, wiener_output, wiener_delta,
                best_output, best_delta, best_str)

def homogeneous_check(model, order, input_, output):
  console.show_status('Checking homogeneous system')
  alpha = 2
  console.supplement('Alpha = {}'.format(alpha))
  truth = (alpha ** order) * output
  delta = model(alpha * input_) - truth
  ratio = delta.norm / truth.norm * 100
  console.supplement('Error Ratio = {:.2f} %'.format(ratio))

def plot(system_output, wiener_output, wiener_delta, vn_output, vn_delta,
          homo_str):
  form_title = 'Input Frequencies = {}'.format(TEST_FREQS)
  fig = Figure(form_title)
  # Add ground truth
  prefix = 'System Output, $||y|| = {:.4f}$'.format(system_output.norm)
  fig.add(Subplot.PowerSpectrum(system_output, prefix=prefix))
  # Add
  prefix = 'Wiener Output, $||\Delta|| = {:.4f}$'.format(wiener_delta.norm)
  fig.add(Subplot.PowerSpectrum(wiener_output, prefix=prefix,
                                Delta=wiener_delta))
  # Add
  prefix = 'VN_{:.2f} Output, $||\Delta|| = {:.4f}$'.format(
    homo_str, vn_delta.norm)
  fig.add(Subplot.PowerSpectrum(vn_output, prefix=prefix, Delta=vn_delta))
  fig.plot(ylim=True)


def main(_):
  console.suppress_logging()
  FLAGS.train = TRAIN
  FLAGS.overwrite = OVERWRITE
  console.start('EXP OB 01')
  # Define system
  system = define_system()
  # Generate data
  training_set, validation_set, test_set = generate_data(system)
  if len(SYS_LOCK_ORDERS) == 1:
    homogeneous_check(system, SYS_LOCK_ORDERS[0], training_set.signls[0],
                      training_set.responses[0])
  # Identification
  # .. wiener
  wiener = Wiener(degree=WN_DEGREE, memory_depth=WN_MEN_DEPTH)
  if WIENER_ON: wiener.identify(training_set, validation_set)
  # .. vn
  homo_strs = NN_HOMO_STRS
  vns = collections.OrderedDict()
  for homo_str in homo_strs:
    console.show_status('Volterra Net homo-strength = {:.2f}'.format(homo_str))
    vn = init_vn('vn_{:.2f}{}'.format(homo_str, POSTFIX), homo_str=homo_str)
    vns[homo_str] = vn
    if FLAGS.train:
      vn.identify(training_set, validation_set,
                  batch_size=50, print_cycle=100, epoch=EPOCH)
  # Verification
  verify(vns, wiener, system, test_set)
  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()
