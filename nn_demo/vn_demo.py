from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from signals.generator import gaussian_white_noise, multi_tone
from models import Volterra, Wiener, NeuralNet
from signals.utils import Figure, Subplot, DataSet

from tframe import FLAGS
from tframe import console

import nn_model_lib


# =============================================================================
#  Global configuration
# =============================================================================
wiener_train = True
FLAGS.train = True
FLAGS.overwrite = True
bshow = False

# =============================================================================
#  Define system to be identify
# =============================================================================
# Define the black box using Volterra model
degree = 5
memory_depth = 3
system = Volterra(degree, memory_depth)
system.lock_orders([1, 2, 3])

# region : Set kernels
system.set_kernel((0,), 1.0)
system.set_kernel((1,), 2.4)
system.set_kernel((2,), 0.5)
system.set_kernel((0, 0), 0.1)
system.set_kernel((1, 0), 2.0)
system.set_kernel((1, 1), 1.2)
system.set_kernel((2, 0), 0.4)
system.set_kernel((2, 1), 3.1)
system.set_kernel((2, 2), 1.7)
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
# endregion : Set kernels

# =============================================================================
#  Identification
# =============================================================================

# region : Generate data sets
num = 5
A = 1
N = 100000
noises = []
noise_responses = []
for i in range(num):
  noise = gaussian_white_noise(A, N, N)
  noise_response = system(noise)
  noises.append(noise)
  noise_responses.append(noise_response)
train_set = DataSet(noises, noise_responses, memory_depth=3, intensity=A)
# endregion : Generate data sets

# Prepare test signal
freqs = [120, 310]
fs = 1000
duration = 2
vrms = [1, 0.7]
noise_power = 1e-3
signal = multi_tone(freqs, fs, duration, vrms, noise_power=noise_power)
system_output = system(signal)

val_set = DataSet(signal, system_output, memory_depth=3)

# Wiener
degree = 3
memory_depth = 3
wiener = Wiener(degree, memory_depth)
# wiener.cross_correlation(noises[0], noise_responses[0], A)
wiener.identify(train_set, val_set)

# VN
degree = 3
memory_depth = 3

homo_strs = []
for i in range(10): homo_strs.append(i * 0.05)
vns = collections.OrderedDict()
for homo_str in homo_strs:
  vn = nn_model_lib.vn_00(memory_depth, 'vn_{:.2f}'.format(homo_str),
                          degree=degree, homo_str=homo_str)
  vns[homo_str] = vn
  if FLAGS.train:
    vn.identify(
      train_set, val_set, probe=None,
      batch_size=50, print_cycle=100, epoch=2, snapshot_cycle=2000,
      snapshot_function=vn.gen_snapshot_function(signal, system_output))

# =============================================================================
#  Verification
# =============================================================================
print('>> ||system_output|| = {:.4f}'.format(system_output.norm))
print('>> System linear coefs = {}'.format(system.kernels.linear_coefs))
console.show_status('VN linear coefs:')
for strength, vn in vns.items():
  console.supplement('vn_{:.2f}: {}'.format(strength, vn.nn.linear_coefs))


# Wiener
wiener_output = wiener(signal)
wiener_delta = system_output - wiener_output
wiener_ratio = wiener_delta.norm / system_output.norm * 100
print('>> Wiener err ratio = {:.2f} %'.format(wiener_ratio))

# VN
console.show_status('VN error ratio:')
best_key, best_vn_ratio = None, 99999
best_delta, best_output = None, None
for strength, vn in vns.items():
  vn_output = vn(signal)
  vn_delta = system_output - vn_output
  vn_ratio = vn_delta.norm / system_output.norm * 100
  console.supplement('VN_{} err ratio = {:.2f} %'.format(strength, vn_ratio))
  if strength == 0 or vn_ratio < best_vn_ratio:
    best_key, best_vn_ra = strength, vn_ratio
    best_delta, best_output = vn_delta, vn_output

# Plot
if bshow:
  form_title = 'Input Frequencies = {}'.format(freqs)

  # Input & Output
  fig = Figure(form_title)
  fig.add(Subplot.PowerSpectrum(signal, prefix='System Input'))
  fig.add(Subplot.PowerSpectrum(system_output, prefix='System Output'))
  # fig.plot()

  # Compare outputs
  fig = Figure(form_title)
  # fig.add(Subplot.PowerSpectrum(signal, prefix='System Input'))
  prefix = 'System Output, $||y|| = {:.4f}$'.format(system_output.norm)
  fig.add(Subplot.PowerSpectrum(system_output, prefix=prefix))
  prefix = 'Wiener Output, $||\Delta|| = {:.4f}$'.format(wiener_delta.norm)
  fig.add(Subplot.PowerSpectrum(wiener_output, prefix=prefix,
                                Delta=wiener_delta))
  prefix = 'VN_{} Output, $||\Delta|| = {:.4f}$'.format(
    best_key, best_delta.norm)
  fig.add(Subplot.PowerSpectrum(best_output, prefix=prefix,
                                Delta=best_delta))
  fig.plot(ylim=True)
