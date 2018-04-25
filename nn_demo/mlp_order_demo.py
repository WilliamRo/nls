from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
mlp_mark = 'mlp_od_12(1)'
vn_mark = 'vn_od_12(1)'

# =============================================================================
#  Define system to be identify
# =============================================================================
# Define the black box using Volterra model
degree = 3
memory_depth = 3
system = Volterra(degree, memory_depth)
system.lock_orders([1, 2])

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
# endregion : Set kernels

# =============================================================================
#  Identification
# =============================================================================

# region : Generate data sets
num = 2
A = 1
N = 50000
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

# MLP
mlp = nn_model_lib.mlp_00(memory_depth, mlp_mark)

if FLAGS.train:
  mlp.identify(train_set, val_set, batch_size=50,
               print_cycle=100, epoch=1, snapshot_cycle=2000,
               snapshot_function=mlp.gen_snapshot_function(
                 signal, system_output))

# VN
vn = nn_model_lib.vn_00(memory_depth, vn_mark)
if FLAGS.train:
  vn.identify(train_set, val_set, batch_size=50,
              print_cycle=100, epoch=1, snapshot_cycle=2000,
              snapshot_function=mlp.gen_snapshot_function(
                signal, system_output))

# =============================================================================
#  Verification
# =============================================================================
print('>> ||system_output|| = {:.4f}'.format(system_output.norm))
# Wiener
wiener_output = wiener(signal)
wiener_delta = system_output - wiener_output
wiener_ratio = wiener_delta.norm / system_output.norm * 100
print('>> Wiener err ratio = {:.2f} %'.format(wiener_ratio))

# MLP
mlp_output = mlp(signal)
mlp_delta = system_output - mlp_output
mlp_ratio = mlp_delta.norm / system_output.norm * 100
print('>> MLP err ratio = {:.2f} %'.format(mlp_ratio))

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
  prefix = 'MLP Output, $||\Delta|| = {:.4f}$'.format(mlp_delta.norm)
  fig.add(Subplot.PowerSpectrum(mlp_output, prefix=prefix,
                                Delta=mlp_delta))
  fig.plot(ylim=(-75, 5))
