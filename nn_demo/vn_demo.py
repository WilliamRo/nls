from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from signals.generator import gaussian_white_noise, multi_tone
from models import Volterra, Wiener, NeuralNet
from signals.utils import Figure, Subplot, DataSet

from tframe import FLAGS
from tframe import console

import model_lib


# =============================================================================
#  Global configuration
# =============================================================================
wiener_train = True
FLAGS.train = False
FLAGS.overwrite = True
bshow = False
nn_mark = 'vn_00'

# =============================================================================
#  Define system to be identify
# =============================================================================
# Define the black box using Volterra model
degree = 5
memory_depth = 3
system = Volterra(degree, memory_depth)

# region : Set kernels
system.set_kernel((0,), 1.0)
system.set_kernel((1,), 0.2)
system.set_kernel((2,), 0.1)
system.set_kernel((0, 0), 0.0)
system.set_kernel((1, 0), 0.0)
system.set_kernel((1, 1), 0.0)
system.set_kernel((2, 0), 0.0)
system.set_kernel((2, 1), 0.0)
system.set_kernel((2, 2), 0.0)
system.set_kernel((0, 0, 0), 0.0)
system.set_kernel((1, 0, 0), 0.0)
system.set_kernel((1, 1, 0), 0.0)
system.set_kernel((1, 1, 1), 0.0)
system.set_kernel((2, 0, 0), 0.0)
system.set_kernel((2, 1, 0), 0.0)
system.set_kernel((2, 1, 1), 0.0)
system.set_kernel((2, 2, 0), 0.0)
system.set_kernel((2, 2, 1), 0.0)
system.set_kernel((2, 2, 2), 0.0)
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
degree = 1
memory_depth = 3
vn = model_lib.vn_00(memory_depth, nn_mark, degree=None)

if FLAGS.train:
  vn.identify(
    train_set, val_set, probe=None,
    batch_size=50, print_cycle=100, epoch=1, snapshot_cycle=2000,
    snapshot_function=vn.gen_snapshot_function(signal, system_output))

# =============================================================================
#  Verification
# =============================================================================
print('>> System linear coefs = {}'.format(system.kernels.linear_coefs))
print('>> VN linear coefs = {}'.format(vn.nn.linear_coefs))
print('>> ||system_output|| = {:.4f}'.format(system_output.norm))

# System_2
# system_2 = Volterra(1, 3)
# system_2.set_kernel((0,), 0.01909399)
# system_2.set_kernel((1,), 0.8786813)
# system_2.set_kernel((2,), 0.74593288)
# system_2_output = system_2(signal)
# system_2_delta = system_output - system_2_output
# system_2_ratio = system_2_delta.norm / system_output.norm * 100
# print('>> System2 err ratio = {:.2f} %'.format(system_2_ratio))

# Wiener
wiener_output = wiener(signal)
wiener_delta = system_output - wiener_output
wiener_ratio = wiener_delta.norm / system_output.norm * 100
print('>> Wiener err ratio = {:.2f} %'.format(wiener_ratio))

# VN
mlp_output = vn(signal)
mlp_delta = system_output - mlp_output
mlp_ratio = mlp_delta.norm / system_output.norm * 100
print('>> VN err ratio = {:.2f} %'.format(mlp_ratio))

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
  prefix = 'VN Output, $||\Delta|| = {:.4f}$'.format(mlp_delta.norm)
  fig.add(Subplot.PowerSpectrum(mlp_output, prefix=prefix,
                                Delta=mlp_delta))
  fig.plot(ylim=(-75, 5))
