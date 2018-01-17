from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from signals.generator import gaussian_white_noise, multi_tone
from models import Volterra, Wiener, NeuralNet
from signals.utils import Figure, Subplot

from tframe import FLAGS
from tframe import console

import models


# =============================================================================
#  Define system to be identify
# =============================================================================
# Define the black box using Volterra model
degree = 3
memory_depth = 3
system = Volterra(degree, memory_depth)

# region : Set kernels
system.set_kernel((0,), 1.0)
system.set_kernel((1,), 0.0)
system.set_kernel((2,), 0.0)
system.set_kernel((0, 0), 0.1)
system.set_kernel((1, 0), 0.0)
system.set_kernel((1, 1), 0.0)
system.set_kernel((2, 0), 0.1)
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
# endregion : Set kernels

# =============================================================================
#  Identification
# =============================================================================
# TFrame configuration
FLAGS.train = False
FLAGS.overwrite = True

# Prepare Gaussian white noise
A = 1
N = 100000
noise = gaussian_white_noise(A, N, N)
noise_response = system(noise)

val_A = 1
val_N = 10000
val_noise = gaussian_white_noise(val_A, val_N, val_N)
val_response = system(val_noise)

# Prepare test signal
freqs = [120, 310]
fs = 1000
duration = 2
vrms = [1, 0.7]
noise_power = 1e-3
signal = multi_tone(freqs, fs, duration, vrms, noise_power=noise_power)
system_output = system(signal)

# Wiener
degree = 2
memory_depth = 3
wiener = Wiener(degree, memory_depth)
wiener.cross_correlation(noise, noise_response, A)

# MLP
mlp = NeuralNet(memory_depth=memory_depth, hidden_dims=[10, 10],
                build_default=True, mark='mlp_10-10')
if FLAGS.train:
  mlp.identify(noise, noise_response, val_noise, val_response, batch_size=50,
               print_cycle=100, epoch=10, snapshot_cycle=500,
               snapshot_function=mlp.gen_snapshot_function(
                 signal, system_output))

# =============================================================================
#  Verification
# =============================================================================
print('>> ||system_output|| = {:.4f}'.format(system_output.norm))
# Wiener
wiener_output = wiener(signal)
wiener_delta = system_output - wiener_output
print('>> ||wiener_delta|| = {:.4f}'.format(wiener_delta.norm))

# MLP
mlp_output = mlp(signal)
mlp_delta = system_output - mlp_output
print('>> ||mlp_delta|| = {:.4f}'.format(mlp_delta.norm))

# Plot
bshow = True
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
