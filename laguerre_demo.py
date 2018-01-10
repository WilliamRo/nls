from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from signals.generator import gaussian_white_noise, multi_tone
from models import Laguerre, Wiener, Volterra
from signals.utils import Figure, Subplot


# Define black box
a = 1
knls = [2.2, 2.5, 0.2, 5.1, 2.8]
A = 1
length = 10000
N = len(knls)
system = Volterra(degree=2, memory_depth=N)
for lags in system.kernels.get_homogeneous_indices(2, N):
  system.set_kernel(lags, knls[lags[0]] * knls[lags[1]] * a)

# =============================================================================
#  Identification
# =============================================================================
# Prepare Gaussian white noise
noise = gaussian_white_noise(A, length, length)
noise_response = system(noise)

# Wiener
degree = 2
memory_depth = N
wiener = Wiener(degree, memory_depth)
wiener.cross_correlation(noise, noise_response, A)

# Laguerre
alpha = 0.05
degree = 2
memory_depth = N
terms = N
laguerre = Laguerre(alpha, degree, memory_depth, terms)
laguerre.cross_correlation(noise, noise_response, A)

# =============================================================================
#  Verification
# =============================================================================
# Prepare test signal
test_signal = multi_tone([120, 310], 1000, 2, noise_power=1e-4)
system_output = system(test_signal)

# Wiener
wiener_output = wiener(test_signal)
wiener_delta = np.linalg.norm(
  wiener_output - system_output) / test_signal.size
print('>> Wiener Delta = {:.4f}'.format(wiener_delta))

# Laguerre
laguerre_output = laguerre(test_signal)
laguerre_delta = np.linalg.norm(
  laguerre_output - system_output) / test_signal.size
print('>> Laguerre Delta = {:.4f}'.format(laguerre_delta))

# Plot
fig = Figure('Laguerre Demo')
fig.add(Subplot.PowerSpectrum(test_signal, prefix='Input Signal'))
fig.add(Subplot.PowerSpectrum(system_output, prefix='System Output'))
fig.add(Subplot.PowerSpectrum(laguerre_output, prefix='Model Output'))
fig.plot()
