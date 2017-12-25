import numpy as np

from signals.generator import gaussian_white_noise
from signals.generator import multi_tone
from models import Volterra
from signals.utils import Figure, Subplot


# ==============================================================================
#  Global configuration
# ==============================================================================
pass

# ==============================================================================
#  Get systems to identify
# ==============================================================================
systems = []

system = Volterra(degree=2, memory_depth=3)
system.kernels.params[(0,)] = 1
# system.kernels.params[(1,)] = 1.6
# system.kernels.params[(2,)] = 2.7
system.kernels.params[(0, 0)] = 1.2
system.kernels.params[(1, 0)] = 3.1
system.kernels.params[(1, 1)] = 1.9
system.kernels.params[(2, 0)] = 5.9
system.kernels.params[(2, 1)] = 1.5
system.kernels.params[(2, 2)] = 2.0

systems.append(system)

# ==============================================================================
#  Generate signals for verification
# ==============================================================================
signals = []
signals.append(gaussian_white_noise(3, 10000, 10000))
signals.append(multi_tone([300, 500], 1500, 2, noise_power=1e-3))

# ==============================================================================
#  Identification
# ==============================================================================
print(">> Identifying ...")
models = []

model = Volterra(degree=2, memory_depth=3)
A, N = 2, 50000
input_ = gaussian_white_noise(A, N, N)
output = system(input_)
model.cross_correlation(input_, output, A)

models.append(model)

# ==============================================================================
#  Test
# ==============================================================================
for i, system in enumerate(systems):
  print('=' * 79)
  print('  System [{}]'.format(i))
  print('=' * 79)
  model = models[i]

  for j, signl in enumerate(signals):
    system_output = system(signl)
    model_output = model(signl)
    delta = np.linalg.norm(system_output - model_output) / signl.size
    fig = Figure('System[{}], signal[{}]: Delta = {}'.format(i, j, delta))
    fig.add(Subplot.PowerSpectrum(signl, prefix='Input'))
    fig.add(Subplot.PowerSpectrum(system_output, prefix='System Output'))
    fig.add(Subplot.PowerSpectrum(model_output, prefix='Model Output'))
    fig.plot()



