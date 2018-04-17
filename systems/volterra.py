from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import math

from signals import Signal
from systems.system_ import System


class Volterra(System):
  """A system for non-linear system based on Volterra series"""

  def __init__(self):
    # Sanity check
    self.kernels = Kernels()
    # Call parent's construction methods
    System.__init__(self)

  # region : Properties

  # endregion : Properties

  # region : Public Methods

  def response(self, input_, **kwargs):
    # Calculate
    y = np.zeros_like(input_)
    pool = self.kernels.params.keys()
    for lags in pool:
      assert isinstance(lags, tuple)
      # lags = (\tau_1, \tau_2, \cdots, \tau_k)
      # prod = h_k(\tau_1, \cdots, \tau_k) * \prod_{i=1}^k x[n-\tau_i]
      prod = self.kernels[lags]
      for lag in lags: prod *= self._delay(input_, lag)
      y += prod

    output = Signal(y)
    output.__array_finalize__(input_)
    return output


  def set_kernel(self, index, value):
    self.kernels[index] = value

  # endregion : Public Methods

  # region : Private Methods

  @staticmethod
  def _delay(x, lag):
    result = np.append(np.zeros((lag,)), x)[:x.size]
    return result

  # endregion : Private Methods

  '''For some reason, do not delete this line'''


class Kernels(object):
  """Sparse Volterra kernel"""
  def __init__(self):
    self.params = collections.OrderedDict()

  # region : Operator Overloading

  def __getitem__(self, item):
    return self.params[tuple(sorted(item, reverse=True))]

  def __setitem__(self, key, value):
    self.params[tuple(sorted(key, reverse=True))] = value

  def __len__(self):
    return len(self.params)

  def __str__(self):
    knls = 'Kernels\n' + '-' * 79
    for lags, val in self.params.items():
      knls += '\nk{} = {:.4f}'.format(lags, val)
    return knls

  # endregion : Operator Overloading

  '''For some reason, do not delete this line'''


# region : Main Functions

def define_and_plot(*args, **kwargs):
  from signals.generator import multi_tone
  from signals.utils import Figure, Subplot

  # Initiate model
  model = Volterra()
  model.set_kernel((0,), 1.0)
  model.set_kernel((2,), 0.0)
  model.set_kernel((0, 0), 0.0)
  model.set_kernel((2, 2), 0.0)
  model.set_kernel((1, 0, 0), 0.2)
  model.set_kernel((2, 2, 2), 0.0)

  # Generate multi tone signal
  freqs = [160, 220]
  signal = multi_tone(freqs, 1000, 2, noise_power=1e-3)
  response = model(signal)

  # Plot
  title = 'Volterra Response, Input freqs = {}'.format(freqs)
  fig = Figure(title)
  fig.add(Subplot.PowerSpectrum(signal, prefix='Input Signal'))
  prefix = 'System Response'
  fig.add(Subplot.PowerSpectrum(response, prefix=prefix))
  fig.plot()


if __name__ == '__main__':
  define_and_plot()

# endregion : Main Functions
