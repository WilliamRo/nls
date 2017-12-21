from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

from signals import Signal
from models import Model


class Volterra(Model):
  """A model for non-linear system based on Volterra series"""

  def __init__(self, degree, memory_depth):
    # Sanity check
    if degree <= 0:
      raise ValueError('degree must be positive')

    if isinstance(memory_depth, list) or isinstance(memory_depth, tuple):
      if len(memory_depth) != degree:
        raise ValueError('Memory depth for each degree should be specified')
      for depth in memory_depth:
        if depth < 0: raise ValueError('Memory depth must be non-negative')
    elif memory_depth < 0:
      raise ValueError('Memory depth must be non-negative')
    else:
      memory_depth = [memory_depth] * degree

    # Initialize fields
    self.degree = degree
    self.memory_depth = memory_depth
    self.kernel = Kernel(degree, memory_depth)

    # Call parent's construction methods
    Model.__init__(self)

  # region : Public Methods

  def inference(self, input_):
    if not isinstance(input_, Signal):
      raise TypeError('!! Input must be an instance of Signal')

    y = np.zeros_like(input_)
    for n in range(len(y)):
      # Calculate y[n]
      for lags, h in self.kernel.items:
        # lags = (\tau_1, \tau_2, \cdots, \tau_k)
        # prod = h_k(\tau_1, \cdots, \tau_k) * \prod_{i=1}^k x[n-\tau_i]
        prod = h
        for lag in lags:
          index = n - lag
          if index < 0:
            prod = 0
            break
          prod *= input_[index]
        # Add prod to y[n]
        y[n] += prod

    output = Signal(y)
    output.__array_finalize__(input_)
    return output

  # endregion : Public Methods

  # region : Operator Overloading

  def __call__(self, input_):
    return self.inference(input_)

  # endregion : Operator Overloading

  '''For some reason, do not delete this line'''


class Kernel(object):
  """Volterra kernel in symmetric form"""
  MAX_PARAMS_COUNT = int(3e7)  # 100~200MB Memory

  def __init__(self, degree, depth):
    assert isinstance(depth, list) or isinstance(depth, tuple)
    self.degree = degree
    self.depth = depth
    self.params = {}

    # Parameters count should be limited
    if self.params_count > Kernel.MAX_PARAMS_COUNT:
      raise ValueError('!! Too much parameters')

    # Initialize parameters
    for d in range(1, degree + 1):
      indices = Kernel.get_indices(d, depth[d - 1])
      for index in indices:
        self.params[index] = np.random.randn() / depth[d - 1] ** d
        # self.params[index] = 0

  # region : Properties

  @property
  def params_count(self):
    count = 0
    for d in range(1, self.degree + 1):
      count += Kernel.nCr(self.depth[d - 1] + d - 1, d)
    return count

  @property
  def items(self):
    return self.params.items()

  # endregion : Properties

  # region : Operator Overloading

  def __getitem__(self, item):
    return self.params[tuple(sorted(item, reverse=True))]

  def __len__(self):
    return len(self.params)

  def __str__(self):
    return "{}".format(self.params)

  # endregion : Operator Overloading

  # region : Static Methods

  @staticmethod
  def get_indices(degree, N):
    indices = []
    for i in range(N):
      if degree == 1: indices.append((i,))
      else:
        sub_indices = Kernel.get_indices(degree - 1, i + 1)
        for sub_index in sub_indices:
          indices.append((i,) + sub_index)

    return indices

  @staticmethod
  def nCr(n, r):
    f = math.factorial
    return int(f(n) / f(r) / f(n - r))

  # endregion : Static Methods

  '''For some reason, do not delete this line'''


if __name__ == '__main__':
  from signals.generator import multi_tone
  print(">> Running module volterra.py")

  fs = 2000
  duration = 1
  freqs = [600, 460, 120]
  vrms = [6, 4, 4]
  phases = None
  signal = multi_tone(freqs, fs, duration, vrms=vrms, phases=phases,
                      noise_power=1e-4)

  model = Volterra(degree=2, memory_depth=1)
  # model.kernel.params[(1, 0)] = 1
  # model.kernel.params[(1,)] = 1
  # model.kernel.params[(0,)] = 1
  output = model(signal)

  # print(model.kernel)
  # print('>> Delta = {}'.format(np.linalg.norm(signal - output)))

  from signals.utils import Figure, Subplot
  fig = Figure('Input & Output')
  db = True
  fig.add(Subplot.AmplitudeSpectrum(signal, prefix='Input Signal', db=db))
  fig.add(Subplot.AmplitudeSpectrum(output, prefix='Output Signal', db=db))
  fig.plot()

