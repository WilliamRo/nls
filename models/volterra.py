from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import math

from signals import Signal
from models import Model


class Volterra(Model):
  """A model for non-linear system based on Volterra series"""

  def __init__(self, degree, memory_depth):
    # Sanity check
    memory_depth = Model._check_degree_and_depth(degree, memory_depth)

    # Initialize fields
    self.degree = degree
    self.memory_depth = memory_depth
    self.kernels = Kernels(degree, memory_depth)

    self._shadow = None
    self._buffer = None

    # Call parent's construction methods
    Model.__init__(self)

  # region : Properties

  @property
  def indices_full(self):
    return self.kernels.get_indices(symmetric=False)

  @property
  def indices_symmetric(self):
    return self.kernels.get_indices(symmetric=True)

  # endregion : Properties

  # region : Public Methods

  def inference(self, input_):
    if not isinstance(input_, Signal):
      raise TypeError('!! Input must be an instance of Signal')

    y = np.zeros_like(input_)
    for lags in self.indices_full:
      # lags = (\tau_1, \tau_2, \cdots, \tau_k)
      # prod = h_k(\tau_1, \cdots, \tau_k) * \prod_{i=1}^k x[n-\tau_i]
      prod = self.kernels[lags]
      if prod == 0: continue

      for lag in lags: prod *= self._delay(input_, lag)
      y += prod

    output = Signal(y)
    output.__array_finalize__(input_)
    return output


  def set_kernel(self, index, value):
    self.kernels[index] = value

  # endregion : Public Methods

  # region : Private Methods

  def _delay(self, x, lag):
    if self._shadow is not x:
      self._shadow = x
      max_lag = max(self.memory_depth)
      assert max_lag > 0

      self._buffer = []
      for tau in range(max_lag):
        self._buffer.append(np.append(np.zeros((tau,)), x)[:x.size])

    return self._buffer[lag]

  # endregion : Private Methods

  # region : Identification in Time Domain

  def cross_correlation(self, input_, output, intensity):
    """Identification using Gaussian white noise excitation
       Example: (given system to be identify)
         model Volterra(degree, memory_depth)
         
         intensity = 2
         length = 1000000
         fs = length
         input_ = gaussian_white_noise(intensity, length, fs)
         output = system(input_)
         
         model.cross_correlation(input_, output, intensity)
    """
    # Sanity check
    if input_.size != output.size:
      raise ValueError(
        '!! Length of input_ {} does not match length of ' 'output {}'.format(
          input_.size, output.size))

    if self.degree > 2:
      raise ValueError('!! Currently this method is only applied with degree'
                       ' less than 3')
    # Preparation
    N = input_.size
    A = intensity

    # For h_1(\tau)
    for tau in range(self.memory_depth[0]):
      xy = np.sum(output[tau:] * input_[:N-tau], dtype=np.float32)
      param = float(xy) / (N - tau) / A
      self.kernels.params[(tau,)] = param

    # For h_2(\tau_1, \tau_2)
    depth = self.memory_depth[1]
    Es = np.zeros(shape=(depth, 1), dtype=np.float32)
    for tau_1, tau_2 in Kernels.get_homogeneous_indices(2, depth):
      max_tau = max(tau_1, tau_2)
      x_1 = input_[(max_tau - tau_1):(N - tau_1)]
      x_2 = input_[(max_tau - tau_2):(N - tau_2)]
      Eyxx = float(np.sum(output[max_tau:] * x_1 * x_2)) / (N - max_tau)
      if tau_1 == tau_2:
        Es[tau_1] = Eyxx
        continue
      self.kernels.params[(tau_1, tau_2)] = Eyxx / (2 * A * A)

    # Solve the diagonal values
    coefs = np.ones(shape=(depth, depth), dtype=np.float32)
    for i in range(depth): coefs[i, i] = 3.
    diags = np.matmul(np.linalg.inv(A * A * coefs), Es)
    for i in range(depth):
      self.kernels.params[(i, i)] = diags[i, 0]

  # endregion : Identification in Time Domain

  # region : Identification in Frequency Domain

  # endregion : Identification in Frequency Domain

  # region : Operator Overloading

  def __call__(self, input_):
    return self.inference(input_)

  # endregion : Operator Overloading

  '''For some reason, do not delete this line'''


class Kernels(object):
  """Volterra kernel in symmetric form"""
  MAX_PARAMS_COUNT = int(3e7)  # 100~200MB Memory

  def __init__(self, degree, depth):
    if not isinstance(depth, list) or isinstance(depth, tuple):
      depth = [depth] * degree
    self.degree = degree
    self.depth = depth
    self.params = {}

    # Parameters count should be limited
    if self.params_count > Kernels.MAX_PARAMS_COUNT:
      raise ValueError('!! Too much parameters')

    # Initialize parameters
    for d in range(1, degree + 1):
      indices = Kernels.get_homogeneous_indices(d, depth[d - 1])
      for index in indices:
        self.params[index] = 0

  # region : Properties

  @property
  def params_count(self):
    count = 0
    for d in range(1, self.degree + 1):
      count += Model.comb(self.depth[d - 1] + d - 1, d)
    return count

  @property
  def items(self):
    return self.ordered_dict.items()

  @property
  def ordered_dict(self):
    od = collections.OrderedDict()
    if self.params.get((), None) is not None: od[()] = self.params[()]
    for d in range(1, self.degree + 1):
      indices = Kernels.get_homogeneous_indices(d, self.depth[d - 1])
      for index in indices:
        od[index] = self.params[index]
    return od

  # endregion : Properties

  # region : Operator Overloading

  def __getitem__(self, item):
    return self.params[tuple(sorted(item, reverse=True))]

  def __setitem__(self, key, value):
    self.params[tuple(sorted(key, reverse=True))] = value

  def __len__(self):
    return len(self.params)

  def __str__(self):
    knls = 'Kernels\n' + '-' * 79
    for lags, val in self.ordered_dict.items():
      knls += '\nk{} = {:.4f}'.format(lags, val)
    return knls

  # endregion : Operator Overloading

  # region : Public Methods

  def get_indices(self, symmetric=False):
    results = []

    for d in range(1, self.degree + 1):
      results += self.get_homogeneous_indices(d, self.depth[d - 1], symmetric)

    return results

  # endregion : Public Methods

  # region : Static Methods

  @staticmethod
  def get_homogeneous_indices(degree, N, symmetric=True):
    indices = []
    for i in range(N):
      if degree == 1: indices.append((i,))
      else:
        sub_indices = Kernels.get_homogeneous_indices(
          degree - 1, (i + 1) if symmetric else N, symmetric)
        for sub_index in sub_indices:
          indices.append((i,) + sub_index)

    return indices

  # endregion : Static Methods

  '''For some reason, do not delete this line'''


if __name__ == '__main__':
  from signals.generator import multi_tone
  print(">> Running module volterra.py")

  fs = 2000
  duration = 1
  freqs = [500, 800]
  vrms = [2, 1]
  phases = None
  signal = multi_tone(freqs, fs, duration, vrms=vrms, phases=phases,
                      noise_power=1e-4)

  model = Volterra(degree=2, memory_depth=1)
  model.kernels.params[(1, 0)] = 1
  model.kernels.params[(1,)] = 1
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

