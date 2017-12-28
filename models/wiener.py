from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from models import Volterra
from signals import Signal


class Wiener(Volterra):
  """Volterra series in Wiener representation"""

  def __init__(self, degree, memory_depth, A=1):
    # Call parent's construction method
    Volterra.__init__(self, degree, memory_depth)
    self.set_kernel((), 0)

    # Initiate fields
    self.A = A

  # region : Public Methods

  def inference(self, input_):
    # Sanity check
    if not isinstance(input_, Signal):
      raise TypeError('!! Input must be an instance of Signal')

    y = np.zeros_like(input_)
    for n in range(self.degree + 1):
      y += self.G_n(n, input_)

    output = Signal(y)
    output.__array_finalize__(input_)
    return output

  def G_n(self, n, x):
    # Sanity check
    if n < 0:
      raise ValueError('!! degree of any Wiener operator must be non-negative')

    if n == 0: y_n = self.kernels[()] * np.ones_like(x)
    else:
      y_n = np.zeros_like(x)
      for i in range(n // 2 + 1):
        y_n += self.G_n_i(n, i, x)

    output = Signal(y_n)
    output.__array_finalize__(x)
    return output

  def G_n_i(self, n, i, x):
    y_i = np.zeros_like(x)

    # multiplicity is n - i
    indices_pool = self.kernels.get_homogeneous_indices(
      n - i, self.memory_depth[n - i - 1], symmetric=False)
    for indices in indices_pool:
      # Determine indices
      lags = indices + indices[slice(n - 2 * i, n - i)]
      x_lags = indices[slice(n - 2 * i)]

      prod = self.kernels[lags]
      if prod == 0: continue
      for lag in x_lags:
        prod *= self._delay(x, lag)
      y_i += prod

    output = Signal(y_i * self._get_coef(n, i))
    output.__array_finalize__(x)
    return output

  # endregion : Public Methods

  # region : Private Methods

  def _get_coef(self, n, i):
    # Verified
    return ((-1)**i * np.math.factorial(n) * self.A**i /
            (2**i * np.math.factorial(n - 2 * i) * np.math.factorial(i)))

  # endregion: Private Methods

  # region : Identification in Time Domain

  def cross_correlation(self, input_, output, intensity):
    x, y = input_, output
    assert isinstance(x, Signal) and isinstance(y, Signal)
    N = x.size
    self.A = intensity

    # Calculate k_0
    self.kernels[()] = input_.average

    # Calculate subsequent kernels
    for n in range(1, self.degree + 1):
      y -= self.G_n(n - 1, x)

      indices_pool = self.kernels.get_homogeneous_indices(
        n, self.memory_depth[n - 1])
      for lags in indices_pool:
        prod = np.copy(y)
        for lag in lags:
          prod *= self._delay(x, lag)
        self.kernels[lags] = float(np.average(prod)) / (
          np.math.factorial(n) * self.A**n)

  # endregion : Identification in Time Domain

  """For some reasons, do not delete this line."""


if __name__ == '__main__':
  model = Wiener(degree=4, memory_depth=3, A=1)

  model.set_kernel((), 3.6)

  model.set_kernel((0,), 0.2)
  model.set_kernel((1,), 1.5)
  model.set_kernel((2,), 0.7)

  model.set_kernel((0, 0), 2.8)
  model.set_kernel((1, 0), 3.3)
  model.set_kernel((1, 1), 1.5)
  model.set_kernel((2, 0), 0.3)
  model.set_kernel((2, 1), 1.0)
  model.set_kernel((2, 2), 0.9)

  model.set_kernel((0, 0, 0), 3.2)
  model.set_kernel((1, 0, 0), 0.4)
  model.set_kernel((1, 1, 0), 1.0)
  model.set_kernel((2, 2, 0), 1.5)
  model.set_kernel((2, 1, 1), 2.0)

  model.set_kernel((0, 0, 0, 0), 1.1)
  model.set_kernel((1, 1, 0, 0), 0.4)
  model.set_kernel((1, 1, 1, 0), 3.1)
  model.set_kernel((1, 1, 1, 1), 1.3)
  model.set_kernel((2, 1, 1, 1), 4.9)

  from signals.generator import gaussian_white_noise
  N = int(1e6)
  input_ = gaussian_white_noise(1, N, N)

  print('>> Properties of input noise: ')
  print('... E[X(t)] = {:.4f}, Var[X(t)] = {:.4f}'.format(
    input_.average, input_.variance))

  print()

  # for indices in model.indices_symmetric:
  #   print('... E[{}] = {:.3f}'.format(
  #     indices, input_.auto_correlation(indices)))

  # assert False

  G = {}
  for i in range(1, model.degree + 1):
    G[i] = model.G_n(i, input_)
    print('>> E[G_{}] = {}'.format(i, np.average(G[i])))

  print()

  for i in range(1, model.degree + 1):
    for j in range(i + 1, model.degree + 1):
      print('>> <G_{}, G_{}> = {}'.format(i, j, np.average(G[i] * G[j])))

  y = model.G_n(0, input_)
  for y_n in G.values(): y += y_n
  print('\n>> E[y(t)] = {}'.format(np.average(y)))


