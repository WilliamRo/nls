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

  def ortho_check(self, input_):
    print('>> Properties of input noise: ')
    print('... E[X(t)] = {:.4f}, Var[X(t)] = {:.4f}\n'.format(
      input_.average, input_.variance))

    G = {}
    for i in range(1, self.degree + 1):
      G[i] = self.G_n(i, input_)
      print('>> E[G_{}] = {:.4f}'.format(i, G[i].average))

    print()
    for i in range(1, self.degree + 1):
      for j in range(i + 1, self.degree + 1):
        print('>> <G_{}, G_{}> = {:.4f}'.format(
          i, j, float(np.average(G[i] * G[j]))))

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
      assert isinstance(indices, list) or isinstance(indices, tuple)
      # Determine indices
      lags = indices + indices[slice(n - 2 * i, n - i)]
      x_lags = indices[slice(n - 2 * i)]

      prod = self.kernels[lags]
      if prod == 0: continue

      for lag in x_lags: prod *= self._delay(x, lag)
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
    x, y = input_.copy(), output.copy()
    assert isinstance(x, Signal) and isinstance(y, Signal)
    self.A = intensity

    # Calculate k_0
    self.kernels[()] = output.average

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

  # region : Experiment

  @staticmethod
  def schetzen_1965(a=1, knls=None, A=1, length=5e5):
    # Check input
    if knls is None:
      knls = [2.2, 2.5, 0.2, 5.1, 2.8]
    assert isinstance(knls, list) or isinstance(knls, tuple)
    N = len(knls)
    length = int(length)

    # Define system using Volterra series
    volterra = Volterra(degree=2, memory_depth=N)
    for lags in volterra.kernels.get_homogeneous_indices(2, N):
      volterra.set_kernel(lags, knls[lags[0]] * knls[lags[1]] * a)

    def system(v):
      y = np.zeros_like(v)
      for tau, knl in enumerate(knls):
        y += volterra._delay(v, tau) * knl
      return a * y * y

    # Define model
    model = Wiener(degree=2, memory_depth=N)

    # Identification
    from signals.generator import gaussian_white_noise
    noise = gaussian_white_noise(A, length, length)
    output_v = volterra(noise)
    output_s = system(noise)

    model.cross_correlation(noise, output_s, A)

    print('>> delta = {:.5f}'.format(np.linalg.norm(output_s - output_v)))

    # Show delta
    max_err, max_knl = 0, 0
    for lags in volterra.kernels.get_homogeneous_indices(2, N):
      pred = model.kernels[lags]
      truth = volterra.kernels[lags]
      delta = np.abs(pred - truth)
      if abs(truth) > max_knl:
        max_knl = abs(truth)
        max_err = delta / pred * 100
      print('k{} = {:.3f}, truth = {:.3f} (delta = {:.3f})'.format(
        lags, pred, truth, delta))

    print('>> Max Err = {:.2f}%'.format(max_err))


  # endregion : Experiment

  """For some reasons, do not delete this line."""


if __name__ == '__main__':
  print('=' * 79)
  Wiener.schetzen_1965()
  print('-' * 79)

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

  model.ortho_check(input_)

