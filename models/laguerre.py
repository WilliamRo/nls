from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from models import Model, Wiener
from models.volterra import Kernels

from signals import Signal


class Laguerre(Wiener):
  """A model for non-linear system based on the Laguerre orthogonal expansion of 
     Wiener series"""

  def __init__(self, alpha, degree, memory_depth, terms):
    # Sanity check
    if not 0 < alpha < 1:
      raise ValueError('!! Alpha must be between 0 and 1')

    # Call parent's construction methods
    Wiener.__init__(self, degree, memory_depth)

    # Initialize fields
    self.alpha = alpha
    self.N = max(self.memory_depth)
    self.terms = terms
    self.J = terms

    self.coefs = Kernels(degree, terms)
    self.coefs[()] = 0.0

    self.phi = np.zeros(shape=(terms, self.N), dtype=np.float64)
    self._shadow = None
    self._Phi = None

    # Initialize buffer
    self._init_buffer()

  # region : Properties

  # endregion : Properties

  # region : Public Methods

  def inference(self, input_, orders=None, *args, **kwargs):
    if not isinstance(input_, Signal):
      raise TypeError('!! Input must be an instance of Signal')

    # Update Phi
    self._update_Phi_naive(input_)

    # Calculate output
    y = self.coefs[()] * np.ones_like(input_)
    pool = (self.coefs.get_indices(symmetric=False) if orders is None
            else self.coefs.get_homogeneous_indices(
      orders, self.memory_depth[orders + 1], symmetric=False))
    for indices in pool:
      y_ = self.coefs[indices] * np.ones_like(input_)
      for index in indices: y_ *= self.Phi[index]
      y += y_

    output = Signal(y)
    output.__array_finalize__(input_)
    return output

  # endregion : Public Methods

  # region : Static Methods

  @staticmethod
  def phi_j(alpha, j, taus):
    assert 0 < alpha < 1
    assert int(j) == j >= 0
    j = int(j)

    coef = np.power(alpha, (taus - j) / 2) * np.sqrt(1 - alpha)
    y = np.zeros_like(taus, dtype=np.float64)
    for k in range(j + 1):
      y_k = np.zeros_like(taus, dtype=np.float64)
      for index, tau in enumerate(taus):
        if tau >= k: y_k[index] = Laguerre.comb(tau, k)
      y_k *= Laguerre.comb(j, k) * alpha**(j - k) * (1 - alpha)**k * (-1)**k
      y += y_k

    return coef * y

  @staticmethod
  def plot_laguerre(alphas=None, lags=25, js=None):
    # Check inputs
    if alphas is None: alphas = [0.2, 0.6]
    if js is None: js = [0, 1, 2, 3, 4]
    t = np.arange(lags + 1)

    if not isinstance(alphas, list) and not isinstance(alphas, tuple):
      alphas = [alphas]
    for alpha in alphas:
      if not np.isscalar(alpha) or not 0 < alpha < 1:
        raise ValueError('!! alpha must be a real number between 0 and 1')

    for point in t:
      if not point == int(point) >= 0:
        raise TypeError('!! Each point in t must be a non-negative integer')

    # Plot
    from signals.utils import Figure, Subplot
    fig = Figure('Discrete Orthogonal Laguerre Functions')
    title = (r"Laguerre bases $\{\phi_j[n;\alpha]\}$")
    for alpha in alphas:
      ys = []
      legends = []
      for j in js:
        ys.append(Laguerre.phi_j(alpha, j, t))
        legends.append(r'$j = {}$'.format(j))
      fig.add(Subplot.Default(t, ys, legends=legends, xlabel='Time Unit',
                              title=title + r', $\alpha = {}$'.format(alpha)))
    fig.plot()

  @staticmethod
  def plot_laguerre_3D(alpha=0.2, L=50, contour=True):
    from signals.utils import Figure, Subplot
    import matplotlib.pyplot as plt

    t = np.arange(L + 1)
    # x, y = np.meshgrid(t, t)
    z = np.zeros(shape=(L + 1, L + 1))
    for j in range(L + 1): z[j] = Laguerre.phi_j(alpha, j, t)

    def plot_function(*args):
      if contour:
        plt.contour(z)
        plt.xlabel('Time Units')
        plt.ylabel('Order of Laguerre function')
      else: raise NotImplementedError('!! 3D not implemented')

    fig = Figure('Discrete Laguerre Functions 3D')
    fig.add(Subplot.Custom(plot_function))
    fig.plot()

  @staticmethod
  def orthonormal_check(max_order=4, alpha=0.5, length=50):
    """<\phi_i, \phi_j> = \sum_{n} phi_i[n] * phi_j[n]"""
    ns = np.arange(length + 1)
    pool = {}
    for i in range(max_order + 1):
      pool[i] = Laguerre.phi_j(alpha, i, ns)

    for i in range(max_order + 1):
      for j in range(i + 1):
        products = np.sum(pool[i] * pool[j])
        print(':: <L_{}, L_{}> = {:.4f}'.format(i, j, products))

  # endregion : Static Methods

  # region : Identification in Time Domain

  def brutal_force(self, input_, output, intensity):
    x, y = input_.copy(), output.copy()
    assert isinstance(x, Signal) and isinstance(y, Signal)
    self.A = intensity

  def cross_correlation(self, input_, output, intensity):
    x, y = input_.copy(), output.copy()
    assert isinstance(x, Signal) and isinstance(y, Signal)
    self.A = intensity

    # Update Phi
    self._update_Phi_naive(input_)

    # Calculate k_0
    self.coefs[()] = output.average

    # Calculate subsequent coefficients
    for indices in self.coefs.get_indices(symmetric=True):
      n = len(indices)
      self.coefs[indices] = float(np.average(self._Q_n(indices) * output)) / (
        np.math.factorial(n) * self.A**n)

  def _Q_n(self, indices):
    assert self._shadow is not None
    n = len(indices)
    y_n = np.zeros_like(self._shadow)
    for i in range(n // 2 + 1):
      y_n += self._Q_n_i(i, indices)
    return y_n

  def _Q_n_i(self, i, indices):
    assert self._shadow is not None
    n = len(indices)
    for j in range(i):
      if indices[n - 2*i + 2*j] != indices[n - 2*i + 2*j + 1]: return 0

    y_n_i = np.ones_like(self._shadow)
    for j in range(n - 2*i):
      y_n_i *= self.Phi[indices[j]]

    return y_n_i * self._get_coef(n, i)

  # endregion : Identification in Time Domain

  # region : Private Methods

  def _init_buffer(self):
    taus = np.arange(self.N)
    for j in range(self.J):
      self.phi[j] = Laguerre.phi_j(self.alpha, j, taus)


  def _update_Phi_naive(self, input_):
    if self._shadow is input_: return
    self.tic()
    self._shadow = input_

    offset = self.N - 1
    x = np.append(np.zeros((offset,)), input_)
    Phi = np.zeros(shape=(self.J, input_.size), dtype=np.float64)
    for j in range(self.terms):
      for n in range(input_.size):
        x_flip = np.flip(x[offset + n - (self.N - 1):offset + n + 1], axis=0)
        Phi[j, n] = np.sum(self.phi[j] * x_flip)

    self.logs['naive_time'] = self.toc()

    self.Phi = Phi
    return Phi


  def _update_Phi_recursive(self, input_):
    """The recursive equation for Phi[j] (j > 0) seems to be erroneous"""
    if self._shadow is input_: return
    self.tic()
    self._shadow = input_

    Phi = np.zeros(shape=(self.J, input_.size), dtype=np.float64)

    # Initialize \Phi_0
    Phi[0][0] = input_[0] * self.phi[0][0]
    sqrt_alpha = np.sqrt(self.alpha)
    sqrt_op_alpha = np.sqrt(1 - self.alpha)
    for n in range(1, input_.size):
      tail_index = n - self.N
      tail = 0 if tail_index < 0 else sqrt_alpha**self.N * input_[tail_index]
      Phi[0][n] = (sqrt_alpha * Phi[0][n - 1] + sqrt_op_alpha * input_[n]
                   - tail * sqrt_op_alpha)

    # Calculate \Phi_{1:J} recursively
    for j in range(1, self.J):
      Phi[j][0] = input_[0] * self.phi[j][0]
      for n in range(1, input_.size):
        Phi[j][n] = (np.sqrt(self.alpha) * Phi[j][n - 1] +
                     np.sqrt(self.alpha) * Phi[j - 1][n] - Phi[j - 1][n - 1])

    self.logs['recur_time'] = self.toc()
    self.Phi = Phi
    return Phi


  # endregion : Private Methods

  """For some reason, do not delete this line"""


if __name__ == "__main__":
  from signals.utils import Figure, Subplot
  from signals.generator import gaussian_white_noise

  # Laguerre.orthonormal_check(3)
  # Laguerre.plot_laguerre(js=[4, 8, 12, 16], lags=50)
  model = Laguerre(alpha=0.5, degree=3, memory_depth=50, terms=5)

  noise = gaussian_white_noise(1, 1000, 1000)

  # print(model._get_coef(1, 0))



  # Phi_naive = model._update_Phi_naive(noise)
  # Phi_recur = model._update_Phi_recursive(noise)

  # delta = np.linalg.norm(Phi_naive - Phi_recur)
  # print('>> Naive update time cost = {:.3f} secs'.format(
  #   model.logs['naive_time']))
  # print('>> Recursive update time cost = {:.3f} secs'.format(
  #   model.logs['recur_time']))
  # print('>> delta = {}'.format(delta))








