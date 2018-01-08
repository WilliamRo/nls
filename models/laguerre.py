from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from models import Model
from signals import Signal


class Laguerre(Model):
  """A model for non-linear system based on the Laguerre orthogonal expansion of 
     Wiener series"""

  def __init__(self, alpha):
    # Sanity check
    if not 0 < alpha < 1:
      raise ValueError('!! Alpha must be between 0 and 1')

    # Initialize fields
    self.alpha = alpha

    # Call parent's construction methods
    Model.__init__(self)

  # region : Properties

  # endregion : Properties

  # region : Static Methods

  @staticmethod
  def phi_j(alpha, j, taus):
    assert 0 < alpha < 1
    assert int(j) == j >= 0
    j = int(j)

    coef = np.power(alpha, (taus - j) / 2) * np.sqrt(1 - alpha)
    y = np.zeros_like(taus, dtype=np.float32)
    for k in range(j + 1):
      y_k = np.zeros_like(taus, dtype=np.float32)
      for index, tau in enumerate(taus):
        if tau >= k: y_k[index] = Laguerre.comb(tau, k)
      y_k *= Laguerre.comb(j, k) * alpha**(j - k) * (1 - alpha)**k * (-1)**k
      y += y_k

    return coef * y

  @staticmethod
  def plot_laguerre(alphas=None, length=25, js=None):
    # Check inputs
    if alphas is None: alphas = [0.2, 0.6]
    if js is None: js = [0, 1, 2, 3, 4]
    t = np.arange(length + 1)

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

  """For some reason, do not delete this line"""


if __name__ == "__main__":
  from signals.utils import Figure, Subplot

  Laguerre.orthonormal_check(3)
  # Laguerre.plot_laguerre()






