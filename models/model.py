from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import operator as op
import functools

import pickle

from signals import Signal


class Model(object):
  """Base class of all models"""
  extension = '.mdl'

  def __init__(self):
    self.logs = {}
    self.time_point = None

  # region : Public Methods

  # region : Utilities

  def tic(self):
    self.time_point = time.time()

  def toc(self):
    if self.time_point is None: return 0
    return time.time() - self.time_point

  def save(self, filename):
    with open(filename + self.extension, 'wb') as output:
      pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

  @staticmethod
  def load(filename):
    with open(filename, 'rb') as input_:
      return pickle.load(input_)

  # endregion : Utilities

  def separate_interp(
      self, input_, alphas=None, max_order=4, verbose=False):

    # Sanity check
    if not isinstance(input_, Signal):
      raise TypeError('!! Input must be an instance of Signal')
    if alphas is None:
      alphas = []
      for order in range(max_order):
        alphas.append((-1)**order * (1 + order * 0.25))
    if verbose: print('>> alphas = {}'.format(alphas))

    N = len(alphas)

    # Generate matrix R
    R = np.zeros(shape=(N, input_.size))
    for i in range(N): R[i] = self(alphas[i] * input_)

    # Generate matrix A
    A = np.zeros(shape=(N, N))
    for r in range(N):
      for c in range(N): A[r, c] = alphas[r] ** (c + 1)
    inv_A = np.linalg.inv(A)
    if verbose: print('>> ||inv(A)|| = {:.4f}'.format(np.linalg.norm(inv_A)))

    # Separate
    results = []
    yn = np.matmul(inv_A, R)
    for order in range(N):
      results.append(Signal(yn[order]))
      results[order].__array_finalize__(input_)

    return results

  def separate(self, input_, max_order):
    results = []
    for n in range(1, max_order + 1):
      results.append(self.inference(input_, n))
    return results

  # endregion : Public Methods

  # region : Abstract Methods

  def inference(self, *args, **kwargs):
    raise NotImplementedError('!! Method not implemented')

  # endregion : Abstract Methods

  # region : Static Methods

  @staticmethod
  def comb(n, r):
    if not np.isscalar(n) or not np.isscalar(r):
      raise TypeError('!! n and r should be scalars')

    r = min(r, n - r)
    if r == 0: return 1
    reduce = functools.reduce
    numer = reduce(op.mul, range(n, n - r, -1))
    denom = reduce(op.mul, range(1, r + 1))

    return numer // denom

  # endregion : Static Methods

  # region : Operator Overloading

  def __call__(self, input_, *args, **kwargs):
    return self.inference(input_, *args, **kwargs)

  # endregion : Operator Overloading

  # region : Private Methods

  @staticmethod
  def _check_degree_and_depth(degree, memory_depth):
    if isinstance(memory_depth, list) or isinstance(memory_depth, tuple):
      if len(memory_depth) != degree:
        raise ValueError('Memory depth for each degree should be specified')
      for depth in memory_depth:
        if depth < 0: raise ValueError('Memory depth must be non-negative')
    elif memory_depth < 0:
      raise ValueError('Memory depth must be non-negative')
    else:
      memory_depth = [memory_depth] * degree

    return memory_depth

  # endregion : Private Methods

  '''For some reason, do not delete this line'''


if __name__ == '__main__':
  # Model.comb(np.arange(3), 2)
  print(Model.comb(1, 0))

  pass
