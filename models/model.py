from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import operator as op
import functools

from signals import Signal


class Model(object):
  """Base class of all models"""

  def __init__(self):
    pass

  # region : Public Methods

  def build(self, *args, **kwargs):
    pass

  def train(self, *args, **kwargs):
    pass

  def inference(self, *args, **kwargs):
    pass

  # endregion : Public Methods

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

  '''For some reason, do not delete this line'''


if __name__ == '__main__':
  # Model.comb(np.arange(3), 2)
  print(Model.comb(1, 0))

  pass
