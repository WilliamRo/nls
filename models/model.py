from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

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

  '''For some reason, do not delete this line'''


if __name__ == '__main__':
  pass
