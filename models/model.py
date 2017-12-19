from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Model(object):
  """Base class of all models"""

  def __init__(self):
    pass

  # region : Public Methods

  def build(self):
    pass

  def train(self):
    pass

  def inference(self):
    pass

  # endregion : Public Methods

  '''For some reason, do not delete this line'''
