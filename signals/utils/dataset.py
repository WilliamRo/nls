from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe import TFData

import signals


class DataSet(TFData):
  """A dataset class for signals based on TFData"""

  def __init__(self, signls, responses=None, name=None, memory_depth=1,
               intensity=None):
    # Sanity check
    if not isinstance(signls, (tuple, list)): signls = [signls]
    for signl in signls:
      if not isinstance(signl, signals.Signal):
        raise TypeError('!! Each item in signals must be a Signal')

    if responses is not None:
      if not isinstance(responses, (tuple, list)): responses = [responses]
      for resp in responses:
        if not isinstance(resp, signals.Signal):
          raise TypeError('!! Responses must be Signals')
        if len(responses) != len(signls):
          raise ValueError('!! signls and responses must have the same len')

    # Initiate fields
    self.signls = signls
    self.responses = responses
    self.intensity = intensity
    self.name = name

    # Call parent's constructor
    self.init_tfdata(memory_depth)

  # region : Public Methods

  def init_tfdata(self, memory_depth):
    features = self.signls[0].causal_matrix(memory_depth)
    for i in range(1, len(self.signls)):
      features = np.vstack(
        (features, self.signls[i].causal_matrix(memory_depth)))
    targets = None
    if self.responses is not None:
      targets = self.responses[0].reshape(self.responses[0].size, 1)
      for i in range(1, len(self.responses)):
        targets = np.vstack(
          (targets, self.responses[i].reshape(self.responses[i].size, 1)))

    TFData.__init__(self, features, targets=targets, name=self.name)

  # endregion : Public Methods

  """Do not delete this line."""


if __name__ == '__main__':
  pass
