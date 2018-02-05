from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe import TFData
from tframe import console

import signals


class DataSet(TFData):
  """A dataset class for signals based on TFData"""

  def __init__(self, signls, responses=None, name='unnamed', memory_depth=None,
               intensity=None):
    # Sanity check
    if not isinstance(signls, (tuple, list)): signls = [signls]
    fs = signls[0].fs
    for signl in signls:
      if not isinstance(signl, signals.Signal):
        raise TypeError('!! Each item in signals must be a Signal')
      if signl.fs != fs:
        raise ValueError('!! signals must have the same fs')

    if responses is not None:
      if not isinstance(responses, (tuple, list)): responses = [responses]
      for resp in responses:
        if not isinstance(resp, signals.Signal):
          raise TypeError('!! Responses must be Signals')
        if len(responses) != len(signls):
          raise ValueError('!! signls and responses must have the same len')
        if resp.fs != fs:
          raise ValueError('!! signals must have the same fs')

    # Initiate fields
    self.signls = signls
    self.responses = responses
    self.intensity = intensity
    self.name = name
    self.fs = fs

    # Call parent's constructor
    if memory_depth is not None: self.init_tfdata(memory_depth)

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

  @staticmethod
  def load(filename):
    data = TFData.load(filename)
    if not isinstance(data, DataSet):
      raise TypeError('!! data is not an instance of DataSet')
    for signl in data.signls: signl.fs = data.fs
    if data.responses is not None:
      for resp in data.responses: resp.fs = data.fs
    return data

  # endregion : Public Methods

  """Do not delete this line."""


def load_wiener_hammerstein(
    filename, validation_size=20000, test_size=88000, depth=None):
  console.show_status('Loading Wiener-Hammerstein benchmark ...')

  # Load dataset and check input parameters
  dataset = DataSet.load(filename)
  assert isinstance(dataset, DataSet)
  u, y = dataset.signls[0], dataset.responses[0]
  L = u.size
  if validation_size + test_size > L:
    raise ValueError(
      '!! validation_size({}) + test_size({}) > total_size({})'.format(
        validation_size, test_size, L))

  # Separate data
  training_size = L - validation_size - test_size
  train_slice = slice(0, training_size)
  training_set = DataSet(u[train_slice], y[train_slice], memory_depth=depth,
                         name='training set')

  val_slice = slice(training_size, training_size + validation_size)
  validation_set = DataSet(u[val_slice], y[val_slice], memory_depth=depth,
                           name='validation set')

  test_slice = slice(L-test_size, L)
  test_set = DataSet(u[test_slice], y[test_slice], memory_depth=depth,
                     name='test set')

  # Show status
  console.show_status('Data set loaded')
  console.supplement('Training set size: {}'.format(
    training_set.signls[ 0].size))
  console.supplement('Validation set size: {}'.format(
    validation_set.signls[ 0].size))
  console.supplement('Test set size: {}'.format(test_set.signls[0].size))

  return training_set, validation_set, test_set


if __name__ == '__main__':
  pass


