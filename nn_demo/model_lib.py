from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import pedia

from tframe.layers import Activation
from tframe.layers import BatchNorm
from tframe.layers import Dropout
from tframe.layers import Linear
from tframe.layers import Input

from tframe import regularizers

from models import NeuralNet
from tframe.models.sl.vn import VolterraNet


# region : Multi-layer Perceptron

def mlp_00(memory_depth, mark):
  D = memory_depth
  hidden_dims = [10, 10, 10]

  activation = lambda: Activation('relu')
  learning_rate = 0.001
  reg = 0.00

  # Initiate model
  model = NeuralNet(memory_depth, mark)
  model.nn.add(Input([D]))

  for dim in hidden_dims:
    model.nn.add(Linear(output_dim=dim, weight_regularizer='l2', strength=reg))
    model.nn.add(activation())

  model.nn.add(Linear(output_dim=1, weight_regularizer='l2', strength=reg))

  # Build model
  model.nn.build(loss='euclid', metric='ratio', metric_name='Err%',
                 optimizer=tf.train.AdamOptimizer(learning_rate))

  return model

# endregion : Multi-layer Perceptron

# region : Volterra Networks

def vn_00(memory_depth, mark, degree=None, homo_str=0.0):
  D = memory_depth
  hidden_dims = [[10, 10, 10]]

  if degree is None: degree = len(hidden_dims) + 1
  elif degree < 1: raise ValueError('!! Degree must be greater than 1')

  activation = lambda: Activation('relu')
  learning_rate = 0.001
  reg = 0.00

  # Initiate model
  model = NeuralNet(D, mark, degree=degree)

  for order in range(2, degree + 1):
    dims = hidden_dims[order - 2]
    for dim in dims:
      model.nn.T[order].add(Linear(dim, weight_regularizer='l2', strength=reg))
      model.nn.T[order].add(activation())
    model.nn.T[order].add(Linear(1, weight_regularizer='l2', strength=reg))

  # Build model
  model.nn.build(loss='euclid', metric='ratio', metric_name='Err%',
                 homo_strength=homo_str,
                 optimizer=tf.train.AdamOptimizer(learning_rate))
  return model

# endregion : Volterra Networks