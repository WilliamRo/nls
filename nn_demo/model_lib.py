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

def vn_00(memory_depth, mark, degree=None):
  D = memory_depth
  hidden_dims = [[10, 10]]
  # hidden_dims = []
  degree = len(hidden_dims) + 1

  activation = lambda: Activation('relu')
  learning_rate = 0.001
  reg = 0.00

  # Initiate model
  model = NeuralNet(D, mark, degree=degree)

  for i in range(len(hidden_dims)):
    d = i + 2
    dims = hidden_dims[i]
    for dim in dims:
      model.nn.T[d].add(Linear(dim, weight_regularizer='l2', strength=reg))
      model.nn.T[d].add(activation())
    model.nn.T[d].add(Linear(1, weight_regularizer='l2', strength=reg))

  # Build model
  model.nn.build(loss='euclid', metric='ratio', metric_name='Err%',
                 optimizer=tf.train.AdamOptimizer(learning_rate))
  return model

# endregion : Volterra Networks