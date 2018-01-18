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


# region : Multi-layer Perceptron

def mlp_00(memory_depth, mark):
  D = memory_depth
  hidden_dims = [8, 8]

  activation = lambda: Activation('relu')
  learning_rate = 0.001

  # Initiate model
  model = NeuralNet(memory_depth, mark)
  model.nn.add(Input([D]))

  for dim in hidden_dims:
    model.nn.add(Linear(output_dim=dim))
    model.nn.add(activation())

  model.nn.add(Linear(output_dim=1))

  # Build model
  model.nn.build(loss='euclid', metric='delta', metric_name='Validation Error',
                 optimizer=tf.train.AdamOptimizer(learning_rate))

  return model

# endregion : Multi-layer Perceptron
