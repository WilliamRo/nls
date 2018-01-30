from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import FLAGS
from tframe import Predictor

from tframe.layers import Activation
from tframe.layers import Linear
from tframe.layers import Input

from models.neural_net import NeuralNet


# region : MLP

def mlp_00(memory_depth, mark):
  # Hyper-parameters
  D = memory_depth
  hidden_dims = [2 * D] * 4
  learning_rate = 0.001

  # Initiate a predictor
  model = NeuralNet(memory_depth, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory_depth]))
  _add_fc_relu_layers(nn, hidden_dims)
  nn.add(Linear(output_dim=1))

  # Build model
  nn.build(loss='euclid', metric='ratio', metric_name='Err %',
           optimizer=tf.train.AdamOptimizer(learning_rate))

  # Return model
  return model

# endregion : MLP

# region : Layers

def _add_fc_relu_layers(nn, hidden_dims):
  assert isinstance(nn, Predictor)
  assert isinstance(hidden_dims, (tuple, list))

  for dim in hidden_dims:
    nn.add(Linear(output_dim=dim))
    nn.add(Activation.ReLU())

# endregion : Utilities
