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

def mlp_00(memory_depth, mark, hidden_dims, learning_rate=0.001):
  strength = 0
  # Initiate a predictor
  model = NeuralNet(memory_depth, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory_depth]))
  _add_fc_relu_layers(nn, hidden_dims, 'lrelu', strength=strength)
  nn.add(Linear(output_dim=1, weight_regularizer='l2', strength=strength))

  # Build model
  nn.build(loss='euclid', metric='rms_ratio', metric_name='RMS(err)%',
           optimizer=tf.train.AdamOptimizer(learning_rate))

  # Return model
  return model

# endregion : MLP

# region : Layers

def _add_fc_relu_layers(nn, hidden_dims, activation='lrelu', strength=0.0):
  assert isinstance(nn, Predictor)
  assert isinstance(hidden_dims, (tuple, list))

  for dim in hidden_dims:
    nn.add(Linear(output_dim=dim, weight_regularizer='l2', strength=strength))
    nn.add(Activation(activation))

# endregion : Utilities


"""LOGS
[1] mlp: 0.004%, 0.487%, 0.487%
    memory = 80
    learning rate = 0.001 on Adam -> 0.00001
    loss = euclid
    hidden_dims = [160] * 4
    activation = lrelu

"""

