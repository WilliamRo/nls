from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import FLAGS
from tframe import Predictor

from tframe.layers import Activation
from tframe.layers import BatchNorm
from tframe.layers import Linear
from tframe.layers import Input
from tframe.layers.homogeneous import Homogeneous
from tframe.layers.parametric_activation import Polynomial
from tframe.nets.resnet import ResidualNet
from tframe import pedia

from models.neural_net import NeuralNet

import layer_combs as lc


# region : Test

def test_00(memory, learning_rate=0.001):
  # Configurations
  mark = 'test'
  D = memory

  # Initiate model
  model = NeuralNet(memory, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory]))

  nn.add(Linear(output_dim=2*D))
  nn.add(Activation('relu'))
  nn.add(Linear(output_dim=2*D))
  nn.add(Activation('relu'))
  nn.add(Linear(output_dim=2*D))
  nn.add(Polynomial(order=3))
  nn.add(Linear(output_dim=1))

  # Build model
  model.default_build(learning_rate)

  # Return model
  return model

# endregion : Test

# region : ResNet

def res_00(memory, blocks, learning_rate=0.001):
  # Configurations
  mark = 'res'
  D = memory
  activation = 'relu'

  # Initiate model
  model = NeuralNet(memory, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([D]))

  def add_res_block():
    net = nn.add(ResidualNet())
    net.add(Linear(output_dim=D))
    net.add(Activation(activation))
    net.add(Linear(output_dim=D))
    net.add_shortcut()
    net.add(Activation(activation))

  for _ in range(blocks): add_res_block()

  nn.add(Linear(output_dim=1))

  # Build model
  model.default_build(learning_rate)

  # Return model
  return model

# endregion : ResNet

# region : SVN

def svn_00(memory, learning_rate=0.001):
  # Configuration
  D = memory
  hidden_dims = [2 * D] * 4
  p_order = 2
  mark = 'svn_{}_{}'.format(hidden_dims, p_order)

  # Initiate a predictor
  model = NeuralNet(memory, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([D]))
  for dim in hidden_dims:
    nn.add(Linear(output_dim=dim))
    nn.add(Polynomial(p_order))
  nn.add(Linear(output_dim=1))

  # Build model
  model.default_build(learning_rate)

  return model

# endregion : SVN

# region : NET

def net_00(memory_depth, learning_rate=0.001):
  # Configuration
  hidden_dim = 10
  homo_order = 4
  mark = 'net_h{}_homo{}'.format(hidden_dim, homo_order)

  # Initiate a predictor
  model = NeuralNet(memory_depth, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory_depth]))
  nn.add(Linear(output_dim=hidden_dim))
  nn.add(inter_type=pedia.sum)
  for i in range(1, homo_order + 1): nn.add_to_last_net(Homogeneous(i))

  # Build model
  model.default_build(learning_rate)

  # Return model
  return model


# endregion : NET

# region : MLP

def mlp_00(memory_depth, hidden_dims, learning_rate=0.001):
  activation = 'relu'
  mark = 'mlp_D{}_L{}_{}'.format(memory_depth, len(hidden_dims), activation)

  # Initiate a predictor
  model = NeuralNet(memory_depth, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory_depth]))
  for i, dim in enumerate(hidden_dims):
    nn.add(Linear(output_dim=dim))
    nn.add(Activation(activation))
  nn.add(Linear(output_dim=1))

  # Build model
  nn.build(loss='euclid', metric='rms_ratio', metric_name='RMS(err)%',
           optimizer=tf.train.AdamOptimizer(learning_rate))

  # Return model
  return model

# endregion : MLP

# region : Layers


# endregion : Utilities


"""LOGS
[1] mlp: 
    memory = 80
    loss = euclid
    hidden_dims = [160] * 4
    activation = lrelu

"""

