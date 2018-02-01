import tensorflow as tf

from tframe import Predictor

from tframe.layers import Activation
from tframe.layers import Linear
from tframe.layers import Input
from tframe.layers.parametric_activation import Polynomial

from models.neural_net import NeuralNet


# region : MLP

def tlp(memory_depth, hidden_dim, mark='tlp'):
  # Hyper-parameters
  learning_rate = 0.001

  # Initiate a predictor
  model = NeuralNet(memory_depth, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory_depth]))
  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Activation('sigmoid'))
  nn.add(Linear(output_dim=1, use_bias=False))

  # Build model
  model.default_build(learning_rate=learning_rate)

  return model

# endregion : MLP

# region : SVN

def svn(memory_depth, order, hidden_dim, mark='svn'):
  # Hyper-parameters
  learning_rate = 0.001

  # Initiate a predictor
  model = NeuralNet(memory_depth, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory_depth]))
  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Polynomial(order=order))
  nn.add(Linear(output_dim=1, use_bias=False))

  # Build model
  # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  model.default_build(optimizer=optimizer, learning_rate=learning_rate)

  return model

# endregion : SVN

# region : PET

def pet(memory, hidden_dim, mark='pet'):
  # Hyper-parameters
  learning_rate = 0.001

  # Initiate a predictor
  model = NeuralNet(memory_depth=memory, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory]))
  nn.add(Linear(output_dim=hidden_dim))
  # TODO

  # Build model
  model.default_build(learning_rate=learning_rate)

  return model

# endregion : PET


























