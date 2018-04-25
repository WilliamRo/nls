import tensorflow as tf

from tframe import FLAGS
from tframe import Predictor

from tframe.layers import Activation
from tframe.layers import Linear
from tframe.layers import Input

from models.neural_net import NeuralNet


# region : MLP

def mlp_00(learning_rate=0.001, memory_depth=80):
  """
  Performance on WH:
    [0] depth = 80
  """
  # Configuration
  hidden_dims = [2 * memory_depth] * 4
  strength = 0
  activation = 'lrelu'

  mark = 'mlp_D{}_{}_{}'.format(memory_depth, hidden_dims, activation)

  # Initiate a predictor
  model = NeuralNet(memory_depth, mark=mark)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory_depth]))
  _add_fc_relu_layers(nn, hidden_dims, activation, strength=strength)
  nn.add(Linear(output_dim=1, weight_regularizer='l2', strength=strength))

  # Build model
  # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  nn._build(loss='euclid', metric='rms_ratio', metric_name='RMS(err)%',
            optimizer=optimizer)

  # Return model
  return model

# endregion : MLP

def _add_fc_relu_layers(nn, hidden_dims, activation='relu', strength=0.0):
  assert isinstance(nn, Predictor)
  assert isinstance(hidden_dims, (tuple, list))

  for dim in hidden_dims:
    nn.add(Linear(output_dim=dim, weight_regularizer='l2', strength=strength))
    nn.add(Activation(activation))
