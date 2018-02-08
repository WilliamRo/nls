import tensorflow as tf

from tframe import Predictor

from tframe.layers import Activation
from tframe.layers import Linear


def _add_fc_relu_layers(nn, hidden_dims, activation='relu', strength=0.0):
  assert isinstance(nn, Predictor)
  assert isinstance(hidden_dims, (tuple, list))

  for dim in hidden_dims:
    nn.add(Linear(output_dim=dim, weight_regularizer='l2', strength=strength))
    nn.add(Activation(activation))
