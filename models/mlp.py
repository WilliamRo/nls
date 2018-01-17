from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe import console
from tframe import Predictor
from tframe import TFData

from models import Model
from signals import Signal
from signals.generator import gaussian_white_noise


class MLP(Model):
  """A model for non-linear system based on multi-layer perceptrons"""

  def __init__(self, memory_depth, hidden_dims=None, mark='mlp', **kwargs):
    # Sanity check
    if memory_depth < 1: raise ValueError('!! Memory depth should be positive')

    # Initialize fields
    self.memory_depth = memory_depth
    self.hidden_dims = hidden_dims
    self.D = memory_depth
    self.mlp = Predictor(mark=mark)

    dont_build = kwargs.get('dont_build', False)
    if not dont_build: self._create_default_mlp()

    # Call parent's construction methods
    Model.__init__(self)

  # region : Public Methods

  def inference(self, input_, **kwargs):
    mlp_input = self._gen_mlp_input(input_)
    tfinput = TFData(mlp_input)
    output = self.mlp.predict(tfinput).flatten()

    output = Signal(output)
    output.__array_finalize__(input_)
    return output

  def train(self, input_, output, val_input=None, val_output=None,
             batch_size=64, print_cycle=100, snapshot_cycle=1000,
             snapshot_function=None, epoch=1):
    # Prepare training set
    input_ = self._gen_mlp_input(input_)
    training_set = TFData(input_, targets=output.reshape(output.size, 1))

    # Prepare validation set
    val_set = None
    if val_input is not None and val_output is not None:
      val_input = self._gen_mlp_input(val_input)
      val_set = TFData(
        val_input, targets=val_output.reshape(val_output.size, 1))

    # Train
    self.mlp.train(batch_size=batch_size, training_set=training_set,
                   validation_set=val_set, print_cycle=print_cycle,
                   snapshot_cycle=snapshot_cycle, epoch=epoch,
                   snapshot_function=snapshot_function)

  # endregion : Public Methods

  # region : Private Methods

  def _gen_mlp_input(self, input_):
    N = input_.size
    x = np.append(np.zeros(shape=(self.D - 1,)), input_)
    features = np.zeros(shape=(N, self.D))
    for i in range(N): features[i] = x[i:i+self.D]
    return features

  def _create_default_mlp(self):
    from tframe.layers import Input, Linear, Activation
    hidden_dims = ([self.D] * 2 if self.hidden_dims is None
                   else self.hidden_dims)
    activation = lambda: Activation('relu')
    learning_rate = 0.001

    # Add layers
    self.mlp.add(Input([self.D]))

    for dim in hidden_dims:
      self.mlp.add(Linear(output_dim=dim))
      self.mlp.add(activation())

    self.mlp.add(Linear(output_dim=1))

    # Build model
    self.mlp.build(
      loss='euclid', optimizer=tf.train.AdamOptimizer(learning_rate),
      metric='delta', metric_name='Val-Delta')

  # endregion : Private Methods


  """For some reason, do not remove this line"""


if __name__ == "__main__":
  model = MLP(memory_depth=3, dont_build=True)
  input_ = np.arange(1, 11)
  print(input_)
  print(model._gen_mlp_input(input_))

