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


class NeuralNet(Model):
  """A model for non-linear system based on neural network"""

  def __init__(self, memory_depth, mark='nn', **kwargs):
    # Sanity check
    if memory_depth < 1: raise ValueError('!! Memory depth should be positive')

    # Call parent's construction methods
    Model.__init__(self)

    # Initialize fields
    self.memory_depth = memory_depth
    self.D = memory_depth
    self.nn = Predictor(mark=mark)

    # Build default nn
    build_default = kwargs.get('build_default', False)
    hidden_dims = kwargs.get('hidden_dims', None)
    if build_default: self._create_default_mlp(hidden_dims)

  # region : Public Methods

  def inference(self, input_, **kwargs):
    if not self.nn.built:
      raise AssertionError('!! Model has not been built yet')
    mlp_input = self._gen_mlp_input(input_)
    tfinput = TFData(mlp_input)
    output = self.nn.predict(tfinput).flatten()

    output = Signal(output)
    output.__array_finalize__(input_)
    return output

  def identify(self, input_, output, val_input=None, val_output=None,
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
    self.nn.train(batch_size=batch_size, training_set=training_set,
                  validation_set=val_set, print_cycle=print_cycle,
                  snapshot_cycle=snapshot_cycle, epoch=epoch,
                  snapshot_function=snapshot_function)

  def gen_snapshot_function(self, input_, response):
    from signals.utils import Figure, Subplot

    # Sanity check
    if not isinstance(input_, Signal) or not isinstance(response, Signal):
      raise TypeError('!! Input and response should be instances of Signal')

    def snapshot_function(obj):
      assert isinstance(obj, Predictor)
      pred = self(input_)
      delta = pred - response

      fig = Figure()
      fig.add(Subplot.PowerSpectrum(response, prefix='Ground Truth'))
      prefix = 'Predicted, $||\Delta||$ = {:.4f}'.format(delta.norm)
      fig.add(Subplot.PowerSpectrum(pred, prefix=prefix, Delta=delta))

      return fig.plot(show=False, ylim=True)

    return snapshot_function

  # endregion : Public Methods

  # region : Private Methods

  def _gen_mlp_input(self, input_):
    N = input_.size
    x = np.append(np.zeros(shape=(self.D - 1,)), input_)
    features = np.zeros(shape=(N, self.D))
    for i in range(N): features[i] = x[i:i+self.D]
    return features

  def _create_default_mlp(self, hidden_dims=None):
    from tframe.layers import Input, Linear, Activation
    hidden_dims = [self.D] * 2 if hidden_dims is None else hidden_dims
    activation = lambda: Activation('relu')
    learning_rate = 0.001

    # Add layers
    self.nn.add(Input([self.D]))

    for dim in hidden_dims:
      self.nn.add(Linear(output_dim=dim))
      self.nn.add(activation())

    self.nn.add(Linear(output_dim=1))

    # Build model
    self.nn.build(
      loss='euclid', optimizer=tf.train.AdamOptimizer(learning_rate),
      metric='delta', metric_name='Val-Delta')

  # endregion : Private Methods


  """For some reason, do not remove this line"""


if __name__ == "__main__":
  model = NeuralNet(memory_depth=3, dont_build=True)
  input_ = np.arange(1, 11)
  print(input_)
  print(model._gen_mlp_input(input_))

