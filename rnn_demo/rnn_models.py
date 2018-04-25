from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.models import Predictor
from tframe.models.recurrent import Recurrent
from tframe.nets.rnn_cells import BasicRNNCell

from tframe.layers import Input
from tframe.layers import Linear

from tframe import console


def vanilla_RNN(mark):
  batch_size = 3
  num_steps = 8
  model = Predictor(mark, net_type=Recurrent)

  # Add functions
  model.add(Input(sample_shape=[1]))
  model.add(BasicRNNCell(state_size=10, inner_struct='concat'))
  model.add(Linear(output_dim=1))

  # Build model
  model._build(loss='euclid', metric='ratio', metric_name='Err %')

  # Return model
  return model


if __name__ == '__main__':
  pass
