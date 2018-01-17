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
  model = NeuralNet(memory_depth, mark)


  return model

# endregion : Multi-layer Perceptron
