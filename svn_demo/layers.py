from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

from tframe.layers.homogeneous import Homogeneous


class Volterra(Layer):
  is_nucleus = True

  full_name = 'volterra'
  abbreviation = 'volterra'

  def __init__(self, orders):
    # Check input
    if not isinstance(orders, (tuple, list)): orders = [orders]
    for order in orders:
      if order < 0: raise ValueError('!! order must be a non-negative integer')
    self.orders = orders
    # self.sublayers =


  @single_input
  def _link(self, input_, **kwargs):
    assert isinstance(input_, tf.Tensor)


