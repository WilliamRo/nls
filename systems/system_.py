from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class System(object):
  """Base class of all systems"""
  def __init__(self):
    self.response_function = System._default_response

  # region : Properties

  @property
  def foo(self):
    return None

  # endregion : Properties

  # region : Public Methods

  def response(self, x):
    return self.response_function(x)

  # endregion : Public Methods

  # region : Private Methods

  @staticmethod
  def _default_response(x):
    return x

  # endregion : Private Methods

  '''For some reason, do not delete this line'''


if __name__ == "__main__":
  print("GG")
