from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from signals.signal_ import Signal


class System(object):
  """Base class of all systems"""
  def __init__(self):
    self.response_function = System._default_response

  # region : Properties



  # endregion : Properties

  # region : Public Methods

  def response(self, signl):
    if not isinstance(signl, Signal):
      raise TypeError('!! Input signl must be an instance of Signal')
    return self.response_function(signl)

  # endregion : Public Methods

  # region : Private Methods

  @staticmethod
  def _default_response(x):
    return x

  # endregion : Private Methods

  '''For some reason, do not delete this line'''


if __name__ == "__main__":
  from signals.generator import multi_tone
  print(">> Running module system_.py")

  fs = 2000
  duration = 1
  freqs = [500, 800]
  vrms = [2, 1]
  phases = [0, np.pi]
  signal = multi_tone(freqs, fs, duration, vrms=vrms, phases=phases,
                      noise_power=1e-2)

  signal.plot(form_title='Input signal', db=True, time_domain=False)
