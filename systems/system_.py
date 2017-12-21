from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from signals.signal_ import Signal
from signals.utils import Figure
from signals.utils import Subplot


class System(object):
  """Base class of all systems"""
  def __init__(self, response_function):
    if not callable(response_function):
      raise TypeError('!! Input response must be callable')
    self.response_function = response_function

  # region : Public Methods

  def response(self, signl):
    if not isinstance(signl, Signal):
      raise TypeError('!! Input must be an instance of Signal')
    output = Signal(self.response_function(signl))
    output.__array_finalize__(signl)
    return output

  # endregion : Public Methods

  # region : Private Methods

  @staticmethod
  def _default_response(x):
    return x

  # endregion : Private Methods

  # region : Operator Overloading

  def __call__(self, signl):
    return self.response(signl)

  # endregion : Operator Overloading

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

  def response(input_):
    y = np.zeros_like(input_)
    for i in range(len(input_)):
      x = input_[i]
      x_p1 = input_[i - 1] if i != 0 else 0
      y[i] = x * x_p1 if x > 0 else 0.4 * x + 0.9 * x_p1
    return y
  system = System(response)
  output = system(signal)

  fig = Figure('Input & Output')
  db = True
  fig.add(Subplot.AmplitudeSpectrum(signal, prefix='Input Signal', db=db))
  fig.add(Subplot.AmplitudeSpectrum(output, prefix='Output Signal', db=db))
  fig.plot()

