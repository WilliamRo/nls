from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import signals.utils as utils


class Signal(np.ndarray):
  """Base class of all signals"""

  def __new__(cls, input_array, fs=None, **kwargs):
    if not isinstance(input_array, np.ndarray):
      raise TypeError("input array for constructing signal should be an "
                      "instance of numpy.ndarray")
    obj = input_array.view(cls)
    obj.fs = fs
    obj.dict = kwargs

    return obj

  # region : Properties

  @property
  def spectrum(self):
    spectrum = np.fft.fft(self)
    # return spectrum
    return np.fft.fftshift(spectrum)

  @property
  def energy_density(self):
    return np.abs(self.spectrum)

  @property
  def duration(self):
    return None if self.fs is None else self.size / self.fs

  @property
  def time_axis(self):
    if self.fs is None: return None
    else: return np.linspace(0, 1, self.size) * self.duration

  @property
  def freq_axis(self):
    if self.fs is None: return None
    else:
      freqs = np.fft.fftfreq(self.size, 1 / self.fs)
      return np.fft.fftshift(freqs)

  @property
  def info_string(self):
    info = ""
    for key in self.dict:
      info += "" if len(info) == 0 else " | "
      info += "{} = {}".format(key, self.dict[key])

    return info

  # endregion : Properties

  # region : Public Methods

  def plot(self, form_title=None, time_domain=False):
    if form_title is None: form_title = self.info_string
    utils.plot(self, form_title, time_domain=time_domain,
               freq_domain=True, show=True)

  # endregion : Public Methods

  # region : Private Methods
  # endregion : Private Methods

  # region : Superclass Preserved

  def __array_finalize__(self, obj):
    if obj is None: return
    self.fs = getattr(obj, 'fs', None)
    self.dict = getattr(obj, 'dict', None)

  # endregion : Superclass Preserved

  '''For some reason, do not delete this line'''


if __name__ == "__main__":
  x = np.arange(10)
  s = Signal(x)
  print(s)
