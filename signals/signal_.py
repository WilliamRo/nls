from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import signals.utils.figure as _figure


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

  # region : Basic Properties

  @property
  def is_real(self):
    return not np.iscomplex(self).any()

  @property
  def duration(self):
    return None if self.fs is None else self.size / self.fs

  # endregion : Basic Properties

  # region : Properties in Transfer Domain

  @property
  def spectrum(self):
    spectrum = np.fft.fft(self)
    # return spectrum
    if not self.is_real or self.fs is None:
      return np.fft.fftshift(spectrum)
    else:
      freqs = np.fft.fftfreq(self.size, 1 / self.fs)
      return spectrum[freqs >= 0]

  @property
  def amplitude_spectrum(self):
    fs = 1. if self.fs is None else self.fs
    # Here we apply a scaling factor of 1/fs so that the amplitude of the
    #  FFT at a frequency component equals that of the CFT and to preserve
    #  Parseval's theorem
    return np.abs(self.spectrum) / fs

  @property
  def power_spectrum_density(self):
    fs = self.size if self.fs is None else self.fs
    return np.abs(self.spectrum * np.conj(self.spectrum) / self.size / fs)

  # endregion : Properties in Transfer Domian

  # region : Statistic Properties

  @property
  def average(self):
    return float(np.average(self))

  @property
  def variance(self):
    return float(np.var(self))

  # endregion : Statistic Properties

  # region : Properties for Plotting

  @property
  def time_axis(self):
    if self.fs is None: return None
    else: return np.linspace(0, 1, self.size) * self.duration

  @property
  def freq_axis(self):
    if self.fs is None: return None
    else:
      freqs = np.fft.fftfreq(self.size, 1 / self.fs)
      if self.is_real:
        return freqs[freqs >= 0]
      else:
        return np.fft.fftshift(freqs)

  # endregion : Properties for Plotting

  # region : Other Properties

  @property
  def info_string(self):
    info = ""
    for key in self.dict:
      info += "" if len(info) == 0 else "  |  "
      info += "{} = {}".format(key, self.dict[key])

    return info

  # endregion : Other Properties

  # endregion : Properties

  # region : Public Methods

  def auto_correlation(self, lags, keep_dim=False):
    if isinstance(lags, int):
      lags = (0, lags)
    if not (isinstance(lags, tuple) or isinstance(lags, list)):
      raise TypeError('!! Input lags must be a a tuple or a list')
    results = np.ones_like(self)
    for lag in lags:
      results *= np.append(np.zeros((lag,)), self)[:self.size]

    if keep_dim: return results
    return float(np.average(results))

  def plot(self, form_title=None, time_domain=False, db=True):
    if form_title is None: form_title = self.info_string
    fig = _figure.Figure(form_title)
    if time_domain: fig.add(_figure.Subplot.TimeDomainPlot(self))
    fig.add(_figure.Subplot.AmplitudeSpectrum(self, db=db))
    fig.plot()

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
  print('>> All is well.')
