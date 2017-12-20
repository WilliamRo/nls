from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import signals


class Figure(object):
  """Class for plot signals in different domains"""
  def __init__(self, figtitle="Untitled", figsize=(10, 8), subplots=None):
    self.figure = plt.figure(figtitle, figsize=figsize, facecolor='w')
    self.figure.canvas.mpl_connect('key_press_event', Figure._on_key)
    self.subplots = []

    if subplots is not None and (
          isinstance(subplots, list) or isinstance(subplots, tuple)):
      for subplot in subplots: self.add(subplot)

  # region : Public Methods

  def plot(self, show=True):
    nrows = len(self.subplots)
    if nrows == 0:
      print('>> Nothing to be plotted')
      return

    # Plot
    row = 0
    for subplot in self.subplots:
      assert isinstance(subplot, Subplot)
      row += 1
      plt.subplot(nrows, 1, row)
      subplot.plot()

    # Adjust subplot layout
    plt.subplots_adjust(hspace=0.4)

    # Show figure
    if show: plt.show()

  def add(self, subplot):
    if not isinstance(subplot, Subplot):
      raise TypeError('Input must be an instance of Subplot')
    self.subplots.append(subplot)

  # endregion : Public Methods

  # region : Private Methods

  @staticmethod
  def _on_key(event):
    if event.key in ['escape']:
      plt.close()

  # endregion : Private Methods

  """For some reason, do not delete this line"""


class Subplot(object):
  """Subplot definition in a figure"""

  def __init__(self, signl, plot, db=False, prefix=None):
    assert isinstance(signl, signals.signal_.Signal)
    assert callable(plot)
    self.signl = signl
    self.plot = lambda: plot(self)
    self.db = db
    self.predix = prefix

  # region : Static Methods

  @staticmethod
  def TimeDomainPlot(signl, prefix=None):
    return Subplot(signl, Subplot._time_domain_plot, db=False, prefix=prefix)

  @staticmethod
  def AmplitudeSpectrum(signl, db=True, prefix=None):
    return Subplot(signl, Subplot._amplitude_spectrum, db=db, prefix=prefix)

  @staticmethod
  def PowerSpectrum(signl, db=True, prefix=None):
    return Subplot(signl, Subplot._power_spectrum, db=db, prefix=prefix)

  # endregion : Static Methods

  # region : Private Methods

  def _time_domain_plot(self):
    xlabel = 'Time'
    if self.signl.fs is not None: xlabel += ' (s)'
    Subplot._plot(self.signl.time_axis, self.signl,
                  title=self._make_title('Time Domain Plot'),
                  xlabel=xlabel, ylabel='Voltage')

  def _amplitude_spectrum(self):
    density = (20 * np.log10(self.signl.amplitude_spectrum) if self.db
               else self.signl.amplitude_spectrum)
    Subplot._plot(self.signl.freq_axis, density)
    ylabel = 'Amplitude'
    if self.db: ylabel += ' (dB)'
    Subplot._plot(self.signl.freq_axis, density,
                  title=self._make_title('Amplitude Spectrum'),
                  xlabel='Frequency (Hz)', ylabel=ylabel)

  def _power_spectrum(self):
    raise NotImplementedError('Sorry dude ...')

  def _make_title(self, title):
    if self.predix is not None: title = '[{}] {}'.format(self.predix, title)
    return title

  @staticmethod
  def _plot(x, y, title=None, xlabel=None, ylabel=None, grid=True):
    assert y is not None and isinstance(y, np.ndarray)
    assert len(y.shape) == 1
    if x is None:
      x = np.arange(y.size)

    plt.plot(x, y)
    plt.xlim(x[0], x[-1])
    if title is not None:
      plt.title(title)
    if xlabel is not None:
      plt.xlabel(xlabel)
    if ylabel is not None:
      plt.ylabel(ylabel)

    plt.grid(grid)

  # endregion : Private Methods

  """For some reason, do not delete this line"""


if __name__ == '__main__':
  from signals.generator import multi_tone
  fs = 2000
  duration = 1
  freqs = [500, 800]
  vrms = [2, 1]
  phases = [0, np.pi]
  signal = multi_tone(freqs, fs, duration, vrms=vrms, phases=phases,
                      noise_power=1e-2)

  fig = Figure(signal.info_string)
  fig.add(Subplot.TimeDomainPlot(signal, prefix='Input signal'))
  fig.add(Subplot.AmplitudeSpectrum(signal, db=True))
  fig.plot()

