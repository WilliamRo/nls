from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import signals


def plot(signal, form_title="", time_domain=False,
         freq_domain=True, db=True, show=True):
  # Sanity check
  if not isinstance(signal, signals.Signal):
    raise TypeError('Input signal should be an instance of Signal')

  # Determine rows to plot
  nrows = 0
  if time_domain: nrows += 1
  if freq_domain: nrows += 1
  fig = plt.figure(form_title, figsize=(10, 8))

  # Plot
  row = 1
  if time_domain:
    plt.subplot(nrows, 1, row)
    _plot(signal.time_axis, signal, title='Time domain',
          xlabel='Time', ylabel='Amplitude')
    row += 1
  if freq_domain:
    plt.subplot(nrows, 1, row)
    density = (20 * np.log10(signal.energy_density) if db
               else signal.energy_density)
    ylabel = 'Energy spectrum density'
    if db: ylabel += ' (dB)'
    _plot(signal.freq_axis, density, title='Frequency domain',
          xlabel='Frequency', ylabel=ylabel)

  # Adjust subplot layout
  plt.subplots_adjust(hspace=0.4)

  # Show figure
  if show:
    plt.show()

  return fig


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


if __name__ == '__main__':
  fs = 2000
  t = np.arange(0, 6, 1 / fs)
  freqs = [500, 600]
  # Generate real multi-tone signal
  x = np.zeros_like(t)
  for freq in freqs:
    x += np.cos(2 * np.pi * freq * t)
  signal = signals.Signal(x, fs=fs)
  # Plot signal
  form_title = 'Signal freqs = {}; Sampling freq = {}'.format(freqs, fs)
  plot(signal, form_title=form_title, time_domain=True)
