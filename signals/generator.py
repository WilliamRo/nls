from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import signals.signal_ as signal_


def multi_tone(freqs, fs, duration):
  """Generate multi-tone signal. 
      numpy will perfectly handle the situation when 1/fs is not an integer"""
  t = np.arange(0, duration, 1 / fs)
  x = np.zeros_like(t)
  for f in freqs:
    x += np.cos(2 * np.pi * f * t)

  return signal_.Signal(x, fs=fs, signal_freqs=freqs, sampling_freq=fs)


class Generator(object):
  """Signal generator"""
  pass


if __name__ == "__main__":
  fs = 1000
  duration = 6
  freqs = [500, 700]
  signal = multi_tone(freqs, fs, duration)
  signal.plot()
