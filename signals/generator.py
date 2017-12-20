from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import signals.signal_ as signal_


def multi_tone(freqs, fs, duration, vrms=None, phases=None,
               noise_power=0):
  """Generate multi-tone signal. 
      numpy will perfectly handle the situation when 1/fs is not an integer"""
  # Determine the root-mean-square voltage
  if vrms is None: vrms = np.ones_like(freqs)
  if len(vrms) != len(freqs):
    raise ValueError('Length of freqs must be the same as vrms')

  # Determine phases
  if phases is None: phases = np.zeros_like(freqs)
  if len(phases) != len(freqs):
    raise ValueError('Length of freqs must be the same as phases')

  t = np.arange(0, duration, 1 / fs)
  x = np.zeros_like(t)
  for i in range(len(freqs)):
    x += vrms[i] * np.sqrt(2) * np.sin(2 * np.pi * freqs[i] * t + phases[i])

  signl = signal_.Signal(x, fs=fs, signal_freqs=freqs, sampling_freq=fs)
  return signl + gaussian_white_noise(signl, noise_power)


def gaussian_white_noise(signl, noise_power=0.001):
  """Reference: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html"""
  # Sanity check
  if not isinstance(signl, signal_.Signal):
    raise TypeError('Input signal must be an instance of Signal')
  if signl.fs is None:
    raise ValueError('The sampling frequency of signal should be specified')

  noise_power = noise_power * signl.fs / 2
  noise = np.random.normal(scale=np.sqrt(noise_power), size=signl.shape) + signl
  return noise


class Generator(object):
  """Signal generator"""
  pass


if __name__ == "__main__":
  fs = 2000
  duration = 1
  freqs = [500, 800]
  vrms = [2, 1]
  phases = [0, np.pi]
  signal = multi_tone(freqs, fs, duration, vrms=vrms, phases=phases,
                      noise_power=1e-2)
  signal.plot(db=True, time_domain=False)
