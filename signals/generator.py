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
    x += vrms[i] * np.sqrt(2) * np.cos(2 * np.pi * freqs[i] * t + phases[i])

  # Instantiate Signal
  signl = signal_.Signal(x, fs=fs, signal_freqs=freqs, sampling_freq=fs)

  # Add gaussian white noise to signal
  # Reference: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html
  noise_power = noise_power * signl.fs / 2
  noise = gaussian_white_noise(noise_power, size=signl.shape, fs=signl.fs)
  return signl + noise


def gaussian_white_noise(intensity, size, fs):
  """Generate a gaussian white noise with specific intensity A.
     That is, R_XX(\tau) = A\delta_(\tau), S_XX(\omega) = A
     R_XX(\tau) = E[x[t]x[t-\tau]] = \sigma^2 \delta(\tau)
     Reference: https://www.gaussianwaves.com/2013/11/simulation-and-analysis-of-white-noise-in-matlab/"""
  noise = np.random.normal(scale=np.sqrt(intensity), size=size)
  signl = signal_.Signal(noise, fs=fs)

  return signl


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
