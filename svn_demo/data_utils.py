import numpy as np

from signals import Signal
from signals.utils.dataset import DataSet
from signals.generator import gaussian_white_noise, multi_tone


def load_svn1997(depth):
  A = 1
  L = 1000 * 100
  fs = 1000
  x = gaussian_white_noise(1, L, fs)
  y = sys97(x)
  train_data = DataSet(x, y, intensity=A, memory_depth=depth)

  val_x = multi_tone([100, 260], fs, 1, noise_power=1e-4)
  val_y = sys97(val_x)
  val_data = DataSet(val_x, val_y, memory_depth=depth)

  return train_data, val_data


def load_whb():
  return None


# region : Systems

def sys97(x):
  assert isinstance(x, Signal)
  N = x.size
  v1 = np.zeros_like(x)
  v2 = np.zeros_like(x)
  for n in range(N):
    x_1 = x[n - 1] if n - 1 >= 0 else 0
    x_2 = x[n - 2] if n - 2 >= 0 else 0

    v1_1 = v1[n - 1] if n - 1 >= 0 else 0
    v1_2 = v1[n - 2] if n - 2 >= 0 else 0
    v1[n] = 1.2 * v1_1 - 0.6 * v1_2 + 0.5 * x_1

    v2_1 = v2[n - 1] if n - 1 >= 0 else 0
    v2_2 = v2[n - 2] if n - 2 >= 0 else 0
    v2_3 = v2[n - 3] if n - 3 >= 0 else 0
    v2[n] = 1.8 * v2_1 - 1.1 * v2_2 + 0.2 * v2_3 + 0.1 * (x_1 + x_2)

  y = (v1 + 0.8 * v2 * v2 - 0.6 * v1 * v1 * v2) * np.sin((v1 + v2) / 5)
  output = Signal(y, fs=x.fs)
  return output

# endregion : Systems


if __name__ == '__main__':
  train_data, val_data = load_svn1997()
  x = val_data.signls[0]
  y = val_data.responses[0]
  x.plot()
  y.plot()
