import numpy as np


def import_data():
  """Import your data here"""
  #
  return None, None


def psudo_import_data():
  from signals.generator import multi_tone
  from systems import Volterra

  # Generate input signal
  fs, duration = 2000, 100
  freqs, vrms = [500, 800], [2, 1]
  phases = [0, np.pi]
  x = multi_tone(freqs, fs, duration, vrms=vrms,
                 phases=phases, noise_power=1e-3)
  # Generate targets
  black_box = Volterra()
  black_box.set_kernel((0,), 1)
  black_box.set_kernel((5,), 0.4)
  black_box.set_kernel((1, 4), 0.1)
  black_box.set_kernel((2, 3), 0.1)
  y = black_box(x)

  return x, y


def wrap(x, y, filename=None):
  from signals.utils import DataSet

  data_set = DataSet(x, y, name='train_set')

  # Save data
  if filename is not None:
    data_set.save(filename)
    print('>> Data set saved to {}'.format(filename))


def plot(x, y, db=True):
  """Plot feature and targets"""
  from signals.utils import Figure, Subplot
  fig = Figure('System Input & Output')
  fig.add(Subplot.AmplitudeSpectrum(x, prefix='Input Signal', db=db))
  fig.add(Subplot.AmplitudeSpectrum(y, prefix='Output Signal', db=db))
  fig.plot()


def step_1(bplot=True):
  """Import and plot your data"""
  # Import data
  x, y = psudo_import_data()
  # x, y = import_data()

  # Plot
  if bplot: plot(x, y)

  return x, y


def step_2():
  """Wrap and save your data"""
  x, y = step_1(False)
  # Wrap data
  wrap(x, y, './hello_nls_data.tfd')


def step_3():
  """Load and check your data"""
  from signals.utils import DataSet
  data_set = DataSet.load('./hello_nls_data.tfd')
  assert isinstance(data_set, DataSet)
  data_set.plot()


if __name__ == '__main__':
  step_2()

