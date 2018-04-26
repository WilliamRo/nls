import numpy as np
from signals.utils.dataset import load_wiener_hammerstein, DataSet


# Configurations
memory_depth = 80
data_dir = '../data/wiener_hammerstein/whb.tfd'
err_amplitude = 1e-5

# Load data
train_set, val_set, test_set = load_wiener_hammerstein(
  data_dir, depth=memory_depth)
assert isinstance(train_set, DataSet)
assert isinstance(val_set, DataSet)
assert isinstance(test_set, DataSet)
u, y = test_set.signls[0], test_set.responses[0]

# Define error function
f_ratio = lambda val: 100 * val / y.rms
def pseud_evaluate(err_bound):
  print('-' * 79)
  pred_y = y + np.random.random_sample(y.shape) * err_bound
  # Error
  err = y - pred_y
  # Mean value
  val = err.average
  print('E[err] = {:.4f} mv ({:.3f}%)'.format(val * 1000, f_ratio(val)))
  # Standard deviation
  val = float(np.std(err))
  print('STD[err] = {:.4f} mv ({:.3f}%)'.format(val * 1000, f_ratio(val)))
  # Root mean square value of the error
  val = err.rms
  print('RMS[err] = {:.4f} mv ({:.3f}%)'.format(val * 1000, f_ratio(val)))

# ....
pseud_evaluate(err_amplitude)


