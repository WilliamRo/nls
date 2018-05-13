from signals.utils import DataSet



def load_data(file_path, depth):
  return psudo_load_data(file_path, depth)


def psudo_load_data(file_path, depth):
  data_set = DataSet.load(file_path)
  data_set.init_tfdata(depth)
  return data_set, data_set, data_set


if __name__ == '__main__':
  data_dir = './hello_nls_data.tfd'
  train_set, val_set, test_set = psudo_load_data(data_dir, depth=5)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)

