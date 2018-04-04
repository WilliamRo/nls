import tensorflow as tf
import config


def inference():
  pass


def build():
  # Create placeholder for inputs
  input_shape = [config.time_steps, config.batch_size, config.num_features]
  inputs = tf.placeholder(dtype=tf.float32, shape=input_shape)

  #



  return inputs


def train():
  pass


def evaluate():
  pass


def load_data():
  pass