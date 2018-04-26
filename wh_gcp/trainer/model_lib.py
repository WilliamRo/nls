from tframe import Predictor
from tframe.layers import Activation
from tframe.layers import Linear
from tframe.layers import Input

from tframe.nets.resnet import ResidualNet

from models.neural_net import NeuralNet


# region : MLP

def mlp_00(mark, memory_depth, hidden_dim, hidden_num, learning_rate,
           activation='relu'):
  # Initiate a predictor
  model = NeuralNet(memory_depth, mark=mark, nn_class=Predictor)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([memory_depth]))
  for i in range(hidden_num):
    nn.add(Linear(output_dim=hidden_dim))
    nn.add(Activation(activation))
  nn.add(Linear(output_dim=1))

  # Build model
  model.default_build(learning_rate)

  # Return model
  return model

# endregion : MLP

# region : ResNet

def res_00(mark, memory_depth, hidden_dim, num_blocks, learning_rate,
           activation='relu'):
  model = NeuralNet(memory_depth, mark=mark, nn_class=Predictor)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add blocks
  nn.add(Input([memory_depth]))
  nn.add(Linear(output_dim=hidden_dim))
  nn.add(Activation(activation))
  def add_res_block():
    net = nn.add(ResidualNet())
    net.add(Linear(output_dim=hidden_dim))
    net.add(Activation(activation))
    net.add_shortcut()
  for _ in range(num_blocks): add_res_block()
  nn.add(Linear(output_dim=1))

  # Build model
  model.default_build(learning_rate)

  # Return model
  return model

# endregion : ResNet
