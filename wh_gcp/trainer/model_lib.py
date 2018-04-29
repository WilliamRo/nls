from tframe import Predictor
from tframe import pedia
from tframe.layers import Activation
from tframe.layers import Linear
from tframe.layers import Input
from tframe.nets.resnet import ResidualNet
from tframe.models.sl.bresnet import BResNet

from models.neural_net import NeuralNet
from models.neural_net import NlsHub


# region : MLP

def mlp_00(th, activation='relu'):
  assert isinstance(th, NlsHub)
  # Initiate a predictor
  model = NeuralNet(th.memory_depth, mark=th.mark, nn_class=Predictor)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input([th.memory_depth]))
  for i in range(th.num_blocks):
    nn.add(Linear(output_dim=th.hidden_dim, weight_regularizer=th.regularizer,
                  strength=th.reg_strength))
    nn.add(Activation(activation))
  nn.add(Linear(output_dim=1))

  # Build model
  model.default_build(th.learning_rate)

  # Return model
  return model

# endregion : MLP

# region : ResNet

def res_00(th, activation='relu'):
  assert isinstance(th, NlsHub)
  model = NeuralNet(th.memory_depth, mark=th.mark, nn_class=Predictor)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add blocks
  nn.add(Input([th.memory_depth]))
  nn.add(Linear(output_dim=th.hidden_dim, weight_regularizer=th.regularizer,
                strength=th.reg_strength))
  nn.add(Activation(activation))
  def add_res_block():
    net = nn.add(ResidualNet())
    net.add(Linear(output_dim=th.hidden_dim,
                   weight_regularizer=th.regularizer, strength=th.reg_strength))
    net.add(Activation(activation))
    net.add_shortcut()
  for _ in range(th.num_blocks): add_res_block()
  nn.add(Linear(output_dim=1))

  # Build model
  model.default_build(th.learning_rate)

  # Return model
  return model

# endregion : ResNet

# region : BResNet

def bres_net00(th, activation='relu'):
  assert isinstance(th, NlsHub)
  # Initiate a neural net model
  model = NeuralNet(th.memory_depth, mark=th.mark, nn_class=BResNet)
  nn = model.nn
  assert isinstance(nn, BResNet)

  # Add layers
  nn.add(Input([th.memory_depth]))
  for _ in range(th.num_blocks):
    nn.add(Linear(output_dim=th.hidden_dim, #weight_initializer='zeros',
                  weight_regularizer=th.regularizer, strength=th.reg_strength))
    nn.add(Activation(activation))
    branch = nn.add_branch()
    branch.add(Linear(output_dim=1))
  # Build
  model.default_build(th.learning_rate)

  # Return model
  return model

def bres_net01(th, activation='relu'):
  assert isinstance(th, NlsHub)
  # Initiate a neural net model
  model = NeuralNet(th.memory_depth, mark=th.mark, nn_class=BResNet)
  nn = model.nn
  assert isinstance(nn, BResNet)

  # Add layers
  nn.add(Input([th.memory_depth]))
  nn._inter_type = pedia.fork
  for _ in range(th.num_blocks):
    branch = nn.add()
    branch.add(Linear(output_dim=th.hidden_dim))
    branch.add(Activation(activation))
    branch.add(Linear(output_dim=1))

  # Build
  model.default_build(th.learning_rate)

  # Return model
  return model

# endregion : BResNet


if __name__ == '__main__':
  th = NlsHub()
  model = bres_net00(th)


















