from tframe import Predictor
from tframe import pedia
from tframe.layers import Activation
from tframe.layers import Linear
from tframe.layers import Input
from tframe.nets.resnet import ResidualNet
from tframe.models.sl.bresnet import BResNet

from tframe.models import Recurrent
from tframe.nets.rnn_cells import BasicRNNCell

from models.neural_net import NeuralNet
from models.neural_net import NlsHub


# region : MLP

def mlp_00(th):
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
    nn.add(Activation(th.actype1))
  nn.add(Linear(output_dim=1))

  # Build model
  model.default_build(th.learning_rate)

  # Return model
  return model

# endregion : MLP

# region : RNN

def rnn0(th):
  assert isinstance(th, NlsHub)
  # Initiate a neural net model
  nn_class = lambda mark: Predictor(mark=mark, net_type=Recurrent)
  model = NeuralNet(th.memory_depth, mark=th.mark, nn_class=nn_class)
  nn = model.nn
  assert isinstance(nn, Predictor)

  # Add layers
  nn.add(Input(sample_shape=[th.memory_depth]))
  for _ in range(th.num_blocks):
    nn.add(BasicRNNCell(state_size=th.hidden_dim, inner_struct='add'))
  nn.add(Linear(output_dim=1))

  # Build
  model.default_build(th.learning_rate)

  return model

# endregion: RNN
















