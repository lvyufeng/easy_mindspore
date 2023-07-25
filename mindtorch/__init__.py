from mindspore import context
BACKEND = context.get_context('device_target')

from .dtype import *
from .tensor import Tensor, setup_tensor
from .ops import *
setup_tensor()

import torch
torch.bool