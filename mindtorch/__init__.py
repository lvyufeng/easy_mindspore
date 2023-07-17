from mindspore import context
BACKEND = context.get_context('device_target')

from .tensor import Tensor
from .parameter import Parameter
from .nn.modules.module import Module

from .tensor import tensor_sum as sum