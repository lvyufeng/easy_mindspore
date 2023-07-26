from mindspore import context
BACKEND = context.get_context('device_target')

from .dtype import *
from ._tensor import Tensor, setup_tensor, tensor
from ._functions import ones, zeros
setup_tensor()

