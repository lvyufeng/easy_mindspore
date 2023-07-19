from mindspore import context
BACKEND = context.get_context('device_target')

from .tensor import Tensor
from .tensor import tensor_sum as sum