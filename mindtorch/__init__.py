from mindspore import context
BACKEND = context.get_context('device_target')

from .tensor import Tensor, setup_tensor

setup_tensor()

