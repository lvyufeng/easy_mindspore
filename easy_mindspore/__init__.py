import mindspore
from mindspore import context

# context.set_context(pynative_synchronize=True)
BACKEND = context.get_context('device_target')

from . import ops
from .dtype import *
from ._tensor import Tensor, tensor, is_tensor
from . import cuda

# def manual_seed(seed):
#     mindspore.set_seed(seed)

# def device(device):
#     return device

# def empty(*shape, dtype=None, requires_grad=False, **kwargs):
#     if dtype is None:
#         dtype = float32
#     return Tensor(*shape, dtype=dtype, requires_grad=requires_grad)
