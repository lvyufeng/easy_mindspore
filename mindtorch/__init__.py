import mindspore
from mindspore import context
BACKEND = context.get_context('device_target')

from .dtype import *
from .config import no_grad
from ._tensor import Tensor, setup_tensor, tensor, is_tensor
from ._functions import ones, zeros, flatten, uniform, randn, zeros_like, ones_like, sqrt
from mindtorch import cuda as cuda

setup_tensor()

def manual_seed(seed):
    mindspore.set_seed(seed)

def device(device):
    return device

def empty(*shape, dtype=None, requires_grad=False, **kwargs):
    if dtype is None:
        dtype = float32
    return Tensor(*shape, dtype=dtype, requires_grad=requires_grad)
