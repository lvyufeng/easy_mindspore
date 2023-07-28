from mindtorch.autograd import Function, Context
from mindtorch._operations import raw_ones, raw_ones_like, raw_zeros, raw_zeros_like
from mindtorch.dtype import *

class Ones(Function):
    @staticmethod
    def forward(ctx:Context, shape, dtype):
        ctx.save_for_backward(shape, dtype)
        y = raw_ones(shape, dtype)
        return y

def ones(*shape, dtype=None, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    if dtype is None:
        dtype = float32
    return Ones.apply(shape=shape, dtype=dtype, requires_grad=requires_grad)


class Zeros(Function):
    @staticmethod
    def forward(ctx:Context, shape, dtype):
        ctx.save_for_backward(shape, dtype)
        y = raw_zeros(shape, dtype)
        return y

def zeros(*shape, dtype=None, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    if dtype is None:
        dtype = float32
    return Zeros.apply(shape=shape, dtype=dtype, requires_grad=requires_grad)

class OnesLike(Function):
    @staticmethod
    def forward(ctx:Context, x):
        y = raw_ones_like(x)
        return y

def ones_like(x, *, dtype=None, requires_grad=False):
    return OnesLike.apply(x, requires_grad=requires_grad)

class ZerosLike(Function):
    @staticmethod
    def forward(ctx:Context, x):
        y = raw_zeros_like(x)
        return y

def zeros_like(x, *, dtype=None, requires_grad=False):
    return ZerosLike.apply(x, requires_grad=requires_grad)
