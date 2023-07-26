from mindtorch.autograd import Function
from mindtorch._operations import raw_ones, raw_ones_like, raw_zeros, raw_zeros_like
from mindtorch.dtype import *

class Ones(Function):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def forward(self):
        y = raw_ones(self.shape, self.dtype)
        return y

    def backward(self):
        return zeros(self.shape, self.shape)

def ones(*shape, dtype=None, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    if dtype is None:
        dtype = float32
    return Ones(shape, dtype)(requires_grad=requires_grad)


class Zeros(Function):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def forward(self):
        y = raw_zeros(self.shape, self.dtype)
        return y

    def backward(self):
        return zeros(self.shape, self.shape)

def zeros(*shape, dtype=None, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    if dtype is None:
        dtype = float32

    return Zeros(shape, dtype)(requires_grad=requires_grad)

class OnesLike(Function):
    def forward(self, x):
        y = raw_ones_like(x)
        return y

    def backward(self, gy):
        return zeros_like(gy)

def ones_like(x, *, dtype=None, requires_grad=False):
    return OnesLike()(x, requires_grad=requires_grad)

class ZerosLike(Function):
    def forward(self, x):
        y = raw_zeros_like(x)
        return y

    def backward(self, gy):
        return zeros_like(x, gy)

def zeros_like(x, *, dtype=None, requires_grad=False):
    return ZerosLike()(x, requires_grad=requires_grad)
