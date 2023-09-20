from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from mindtorch.autograd import Function, Context
from mindtorch.dtype import *

class Ones(Function):
    @staticmethod
    def forward(ctx:Context, shape, dtype):
        y = ops.ones(shape, dtype)
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
        y = ops.zeros(shape, dtype)
        return y

def zeros(*shape, dtype=None, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    if dtype is None:
        dtype = float32
    return Zeros.apply(shape=shape, dtype=dtype, requires_grad=requires_grad)

class Uniform(Function):
    @staticmethod
    def forward(ctx:Context, shape):
        _uniform = _get_cache_prim(ops.UniformReal)()
        y = _uniform(shape)
        return y

def uniform(*shape, dtype=None, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    if dtype is None:
        dtype = float32
    return Uniform.apply(shape=shape, requires_grad=requires_grad)

class Normal(Function):
    @staticmethod
    def forward(ctx:Context, shape):
        y = ops.standard_normal(shape)
        return y

def randn(*shape, dtype=None, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    if dtype is None:
        dtype = float32
        out = Normal.apply(shape=shape, requires_grad=requires_grad)
        return out.to(dtype)
    return Normal.apply(shape=shape, requires_grad=requires_grad)

class OnesLike(Function):
    @staticmethod
    def forward(ctx:Context, x):
        y = ops.ones_like(x)
        return y

def ones_like(x, *, dtype=None, requires_grad=False):
    return OnesLike.apply(x, requires_grad=requires_grad)

class ZerosLike(Function):
    @staticmethod
    def forward(ctx:Context, x):
        y = ops.zeros_like(x)
        return y

def zeros_like(x, *, dtype=None, requires_grad=False):
    return ZerosLike.apply(x, requires_grad=requires_grad)
