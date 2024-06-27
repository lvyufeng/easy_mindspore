from mindspore import ops as _ops
from mindspore.ops import Primitive
from ..executor import execute


_cast = _ops.Cast()
def cast(input, dtype):
    return execute(_cast, input, dtype)

_assign = _ops.Assign()
def assign(input, other):
    return execute(_assign, input, other)

__all__ = ['cast', 'assign']
