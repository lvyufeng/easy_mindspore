from mindspore._c_expression import typing
from mindspore import ops as _ops
from mindspore.ops import Primitive
from ..executor import execute
from easy_mindspore import MS_22


_cast = _ops.Cast()
def cast(input, dtype):
    if MS_22:
        return execute(_cast, input, dtype)
    return execute(_cast, input, typing.type_to_type_id(dtype))

_assign = _ops.Assign()
def assign(input, other):
    return execute(_assign, input, other)

__all__ = ['cast', 'assign']
