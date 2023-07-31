from mindspore import ops
from mindspore.ops import Primitive
from mindspore.common.api import _pynative_executor as executor

ones_like_op = Primitive('OnesLike')
ones_like_op.init_prim_io_names(inputs=['x'], outputs=['y'])
def raw_ones_like(x, *, dtype=None):
    return executor.real_run_op(ones_like_op, 'OnesLike', [x])

zeros_like_op = Primitive('ZerosLike')
zeros_like_op.init_prim_io_names(inputs=['x'], outputs=['y'])
def raw_zeros_like(x, *, dtype=None):
    return executor.real_run_op(zeros_like_op, 'ZerosLike', [x])

_zeros = Primitive('Zeros')
def raw_zeros(shape, dtype):
    return executor.real_run_op(_zeros, 'Zeros', [shape, dtype])

_ones = Primitive('Ones')
def raw_ones(shape, dtype):
    return executor.real_run_op(_ones, 'Ones', [shape, dtype])

_uniform = Primitive('UniformReal')
_uniform.init_prim_io_names(inputs=['shape'], outputs=['output'])
_uniform.add_prim_attr('seed0', 0)
_uniform.add_prim_attr('seed2', 0)
def raw_uniform(shape):
    return executor.real_run_op(_uniform, 'UniformReal', [shape])


_normal = Primitive("StandardNormal")
_normal.init_prim_io_names(inputs=['shape'], outputs=['output'])
_normal.add_prim_attr('seed0', 0)
_normal.add_prim_attr('seed2', 0)
def raw_normal(shape):
    return executor.real_run_op(_normal, 'StandardNormal', [shape])
