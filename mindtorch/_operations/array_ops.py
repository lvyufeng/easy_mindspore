from mindspore import ops
from mindspore.ops import Primitive
from mindspore.common.api import _pynative_executor as executor

from mindtorch.dtype import int64

tile_op = Primitive('Tile')
tile_op.init_prim_io_names(inputs=['x', 'multiples'], outputs=['output'])
def raw_tile(x, multiples):
    return executor.real_run_op(tile_op, "Tile", (x, multiples))

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

cast_op = Primitive('Cast')
cast_op.init_prim_io_names(inputs=['x', 'dst_type'], outputs=['output'])
def raw_cast(x, dtype):
    return executor.real_run_op(cast_op, "Cast", (x, dtype))

unsqueeze_op = Primitive('ExpandDims')
unsqueeze_op.init_prim_io_names(inputs=['x', 'axis'], outputs=['output'])
def raw_unsqueeze(x, axis):
    return executor.real_run_op(unsqueeze_op, "ExpandDims", (x, axis))

squeeze_op = Primitive('Squeeze')
squeeze_op.init_prim_io_names(inputs=['x'], outputs=['output'])
def raw_squeeze(x, axis):
    if not isinstance(axis, tuple):
        axis = (axis,)
    squeeze_op.add_prim_attr("axis", axis)
    return executor.real_run_op(squeeze_op, "Squeeze", (x,))

_reshape = ops.Reshape()
def raw_reshape(x, shape):
    return executor.real_run_op(_reshape, "Reshape", (x, shape))

_transpose = ops.Transpose()
def raw_transpose(x, perm):
    return executor.real_run_op(_transpose, "Transpose", (x, perm))

_broadcast_to = Primitive('BroadcastTo')
def raw_broadcast_to(x, shape):
    _broadcast_to.add_prim_attr("shape", shape)
    return executor.real_run_op(_broadcast_to, "BroadcastTo", (x,))

_argmax = Primitive('Argmax')
_argmax.init_prim_io_names(inputs=['x'], outputs=['output'])
_argmax.add_prim_attr('output_type', int64)
def raw_argmax(x, axis):
    _argmax.add_prim_attr('axis', axis)
    return executor.real_run_op(_argmax, "Argmax", (x,))

_equal = Primitive('Equal')
_equal.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])
def raw_equal(x, y):
    return executor.real_run_op(_equal, "Equal", (x, y))

_cast = Primitive('Cast')
_cast.init_prim_io_names(inputs=['x', 'dst_type'], outputs=['output'])
def raw_cast(x, dtype):
    return executor.real_run_op(_cast, "Cast", (x, dtype))

