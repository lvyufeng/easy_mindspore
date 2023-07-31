from mindspore import ops
from mindspore.ops import Primitive
from mindspore.common.api import _pynative_executor as executor

from mindtorch.dtype import int64

tile_op = Primitive('Tile')
tile_op.init_prim_io_names(inputs=['x', 'multiples'], outputs=['output'])
def raw_tile(x, multiples):
    return executor.real_run_op(tile_op, "Tile", (x, multiples))

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

_cast = Primitive('Cast')
_cast.init_prim_io_names(inputs=['x', 'dst_type'], outputs=['output'])
def raw_cast(x, dtype):
    return executor.real_run_op(_cast, "Cast", (x, dtype))

_log_softmax = Primitive('LogSoftmax')
def raw_log_softmax(x, axis=-1):
    _log_softmax.add_prim_attr('axis', axis)
    return executor.real_run_op(_log_softmax, "LogSoftmax", (x,))

_log_softmax_grad = Primitive('LogSoftmaxGrad')
def raw_log_softmax_grad(y, gy, axis=-1):
    _log_softmax_grad.add_prim_attr('axis', axis)
    return executor.real_run_op(_log_softmax_grad, "LogSoftmaxGrad", (y, gy))

# lt, le, eq, ne, gt, ge
_equal = Primitive('Equal')
_equal.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])
def raw_equal(x, y):
    return executor.real_run_op(_equal, "Equal", (x, y))

_lt = ops.Less()
def raw_lt(x, y):
    return executor.real_run_op(_lt, "Less", (x, y))

_le = ops.LessEqual()
def raw_le(x, y):
    return executor.real_run_op(_le, "LessEqual", (x, y))

_ne = ops.NotEqual()
def raw_ne(x, y):
    return executor.real_run_op(_ne, "NotEqual", (x, y))

_gt = ops.Greater()
def raw_gt(x, y):
    return executor.real_run_op(_gt, "Greater", (x, y))

_ge = ops.GreaterEqual()
def raw_ge(x, y):
    return executor.real_run_op(_ge, "GreaterEqual", (x, y))
