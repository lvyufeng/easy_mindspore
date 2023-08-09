from mindspore import ops
from mindspore.ops import Primitive
from mindspore.common.api import _pynative_executor as executor
from mindspore._c_expression import Tensor
from mindspore.ops._tracefunc import PackFunc

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

_gather = Primitive('Gather')
_gather.add_prim_attr("batch_dims", 0)
_gather.init_prim_io_names(inputs=['params', 'indices', 'axis'], outputs=['output'])
def raw_gather(params, indices, axis):
    return executor.real_run_op(_gather, "Gather", (params, indices, axis))

_unsorted_segment_sum = Primitive('UnsortedSegmentSum')
_unsorted_segment_sum.init_prim_io_names(inputs=['x', 'segment_ids', 'num_segments'], outputs=['y'])
def raw_unsorted_segment_sum(x, segment_ids, num_segments):
    return executor.real_run_op(_unsorted_segment_sum, "UnsortedSegmentSum", (x, segment_ids, num_segments))

_concat = Primitive('Concat')
def raw_concat(inputs, axis):
    _concat.add_prim_attr("axis", axis)
    return executor.real_run_op(_concat, "Concat", [inputs])

_slice = Primitive('Slice')
_slice.init_prim_io_names(inputs=['x', 'begin', 'size'], outputs=['output'])
def raw_slice(x, begin, size):
    return executor.real_run_op(_slice, "Slice", [x, begin, size])

_masked_fill = Primitive('MaskedFill')
_masked_fill.init_prim_io_names(inputs=['input', 'mask', 'value'], outputs=['output'])
def raw_masked_fill(input, mask, value):
    return executor.real_run_op(_masked_fill, "MaskedFill", [input, mask, value])

_split = Primitive('Split')
def raw_split(x, axis, output_num):
    _split.add_prim_attr('axis', axis)
    _split.add_prim_attr('output_num', output_num)
    _split.add_prim_attr('num_split', output_num)
    return executor.real_run_op(_split, "Split", [x])

def _pack_unfold(data, dimension, size, step):
    if dimension < 0:
        dimension += data.ndim
    indices = []
    for i in range(0, data.shape[dimension] - size + 1, step):
        indices.append(list(range(i, i + size)))

    indices = Tensor(indices)
    output = ops.gather(data, indices, axis=dimension)
    return ops.swapaxes(output, dimension + 1, -1)

_fused_unfold = PackFunc(_pack_unfold, str(id(_pack_unfold)), None, True)
def fused_unfold(data, dimension, size, step):
    return executor.real_run_op(_fused_unfold, 'PackFuc', [data, dimension, size, step])
