from mindspore.ops import Primitive
from mindspore.common.api import _pynative_executor as executor
from mindtorch import BACKEND
from .array_ops import raw_squeeze, raw_unsqueeze

sum_op = Primitive('ReduceSum')
sum_op.init_prim_io_names(inputs=['input_x', 'axis'], outputs=['y'])
sum_op.add_prim_attr('skip_mode', False)
def raw_sum(x, axis=None, keepdims=False):
    sum_op.add_prim_attr('keep_dims', keepdims)
    if axis is None:
        axis = ()
    return executor.real_run_op(sum_op, 'ReduceSum', [x, axis])

add_op = Primitive('Add')
def raw_add(x, y):
    return executor.real_run_op(add_op, 'Add', [x, y])

mul_op = Primitive('Mul')
def raw_mul(x, y):
    return executor.real_run_op(mul_op, 'Mul', [x, y])

neg_op = Primitive('Neg')
def raw_neg(x):
    return executor.real_run_op(neg_op, 'Neg', [x])

matmul_op = Primitive('MatMul')
matmul_op.init_prim_io_names(inputs=['x1', 'x2'], outputs=['output'])
def raw_matmul(x, y, transpose_a=False, transpose_b=False):
    matmul_op.add_prim_attr('transpose_a', transpose_a)
    matmul_op.add_prim_attr('transpose_b', transpose_b)
    if BACKEND == 'Ascend':
        matmul_op.add_prim_attr('transpose_x1', transpose_a)
        matmul_op.add_prim_attr('transpose_x2', transpose_b)

    if len(x.shape) == 1:
        x = raw_unsqueeze(x, 1 if transpose_a else 0)
        out = executor.real_run_op(matmul_op, "MatMul", (x, y))
        out = raw_squeeze(out, 1 if transpose_a else 0)
        return out

    if len(y.shape) == 1:
        y = raw_unsqueeze(y, 0 if transpose_b else 1)
        out = executor.real_run_op(matmul_op, "MatMul", (x, y))
        out = raw_squeeze(out, 0 if transpose_b else 1)
        return out

    return executor.real_run_op(matmul_op, "MatMul", (x, y))

stridedslice_op = Primitive('StridedSlice')
stridedslice_op.init_prim_io_names(inputs=['x', 'begin', 'end', 'strides'], outputs=['output'])
def raw_strided_slice(x, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
    stridedslice_op.add_prim_attr('begin_mask', begin_mask)
    stridedslice_op.add_prim_attr('end_mask', end_mask)
    stridedslice_op.add_prim_attr('ellipsis_mask', ellipsis_mask)
    stridedslice_op.add_prim_attr('new_axis_mask', new_axis_mask)
    stridedslice_op.add_prim_attr('shrink_axis_mask', shrink_axis_mask)
    return executor.real_run_op(stridedslice_op, "StridedSlice", (x, begin, end, strides))


stridedslice_grad_op = Primitive('StridedSliceGrad')
stridedslice_grad_op.init_prim_io_names(inputs=['dy', 'shapex', 'begin', 'end', 'strides'], outputs=['output'])
def raw_strided_slice_grad(dout, x_shape, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
    stridedslice_op.add_prim_attr('begin_mask', begin_mask)
    stridedslice_op.add_prim_attr('end_mask', end_mask)
    stridedslice_op.add_prim_attr('ellipsis_mask', ellipsis_mask)
    stridedslice_op.add_prim_attr('new_axis_mask', new_axis_mask)
    stridedslice_op.add_prim_attr('shrink_axis_mask', shrink_axis_mask)
    return executor.real_run_op(stridedslice_op, "StridedSliceGrad", (dout, x_shape, begin, end, strides))
