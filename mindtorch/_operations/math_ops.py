from mindspore import ops
from mindspore.ops import Primitive, PrimitiveWithInfer
from mindspore.common.api import _pynative_executor as executor
from mindtorch import BACKEND
from .array_ops import raw_squeeze, raw_unsqueeze

_sum = Primitive('ReduceSum')
_sum.init_prim_io_names(inputs=['input_x', 'axis'], outputs=['y'])
_sum.add_prim_attr('skip_mode', False)
def raw_sum(x, axis=None, keepdims=False):
    _sum.add_prim_attr('keep_dims', keepdims)
    if axis is None:
        axis = ()
    return executor.real_run_op(_sum, 'ReduceSum', [x, axis])

_addcmul = ops.Addcmul()
def raw_addcmul(input, tensor0, tensor1, value):
    return executor.real_run_op(_addcmul, 'Addcmul', [input, tensor0, tensor1, value])


_add = ops.Add()
def raw_add(x, y):
    return executor.real_run_op(_add, 'Add', [x, y])

_sub = ops.Sub()
def raw_sub(x, y):
    return executor.real_run_op(_sub, 'Sub', [x, y])

_mul = ops.Mul()
def raw_mul(x, y):
    return executor.real_run_op(_mul, 'Mul', [x, y])

_div = ops.Div()
def raw_div(x, y):
    return executor.real_run_op(_div, 'Div', [x, y])

_pow = ops.Pow()
def raw_pow(x, pow):
    return executor.real_run_op(_pow, 'Pow', [x, pow])

_sqrt = ops.Sqrt()
def raw_sqrt(x):
    return executor.real_run_op(_sqrt, 'Sqrt', [x])

_sin = ops.Sin()
def raw_sin(x):
    return executor.real_run_op(_sin, 'Sin', [x])

_cos = ops.Cos()
def raw_cos(x):
    return executor.real_run_op(_cos, 'Cos', [x])

_tanh = ops.Tanh()
def raw_tanh(x):
    return executor.real_run_op(_tanh, 'Tanh', [x])

_exp = ops.Exp()
def raw_exp(x):
    return executor.real_run_op(_exp, 'Exp', [x])

_log = ops.Log()
def raw_log(x):
    return executor.real_run_op(_log, 'Log', [x])

neg_op = Primitive('Neg')
def raw_neg(x):
    return executor.real_run_op(neg_op, 'Neg', [x])

_square = Primitive('Square')
_square.init_prim_io_names(inputs=['input_x'], outputs=['output'])
def raw_square(x):
    return executor.real_run_op(_square, 'Square', [x])


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
    stridedslice_grad_op.add_prim_attr('begin_mask', begin_mask)
    stridedslice_grad_op.add_prim_attr('end_mask', end_mask)
    stridedslice_grad_op.add_prim_attr('ellipsis_mask', ellipsis_mask)
    stridedslice_grad_op.add_prim_attr('new_axis_mask', new_axis_mask)
    stridedslice_grad_op.add_prim_attr('shrink_axis_mask', shrink_axis_mask)
    return executor.real_run_op(stridedslice_grad_op, "StridedSliceGrad", (dout, x_shape, begin, end, strides))
