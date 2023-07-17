from mindspore import ops
from mindspore.ops import Primitive
from mindspore.common.api import _PyNativeExecutor

executor = _PyNativeExecutor()

sum_op = Primitive('ReduceSum')
sum_op.init_prim_io_names(inputs=['input_x', 'axis'], outputs=['y'])
sum_op.add_prim_attr('skip_mode', False)
def _sum(x, axis=None, keepdims=False):
    sum_op.add_prim_attr('keep_dims', keepdims)
    if axis is None:
        axis = ()
    return executor.real_run_op(sum_op, 'ReduceSum', [x, axis])


ones_like_op = Primitive('OnesLike')
ones_like_op.init_prim_io_names(inputs=['x'], outputs=['y'])
def _ones_like(x, *, dtype=None):
    return executor.real_run_op(ones_like_op, 'OnesLike', [x])

zeros_like_op = Primitive('ZerosLike')
zeros_like_op.init_prim_io_names(inputs=['x'], outputs=['y'])
def _zeros_like(x, *, dtype=None):
    return executor.real_run_op(zeros_like_op, 'ZerosLike', [x])

add_op = Primitive('Add')
def _add(x, y):
    return executor.real_run_op(add_op, 'Add', [x, y])

mul_op = Primitive('Mul')
def _mul(x, y):
    return executor.real_run_op(mul_op, 'Mul', [x, y])

neg_op = Primitive('Neg')
def _neg(x):
    return executor.real_run_op(neg_op, 'Neg', [x])

matmul_op = Primitive('MatMul')
matmul_op.init_prim_io_names(inputs=['x1', 'x2'], outputs=['output'])
def _matmul(x, y, transpose_a=False, transpose_b=False):
    matmul_op.add_prim_attr('transpose_a', transpose_a)
    matmul_op.add_prim_attr('transpose_b', transpose_b)
    return executor.real_run_op(matmul_op, "MatMul", (x, y))
