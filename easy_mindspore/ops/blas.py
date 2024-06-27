
from mindspore import ops as _ops
from mindspore.ops import Primitive

from ..executor import execute

# addbmm
def addbmm(input, batch1, batch2, *, beta=1, alpha=1):
    return beta * input + alpha * bmm(batch1, batch2).sum(dim=0)

# addmm


# addmv


# addr


# baddbmm
def baddbmm(input, batch1, batch2, *, beta=1, alpha=1):
    return beta * input + alpha * bmm(batch1, batch2)

# bmm
_bmm = _ops.BatchMatMul()
def bmm(input, other):
    return execute(_bmm, input, other)

# chain_matmul


# cholesky
_ops.Cholesky

# cholesky_inverse
_ops.CholeskyInverse

# cholesky_solve
_ops.CholeskySolve

# dot
_ops.dot
def dot(input, other):
    return (input * other).sum()

# geqrf
_ops.Geqrf

# ger
_ops.Ger

# inner
_ops.tensor_dot
_ops.inner

# inverse
_ops.Inv

# det
_ops.MatrixDeterminant

# logdet
_ops.logdet
_ops.LogMatrixDeterminant

# slogdet
_ops.slogdet

# lu

# lu_solve
_ops.LuSolve


# lu_unpack
_ops.LuUnpack

# matmul
_matmul = _ops.MatMul()
def matmul(input, other):
    if input.ndim == 2 and other.ndim == 2:
        return execute(_matmul, input, other)
    return bmm(input, other)

# matrix_power
_ops.MatrixPower

# matrix_exp
_ops.MatrixExp

# mm


# mv


# orgqr
_ops.Orgqr

# ormqr
_ops.Ormqr

# outer
def outer(input, vec2):
    return input.reshape(-1, 1) * vec2

# pinverse


# qr
_ops.Qr

# svd
_ops.Svd

# svd_lowrank

# pca_lowrank


# lobpcg


# trapz


# trapezoid


# cumulative_trapezoid


# triangular_solve


# vdot
