import numpy as np
from mindspore import ops as _ops
from mindspore.ops import Primitive
from mindspore.ops.operations.math_ops import Fmax, Fmin

from ..executor import execute

# allclose
def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return np.allclose(input.numpy(), other.numpy(), rtol, atol, equal_nan)

# argsort
def argsort(input, dim=-1, descending=False, stable=False):
    return sort(input, dim, descending, stable)[1]

# eq
_eq = _ops.Equal()
def eq(input, other):
    return execute(_eq, input, other)

# equal
def equal(input, other):
    return eq(input, other)

# ge
_ge = _ops.GreaterEqual()
def ge(input, other):
    return execute(_ge, input, other)

# gt
_gt = _ops.Greater()
def gt(input, other):
    return execute(_gt, input, other)

# greater
def greater(input, other):
    return gt(input, other)

# isclose
def isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return np.isclose(input.numpy(), other.numpy(), rtol, atol, equal_nan)

# isfinite
_isfinite = _ops.IsFinite()
def isfinite(input):
    return execute(_isfinite, input)

# isin

# isinf
_isinf = _ops.IsInf()
def isinf(input):
    return execute(_isinf, input)

# isposinf

# isneginf


# isnan
_isnan = _ops.IsNan()
def isnan(input):
    return execute(_isnan, input)

# isreal

# kthvalue

# le
_le = _ops.LessEqual()
def le(input, other):
    return execute(_le, input, other)

# less_equal
def less_equal(input, other):
    return le(input, other)

# lt
_lt = _ops.Less()
def lt(input, other):
    return execute(_lt, input, other)

# less
def less(input, other):
    return lt(input, other)

# maximum
_maximum = _ops.Maximum()
def maximum(input, other):
    return execute(_maximum, input, other)

# minimum
_minimum = _ops.Minimum()
def minimum(input, other):
    return execute(_minimum, input, other)


# fmax
_fmax = Fmax()
def fmax(input, other):
    return execute(_fmax, input, other)

# fmin
_fmin = Fmin()
def fmin(input, other):
    return execute(_fmin, input, other)

# ne
_ne = _ops.NotEqual()
def ne(input, other):
    return execute(_ne, input, other)

# not_equal
def not_equal(input, other):
    return ne(input, other)

# sort
_ops.sort
_sort = Primitive('Sort')
_sort.init_prim_io_names(inputs=['x'], outputs=['y1', 'y2'])
def sort(input, dim=-1, descending=False, stable=False):
    _sort.add_prim_attr('axis', dim)
    _sort.add_prim_attr('descending', descending)
    return execute(_sort, input)

# topk
_topk = Primitive('TopK')
_topk.init_prim_io_names(inputs=['input', 'k'], outputs=['values', 'indices'])
def topk(input, k, dim=None, largest=True, sorted=True):
    _topk.add_prim_attr("sorted", sorted)
    return execute(_topk, input, k)

# msort
def msort(input):
    return sort(input, dim=0)
