import numpy as np
import mindspore
from mindspore import ops as _ops
from mindspore.ops import Primitive
from mindspore.ops.operations.math_ops import Fmax, Fmin

from ..executor import execute
from .pointwise import pow

import easy_mindspore
from easy_mindspore import MS_22

device_target = mindspore.get_context('device_target')

# argmax
def argmax(input, dim=None, keepdim=False):
    if dim is None:
        dim = -1
        input = input.reshape(-1)
    return max(input, dim, keepdim)[1]

# argmin
def argmin(input, dim=None, keepdim=False):
    if dim is None:
        dim = -1
        input = input.reshape(-1)
    return min(input, dim, keepdim)[1]

# amax
_amax = Primitive('ReduceMax')
_amax.init_prim_io_names(inputs=['input_x', 'axis'], outputs=['y'])
def amax(input, dim, keepdim=False):
    _amax.add_prim_attr('keep_dims', keepdim)
    return execute(_amax, input, dim)

# amin
_amin = Primitive('ReduceMin')
_amin.init_prim_io_names(inputs=['input_x', 'axis'], outputs=['y'])
def amin(input, dim, keepdim=False):
    _amin.add_prim_attr('keep_dims', keepdim)
    return execute(_amin, input, dim)

# aminmax
def aminmax(input, *, dim=None, keepdim=False):
    if dim is None:
        dim = ()
    return amin(input, dim, keepdim), amax(input, dim, keepdim)

# all
_all = Primitive('ReduceAll')
_all.init_prim_io_names(inputs=['input_x', 'axis'], outputs=['y'])
def all(input, dim, keepdim=False, *, dtype=None):
    _all.add_prim_attr('keep_dims', keepdim)
    return execute(_all, input, dim).to(dtype)

# any
_any = Primitive('ReduceAny')
_any.init_prim_io_names(inputs=['input_x', 'axis'], outputs=['y'])
def any(input, dim, keepdim=False, *, dtype=None):
    _any.add_prim_attr('keep_dims', keepdim)
    return execute(_any, input, dim).to(dtype)

# max
_ops.ArgMaxWithValue
_max = Primitive('ArgMaxWithValue')
_max.init_prim_io_names(inputs=['x'], outputs=['index', 'values'])
def max(input, dim, keepdim=False):
    _max.add_prim_attr('keep_dims', keepdim)
    _max.add_prim_attr('dimension', dim)
    _max.add_prim_attr('axis', dim)
    if MS_22:
        max_indices, max = execute(_max, input)
    else:
        max_indices, max = execute(_max, input, dim, keepdim)
    return max, max_indices

# min
_min = Primitive('ArgMinWithValue')
_min.init_prim_io_names(inputs=['x'], outputs=['index', 'values'])
def min(input, dim, keepdim=False):
    _min.add_prim_attr('keep_dims', keepdim)
    _min.add_prim_attr('dimension', dim)
    _min.add_prim_attr('axis', dim)
    min_indices, min = execute(_min, input)
    return min, min_indices

# dist
_ops.dist

# logsumexp
_ops.logsumexp

# mean
_mean = Primitive('ReduceMean')
_mean.init_prim_io_names(inputs=['input_x', 'axis'], outputs=['y'])
def mean(input, dim, keepdim=False, *, dtype=None):
    if MS_22:
        _mean.add_prim_attr('keep_dims', keepdim)
        return execute(_mean, input, dim).to(dtype)
    return execute(_mean, input, dim, keepdim).to(dtype)

# nanmean


# median
_median = Primitive('Median')
_median.init_prim_io_names(inputs=['x'], outputs=['y', 'indices'])
def median(input, dim=-1, keepdim=False):
    _median.add_prim_attr("cust_aicpu", 'Median')
    _median.add_prim_attr('global_median', False)
    _median.add_prim_attr('axis', dim)
    _median.add_prim_attr('keep_dims', keepdim)
    _median.add_prim_attr('ignore_nan', False)
    return execute(_median, input)

# nanmedian
def nanmedian(input, dim=-1, keepdim=False):
    _median.add_prim_attr("cust_aicpu", 'Median')
    _median.add_prim_attr('global_median', False)
    _median.add_prim_attr('axis', dim)
    _median.add_prim_attr('keep_dims', keepdim)
    _median.add_prim_attr('ignore_nan', True)
    return execute(_median, input)

# mode


# norm
_ops.norm

# nansum


# prod
_prod = Primitive('ReduceProd')
_prod.init_prim_io_names(inputs=['input_x', 'axis'], outputs=['y'])
def prod(input, dim, keepdim=False, *, dtype=None):
    _prod.add_prim_attr('keep_dims', keepdim)
    return execute(_prod, input, dim).to(dtype)

# quantile
_quantile = Primitive('Quantile')
def quantile(input, q, dim=None, keepdim=False, *, interpolation='linear'):
    _quantile.add_prim_attr("dim", dim)
    _quantile.add_prim_attr("keep_dims", keepdim)
    _quantile.add_prim_attr("ignore_nan", False)
    return execute(_quantile, input, q)

# nanquantile
def nanquantile(input, q, dim=None, keepdim=False, *, interpolation='linear'):
    _quantile.add_prim_attr("dim", dim)
    _quantile.add_prim_attr("keep_dims", keepdim)
    _quantile.add_prim_attr("ignore_nan", True)
    return execute(_quantile, input, q)

# std
_std = Primitive('ReduceStd')
_std.init_prim_io_names(inputs=['input_x'], outputs=['output_std', 'output_mean'])
def std(input, dim=None, *, correction=1, keepdim=False):
    if device_target == 'GPU':
        _std.set_device('CPU')
    unbiased = bool(correction)
    if dim is None:
        dim = ()
    if isinstance(dim, int):
        dim = (dim,)
    _std.add_prim_attr('axis', dim)
    _std.add_prim_attr('unbiased', unbiased)
    _std.add_prim_attr('keep_dims', keepdim)
    return execute(_std, input)

# std_mean
def std_mean(input, dim=None, *, correction=1, keepdim=False):
    return std(input, dim, correction=correction, keepdim=keepdim), \
        mean(input, dim, keepdim)

# sum
_ops.ReduceSum
_sum = Primitive('ReduceSum')
_sum.init_prim_io_names(inputs=['input_x', 'axis'], outputs=['y'])
_sum.add_prim_attr('skip_mode', False)
def sum(input, dim=None, keepdim=False, *, dtype=None):
    if input.dtype == easy_mindspore.bool:
        input = input.to(easy_mindspore.int32)
    if dim is None:
        dim = ()
    _sum.add_prim_attr('keep_dims', keepdim)
    if MS_22:
        return execute(_sum, input, dim).to(dtype)
    return execute(_sum, input, dim, keepdim, False).to(dtype)

# unique
_ops.unique
_ops.Unique
def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    pass

# unique_consecutive
_ops.UniqueConsecutive
_unique_consecutive = Primitive('UniqueConsecutive')
_unique_consecutive.init_prim_io_names(inputs=['x'], outputs=['output'])
def unique_consecutive(input, return_inverse, return_counts, dim=None):
    _unique_consecutive.add_prim_attr("return_idx", return_inverse)
    _unique_consecutive.add_prim_attr("return_counts", return_counts)
    _unique_consecutive.add_prim_attr("axis", dim)
    return execute(_unique_consecutive, input)

# var
def var(input, dim=None, *, correction=1, keepdim=False):
    return pow(std(input, dim, correction=correction, keepdim=keepdim), 2)

# var_mean
def var_mean(input, dim=None, *, correction=1, keepdim=False):
    return pow(std(input, dim, correction=correction, keepdim=keepdim), 2), \
        mean(input, dim, keepdim)

# count_nonzero
_count_nonzero = Primitive('CountNonZero')
_count_nonzero.init_prim_io_names(inputs=['x'], outputs=['y'])
def count_nonzero(input, dim=None):
    _count_nonzero.add_prim_attr("dims", dim)
    return execute(_count_nonzero, input)
