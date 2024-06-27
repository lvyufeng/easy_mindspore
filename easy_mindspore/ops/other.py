import copy
from mindspore import ops as _ops
from mindspore.ops import Primitive
from mindspore.ops.operations.math_ops import CumulativeLogsumexp

from ..executor import execute
from .creation import ones_like
from .array import reshape
from ..dtype import int32, int64

# atleast_1d


# atleast_2d


# atleast_3d


# bincount
_bincount = _ops.Bincount()
def bincount(input, weights=None, minlength=0):
    if minlength != 0:
        minlength = Tensor(minlength, int32)
    else:
        minlength = input.max().to(int32)
    if weights is None:
        weights = ones_like(input)
    return execute(_bincount, input, minlength, weights)

# block_diag


# broadcast_tensors


# broadcast_to
_broadcast_to = Primitive('BroadcastTo')
def broadcast_to(input, shape):
    _broadcast_to.add_prim_attr('shape', shape)
    return execute(_broadcast_to, shape)

# broadcast_shapes


# bucketize
_ops.Bucketize

# cartesian_prod


# cdist
_ops.Cdist

# clone
def clone(input):
    return copy.deepcopy(input)

# combinations


# corrcoef


# cov


# cross
_ops.Cross

# cummax
_ops.Cummax

# cummin
_ops.Cummin

# cumprod
_ops.CumProd

# cumsum
_ops.CumSum

# diag
_ops.Diag

# diag_embed


# diagflat


# diagonal
_ops.FillDiagonal

# diff


# einsum
_ops.Einsum

# flatten
def flatten(x, start_dim=1, end_dim=-1):
    """Flattens the input. Does not affect the batch size."""
    if end_dim < 0:
        end_dim = x.ndim + end_dim
    new_shape = x.shape[:start_dim] + (-1,) + x.shape[end_dim + 1:]
    return reshape(x, new_shape)

# flip


# fliplr


# flipud


# kron


# rot90


# gcd


# histc


# histogram


# histogramdd


# meshgrid
_meshgrid = _ops.Meshgrid('ij')
def meshgrid(*tensors, indexing=None):
    if indexing is not None:
        _meshgrid.add_prim_attr('indexing', indexing)
    return execute(_meshgrid, tensors)

# lcm


# logcumsumexp
CumulativeLogsumexp

# ravel


# renorm


# repeat_interleave
_ops.repeat_interleave

# roll


# searchsorted
_ops.SearchSorted

# tensordot

# trace
_ops.Trace

# tril
_ops.Tril

# tril_indices
_ops.TrilIndices

# triu
_ops.Triu

# triu_indices
_ops.TriuIndices

# unflatten
def unflatten(x, dim, sizes):
    new_shape = x.shape[:dim] + sizes
    return reshape(x, new_shape)

# vander


# view_as_real

# view_as_complex


# resolve_conj


# resolve_neg
