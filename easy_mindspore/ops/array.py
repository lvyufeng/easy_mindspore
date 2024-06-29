from mindspore import ops as _ops
from mindspore.ops import Primitive

from ..executor import execute
from easy_mindspore import MS_22

# adjoint


# argwhere
_nonzero = _ops.NonZero()
def argwhere(input):
    return execute(_nonzero, input)

# cat
_ops.Concat
_cat = Primitive('Concat')
def cat(tensors, dim=0):
    if MS_22:
        _cat.add_prim_attr('axis', dim)
        return execute(_cat, tensors)
    return execute(_cat, tensors, dim)

# concat
def concat(tensors, dim=0):
    return cat(tensors, dim)

# concatenate
def concatenate(tensors, dim=0):
    return cat(tensors, dim)

# conj
_conj = _ops.Conj()
def conj(input):
    return execute(_conj, input)

# chunk
_chunk = Primitive('Split')
def chunk(input, chunks, dim=0):
    if MS_22:
        _chunk.add_prim_attr('num_split', chunks)
        _chunk.add_prim_attr('output_num', chunks)
        _chunk.add_prim_attr('axis', dim)
        return execute(_chunk, input)
    _chunk._set_prim_arg("axis", dim)
    _chunk._set_prim_arg("output_num", chunks)

    return execute(_chunk, input, dim, chunks)

# dsplit


# column_stack


# dstack


# gather
_ops.Gather
_gather = Primitive('Gather')
_gather.add_prim_attr("batch_dims", 0)
_gather.init_prim_io_names(inputs=['params', 'indices', 'axis'], outputs=['output'])
def tf_gather(params, indices, axis):
    if MS_22:
        return execute(_gather, params, indices, axis)
    return execute(_gather, params, indices, axis, 0)

# hsplit


# hstack


# index_add


# index_copy


# index_reduce


# index_select


# masked_select


# movedim


# moveaxis


# narrow
_ops.narrow
_ops.Slice

# narrow_copy


# nonzero
def nonzero(input, *, as_tuple=False):
    return argwhere(input)

# permute
_permute = _ops.Transpose()
def permute(input, dims):
    return execute(_permute, input, dims)

# reshape
_reshape = _ops.Reshape()
def reshape(input, shape):
    return execute(_reshape, input, shape)

# row_stack

# select
def select(input, dim, index):
    slices = ()
    for _ in range(dim):
        slices += (slice(None, None, None),)
    slices += (index,)
    return input[slices]

# scatter
_scatter = Primitive('TensorScatterElements')
_scatter.init_prim_io_names(inputs=['data', 'indices', 'updates'], outputs=['y'])
def scatter(input, dim, index, src):
    _scatter.add_prim_attr('axis', dim)
    _scatter.add_prim_attr('reduction', 'none')
    return execute(_scatter, input, index, src)

# diagonal_scatter


# select_scatter


# slice_scatter


# scatter_add
def scatter_add(input, dim, index, src):
    _scatter.add_prim_attr('axis', dim)
    _scatter.add_prim_attr('reduction', 'add')
    return execute(_scatter, input, index, src)

# scatter_reduce


# split
_ops.Split
_split = Primitive('Split')
def split(tensor, split_size_or_sections, dim=0):
    assert isinstance(split_size_or_sections, int)
    num_split = tensor.shape[dim] // split_size_or_sections
    if MS_22:
        _split.add_prim_attr('num_split', num_split)
        _split.add_prim_attr('output_num', num_split)
        _split.add_prim_attr('axis', dim)
        return execute(_split, tensor)

    _split._set_prim_arg("axis", dim)
    _split._set_prim_arg("output_num", num_split)
    return execute(_split, tensor, dim, num_split)

# squeeze
_squeeze = Primitive('Squeeze')
_squeeze.init_prim_io_names(inputs=['x'], outputs=['output'])
def squeeze(input, dim=None):
    if dim is None:
        dim = ()
    if isinstance(dim, int):
        dim = (dim,)
    _squeeze.add_prim_attr("axis", (dim,))
    return execute(_squeeze, input)

# stack
_stack = Primitive('Stack')
_stack.init_prim_io_names(inputs=['x'], outputs=['y'])
def stack(tensors, dim=0):
    _stack.add_prim_attr('axis', dim)
    return execute(_stack, tensors)

# swapaxes
def swapaxes(input, dim0, dim1):
    return transpose(input, dim0, dim1)

# swapdims
def swapdims(input, dim0, dim1):
    return transpose(input, dim0, dim1)

# take


# take_along_dim


# tensor_split


# tile
_tile = _ops.Tile()
def tile(input, dims):
    return execute(_tile, input, dims)

# transpose
def transpose(input, dim0, dim1):
    ranks = list(range(input.ndim))
    rank0 = ranks[dim0]
    rank1 = ranks[dim1]
    ranks[dim0] = rank1
    ranks[dim1] = rank0
    return permute(input, tuple(ranks))

# unbind
_unbind = Primitive('Unstack')
_unbind.init_prim_io_names(inputs=['x'], outputs=['y'])
def unbind(input, dim=0):
    _unbind.add_prim_attr('axis', dim)
    return execute(_unbind, input)

# unravel_index
_ops.UnravelIndex

# unsqueeze
_unsqueeze = _ops.ExpandDims()
def unsqueeze(input, dim):
    return execute(_unsqueeze, input, dim)

# vsplit

# vstack

# where
_stridedslice = Primitive('StridedSlice')
_stridedslice.init_prim_io_names(inputs=['x', 'begin', 'end', 'strides'], outputs=['output'])
_stridedslice.add_prim_attr('side_effect_mem', True)
_stridedslice.add_prim_attr("view_op", True)
def strided_slice(x, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
    if MS_22:
        _stridedslice.add_prim_attr('begin_mask', begin_mask)
        _stridedslice.add_prim_attr('end_mask', end_mask)
        _stridedslice.add_prim_attr('ellipsis_mask', ellipsis_mask)
        _stridedslice.add_prim_attr('new_axis_mask', new_axis_mask)
        _stridedslice.add_prim_attr('shrink_axis_mask', shrink_axis_mask)
        return execute(_stridedslice, x, begin, end, strides)
    return execute(_stridedslice, x, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)

_scatter_nd_update = _ops.ScatterNdUpdate()
def scatter_nd_update(input, indices, updates):
    return execute(_scatter_nd_update, input, indices, updates)

__all__ = [
    # adjoint
    "argwhere",
    "cat",
    "concat",
    "concatenate",
    "conj",
    "chunk",
    # dsplit
    # column_stack
    # dstack
    # gather
    "tf_gather",
    # hsplit
    # hstack
    # index_add
    # index_copy
    # index_reduce
    # index_select
    # masked_select
    # movedim
    # moveaxis
    # narrow
    # narrow_copy
    "nonzero",
    "permute",
    "reshape",
    # row_stack
    "select",
    "scatter",
    # diagonal_scatter
    # select_scatter
    # slice_scatter
    "scatter_add",
    # scatter_reduce
    "split",
    "squeeze",
    "stack",
    "swapaxes",
    "swapdims",
    # take
    # take_along_dim
    # tensor_split
    "tile",
    "transpose",
    "unbind",
    # unravel_index
    'unsqueeze',
    # vsplit
    # vstack
    # where
]
