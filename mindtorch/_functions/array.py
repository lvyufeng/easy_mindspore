from mindtorch.autograd import Function, Context
from mindtorch._operations import raw_sum, raw_reshape, raw_transpose, raw_broadcast_to, \
    raw_matmul, raw_strided_slice, raw_strided_slice_grad, raw_argmax, raw_equal, \
    raw_cast, raw_log_softmax, raw_log_softmax_grad, raw_lt, raw_le, raw_ne, raw_gt, raw_ge, \
    raw_gather, raw_unsorted_segment_sum, raw_concat, raw_slice
from mindtorch._functions import utils
from mindtorch import dtype, tensor, Tensor
from .utils import ensure_tensor
from .creation import zeros_like
# =============================================================================
# Tensor operations: reshape / transpose / expand_dims / flatten
# =============================================================================
class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, x, shape):
        ctx.save_for_backward(x._shape)
        y = raw_reshape(x, shape)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x_shape, = ctx.saved_values
        return reshape(gy, x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return x
    return Reshape.apply(x, shape=shape)


def flatten(x, start_dim=1, end_dim=-1):
    """Flattens the input. Does not affect the batch size."""
    if end_dim < 0:
        end_dim = x.ndim + end_dim
    new_shape = x.shape[:start_dim] + (-1,) + x.shape[end_dim + 1:]
    return reshape(x, new_shape)

def unflatten(x, dim, sizes):
    new_shape = x.shape[:dim] + sizes
    return reshape(x, new_shape)


class Transpose(Function):
    @staticmethod
    def forward(ctx: Context, x, axes):
        ctx.save_for_backward(axes)
        y = raw_transpose(x, axes)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        axes, = ctx.saved_values
        if axes is None:
            return _transpose(gy)

        axes_len = len(axes)
        inv_axes = utils.argsort([ax % axes_len for ax in axes])
        return _transpose(gy, tuple(inv_axes))


def _transpose(x, axes=None):
    if axes is None:
        axes = (1, 0)
    return Transpose.apply(x, axes=axes)

def permute(input, dims):
    return _transpose(input, dims)

def transpose(input, dim0, dim1):
    axes = list(range(input.ndim))
    if dim0 < 0:
        dim0 = input.ndim + dim0
    if dim1 < 0:
        dim1 = input.ndim + dim1
    axes[dim0] = dim1
    axes[dim1] = dim0
    return _transpose(input, tuple(axes))

def expand_dims(x, axis):
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))

def squeeze(x, dim=None):
    shape = x.shape
    if dim is None:
        new_shape = tuple([s for s in shape if s != 1 ])
    else:
        new_shape = list(shape)
        new_shape.pop(dim)
        new_shape = tuple(new_shape)
    return reshape(x, new_shape)

# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================
class Sum(Function):
    @staticmethod
    def forward(ctx: Context, x, axis, keepdims):
        ctx.save_for_backward(x._shape, axis, keepdims)
        y = raw_sum(x, axis=axis, keepdims=keepdims)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x_shape, axis, keepdims = ctx.saved_values
        gy = utils.reshape_sum_backward(gy, x_shape, axis, keepdims)
        gx = broadcast_to(gy, x_shape)
        return gx

def sum(x, dim=None, keepdims=False):
    if x.dtype == dtype.bool:
        x = x.long()
    return Sum.apply(x, axis=dim, keepdims=keepdims)

def mean(x, dim=None, keepdims=False):
    y = sum(x, dim, keepdims)
    return y * (y.data._size / x.data._size)

class SumTo(Function):
    @staticmethod
    def forward(ctx: Context, x, shape):
        ctx.save_for_backward(x._shape)
        y = utils.sum_to(x, shape)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x_shape, = ctx.saved_values
        gx = broadcast_to(gy, x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return x
    return SumTo.apply(x, shape=shape)


class BroadcastTo(Function):
    @staticmethod
    def forward(ctx: Context, x, shape):
        ctx.save_for_backward(x._shape)
        y = raw_broadcast_to(x, shape)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x_shape, = ctx.saved_values
        gx = sum_to(gy, x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return x
    return BroadcastTo.apply(x, shape=shape)

class GetItem(Function):
    @staticmethod
    def forward(ctx: Context, x, slices):
        slices = utils.slice_helper(slices)
        ctx.save_for_backward(slices, x._shape)
        y = raw_strided_slice(x, *slices)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        slices, x_shape = ctx.saved_values
        return get_item_grad(gy, slices, x_shape)


class GetItemGrad(Function):
    @staticmethod
    def forward(ctx: Context, gy, slices, in_shape):
        # slices = utils.slice_helper(slices)
        ctx.save_for_backward(slices)
        gx = raw_strided_slice_grad(gy, in_shape, *slices)
        return gx

    @staticmethod
    def backward(ctx: Context, ggx):
        slices, = ctx.saved_tensors
        return get_item(ggx, slices)


def get_item(x, slices):
    return GetItem.apply(x, slices=slices)

def get_item_grad(gy, slices, x_shape):
    return GetItemGrad.apply(gy, slices=slices, in_shape=x_shape)

class Argmax(Function):
    @staticmethod
    def forward(ctx: Context, x, axis):
        y = raw_argmax(x, axis)
        return y

def argmax(x, axis):
    return Argmax.apply(x, axis=axis, requires_grad=False)

class Cast(Function):
    @staticmethod
    def forward(ctx: Context, x, dtype):
        ctx.save_for_backward(x.dtype)
        return raw_cast(x, dtype)
    
    @staticmethod
    def backward(ctx: Context, gy):
        dtype, = ctx.saved_tensors
        return cast(gy, dtype)

def cast(x, dtype):
    return Cast.apply(x, dtype=dtype)


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx: Context, x, axis):
        ctx.save_for_backward(axis)
        return raw_log_softmax(x, axis)

    @staticmethod
    def backward(ctx: Context, gy):
        axis, = ctx.saved_tensors
        y, = ctx.outputs
        gx = raw_log_softmax_grad(y().data, gy.data, axis)
        return tensor(gx, gy.requires_grad)

# cmp operators
class Equal(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        return raw_equal(x, y)

def equal(x, y):
    if isinstance(y, Tensor):
        return Equal.apply(x, y, requires_grad=False)
    return Equal.apply(x, y=y, requires_grad=False)

class Less(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        return raw_lt(x, y)

def less(x, y):
    if isinstance(y, Tensor):
        return Less.apply(x, y, requires_grad=False)
    return Less.apply(x, y=y, requires_grad=False)

class LessEqual(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        return raw_le(x, y)

def le(x, y):
    if isinstance(y, Tensor):
        return LessEqual.apply(x, y, requires_grad=False)
    return LessEqual.apply(x, y=y, requires_grad=False)

class Greater(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        return raw_gt(x, y)

def greater(x, y):
    if isinstance(y, Tensor):
        return Greater.apply(x, y, requires_grad=False)
    return Greater.apply(x, y=y, requires_grad=False)

class GreaterEqual(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        return raw_ge(x, y)

def ge(x, y):
    if isinstance(y, Tensor):
        return GreaterEqual.apply(x, y, requires_grad=False)
    return GreaterEqual.apply(x, y=y, requires_grad=False)

class Gather(Function):
    @staticmethod
    def forward(ctx: Context, params, indices, axis):
        ctx.save_for_backward(axis)
        return raw_gather(params, indices, axis)

    @staticmethod
    def backward(ctx: Context, dout):
        params, indices = ctx.inputs
        axis, = ctx.saved_values

        orig_indices = indices
        if dout.ndim == 0:
            dout = expand_dims(dout, -1)

        if indices.ndim == 0:
            indices = expand_dims(indices, -1)
            out_shp = utils._regenerate_output_shape(params.shape, indices.shape, axis)
            dout = reshape(dout, out_shp)

        x_shp = params.shape
        out_shp = dout.shape
        ind_shp = indices.shape
        # Example: out_shape:(3,2,3) axis 1 -> (1,0,2)
        perm_1 = utils.generate_shape_index(out_shp, ind_shp, axis)
        values_transpose = _transpose(dout, perm_1)
        params_grad = raw_unsorted_segment_sum(values_transpose.data, indices.data, params.shape[axis])
        # Example: out_shape:(3,2,3) axis 2 -> (1,2,0)
        perm_2 = utils._generate_inverse_index(x_shp, axis)
        params_grad = raw_transpose(params_grad, perm_2)
        return tensor(params_grad, dout.requires_grad), zeros_like(orig_indices)

def gather(params, indices, axis):
    return Gather.apply(params, indices, axis=axis)

class Concat(Function):
    @staticmethod
    def forward(ctx: Context, *inputs, dim):
        input_sizes = [i.shape[dim] for i in inputs]
        ctx.save_for_backward(dim, tuple(input_sizes))
        return raw_concat(inputs, dim)

    @staticmethod
    def backward(ctx: Context, grad_output):
        dim, input_sizes = ctx.saved_values
        return tuple(narrow(grad_output, dim, end - size, size) for size, end
                            in zip(input_sizes, utils._accumulate(input_sizes)))

def cat(tensors, dim=0):
    return Concat.apply(*tensors, dim=dim)

class Slice(Function):
    @staticmethod
    def forward(ctx: Context, input, begin, size):
        ctx.save_for_backward(begin, size)
        return raw_slice(input, begin, size)

def slice(input, begin, size):
    return Slice.apply(input, begin=begin, size=size)

def narrow(input, axis, start, length):
    begins = [0] * input.ndim
    begins[axis] = start
    sizes = list(input.shape)
    sizes[axis] = length
    return slice(input, begins, sizes)

