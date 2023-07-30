from mindtorch.autograd import Function, Context
from mindtorch._operations import raw_sum, raw_reshape, raw_transpose, raw_broadcast_to, \
    raw_matmul, raw_strided_slice, raw_strided_slice_grad, raw_argmax, raw_equal, \
    raw_cast
from mindtorch._functions import utils
from mindtorch import dtype
from .utils import ensure_tensor
# =============================================================================
# Tensor operations: reshape / transpose / expand_dims / flatten
# =============================================================================
class Reshape(Function):
    @staticmethod
    def forward(ctx: Context, x, shape):
        ctx.save_for_backward(x.shape)
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
            return transpose(gy)

        axes_len = len(axes)
        inv_axes = utils.argsort([ax % axes_len for ax in axes])
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    if axes is None:
        axes = (1, 0)
    return Transpose.apply(x, axes=axes)

def expand_dims(x, axis):
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))

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

def sum(x, axis=None, keepdims=False):
    if x.dtype == dtype.bool:
        x = x.long()
    return Sum.apply(x, axis=axis, keepdims=keepdims)

def mean(x, axis=None, keepdims=False):
    y = sum(x, axis, keepdims)
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


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, x, w, transpose_a, transpose_b):
        y = raw_matmul(x, w, transpose_a, transpose_b)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, W = ctx.inputs
        gx = matmul(gy, W, transpose_b=True)
        gW = matmul(x, gy, transpose_a=True)
        return gx, gW


def matmul(x, w, transpose_a=False, transpose_b=False):
    return MatMul.apply(x, w, transpose_a=transpose_a, transpose_b=transpose_b)


class GetItem(Function):
    @staticmethod
    def forward(ctx: Context, x, slices):
        ctx.save_for_backward(slices, x._shape)
        slices = utils.slice_helper(slices)
        y = raw_strided_slice(x, *slices)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        slices, x_shape = ctx.saved_values
        f = GetItemGrad(slices, x_shape)
        return f(gy)


class GetItemGrad(Function):
    @staticmethod
    def forward(ctx: Context, gy, slices, in_shape):
        ctx.save_for_backward(slices)
        slices = utils.slice_helper(slices)
        gx = raw_strided_slice_grad(gy, in_shape, *slices)
        return gx

    @staticmethod
    def backward(ctx: Context, ggx):
        slices, = ctx.saved_tensors
        return get_item(ggx, slices)


def get_item(x, slices):
    return GetItem.apply(x, slices)

class Argmax(Function):
    @staticmethod
    def forward(ctx: Context, x, axis):
        y = raw_argmax(x, axis)
        return y

def argmax(x, axis):
    return Argmax.apply(x, axis=axis, requires_grad=False)

class Equal(Function):
    @staticmethod
    def forward(ctx: Context, x, y):
        return raw_equal(x, y)

def equal(x, y):
    y = ensure_tensor(y, x.dtype)
    return Equal.apply(x, y, requires_grad=False)

class Cast(Function):
    @staticmethod
    def forward(ctx: Context, x, dtype):
        ctx.save_for_backward(dtype)
        return raw_cast(x, dtype)
    
    @staticmethod
    def backward(ctx: Context, gy):
        dtype, = ctx.saved_tensors
        return cast(gy, dtype)

def cast(x, dtype):
    return Cast.apply(x, dtype=dtype)
