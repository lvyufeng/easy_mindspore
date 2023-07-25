from mindtorch.autograd import Function
from mindtorch._operations import raw_sum, raw_reshape, raw_transpose, raw_broadcast_to, \
    raw_matmul, raw_add, raw_strided_slice, raw_strided_slice_grad
from mindtorch._functions import utils

# =============================================================================
# Tensor operations: reshape / transpose / expand_dims / flatten
# =============================================================================
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = raw_reshape(x, self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return x
    return Reshape(shape)(x)


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
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = raw_transpose(x, self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = utils.argsort([ax % axes_len for ax in self.axes])
        return transpose(gy, inv_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)

def expand_dims(x, axis):
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))

# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = raw_sum(x, axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis,
                                        self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return x
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = raw_broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return x
    return BroadcastTo(shape)(x)


class MatMul(Function):
    def __init__(self, transpose_a, transpose_b):
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def forward(self, x, W):
        y = raw_matmul(x, W, self.transpose_a, self.transpose_b)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W, transpose_b=True)
        gW = matmul(x, gy, transpose_a=True)
        return gx, gW


def matmul(x, w, transpose_a=False, transpose_b=False):
    return MatMul(transpose_a, transpose_b)(x, w)


class Linear(Function):
    def forward(self, x, W, b):
        y = raw_matmul(x, W)
        if b is not None:
            y = raw_add(y, b)
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W, transpose_b=True)
        gW = matmul(x, gy, transpose_a=True)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None  # Release t.data (ndarray) for memory efficiency
    return y

class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        slices = utils.slice_helper(self.slices)
        y = raw_strided_slice(x, *slices)
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        slices = utils.slice_helper(self.slices)
        gx = raw_strided_slice_grad(gy, self.in_shape, *slices)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)
