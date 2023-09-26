import numpy as np
from mindspore import ops, Tensor

from mindtorch import tensor
from mindtorch.autograd import Function, Context
from .array import sum_to
from .utils import ensure_tensor

class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, x, w, transpose_a, transpose_b):
        ctx.save_for_backward(transpose_a, transpose_b)
        x, w = Tensor(x), Tensor(w)
        if transpose_a:
            x = x.T
        if transpose_b:
            w = w.T
        y = ops.matmul(x, w)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, W = ctx.inputs
        transpose_a, transpose_b = ctx.saved_tensors
        gx = matmul(gy, W, transpose_b=not transpose_b)
        gW = matmul(x, gy, transpose_a=not transpose_a)
        if transpose_a:
            gx = gx.T
        if transpose_b:
            gW = gW.T
        return gx, gW


def matmul(x, w, transpose_a=False, transpose_b=False):
    return MatMul.apply(x, w, transpose_a=transpose_a, transpose_b=transpose_b)

_fused_bmm0 = ops.BatchMatMul(transpose_b=True)
_fused_bmm1 = ops.BatchMatMul(transpose_a=True)
def _pack_bmm_grad(x, w, gy):
    gx = _fused_bmm0(gy, w)
    gw = _fused_bmm1(x, gy)
    return gx, gw

class BatchMatMul(Function):
    @staticmethod
    def forward(ctx: Context, x, w):
        # ctx.save_for_backward(transpose_a, transpose_b)
        y = ops.bmm(x, w)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, W = ctx.inputs
        # transpose_a, transpose_b = ctx.saved_tensors
        # gx = batch_matmul(gy, W, transpose_b=not transpose_b)
        # gW = batch_matmul(x, gy, transpose_a=not transpose_a)
        # if transpose_a:
        #     gx = gx.T
        # if transpose_b:
        #     gW = gW.T
        # return gx, gW
        gx, gw = _pack_bmm_grad(x.data, W.data, gy.data)
        return tensor(gx), tensor(gw)


def batch_matmul(x, w):
    return BatchMatMul.apply(x, w)

bmm = batch_matmul

class Square(Function):
    @staticmethod
    def forward(ctx: Context, x):
        y = ops.square(x)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, = ctx.inputs
        gx = 2 * x * gy
        return gx

def square(x):
    return Square.apply(x)

class Add(Function):
    @staticmethod
    def forward(ctx: Context, x0, x1):
        ctx.save_for_backward(x0._shape, x1._shape)
        y = ops.add(x0, x1)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x0_shape, x1_shape = ctx.saved_tensors
        gx0, gx1 = gy, gy
        if x0_shape != x1_shape:  # for broadcaset
            gx0 = sum_to(gx0, x0_shape)
            gx1 = sum_to(gx1, x1_shape)
            # gx0, gx1 = fused_add_grad(gy.data, x0_shape, x1_shape)
            # gx0, gx1 = tensor(gx0), tensor(gx1)
        return gx0, gx1

def add(input, other, *, alpha=1):
    other = ensure_tensor(other)
    if alpha == 1:
        return Add.apply(input, other)
    other = other * alpha
    return Add.apply(input, other)

class Mul(Function):
    @staticmethod
    def forward(ctx: Context, x0, x1):
        y = ops.mul(x0, x1)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x0, x1 = ctx.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1):
    x1 = ensure_tensor(x1)
    return Mul.apply(x0, x1)

class Neg(Function):
    @staticmethod
    def forward(ctx: Context, x):
        return ops.neg(x)

    @staticmethod
    def backward(ctx: Context, gy):
        return -gy


def neg(x):
    return Neg.apply(x)


class Sub(Function):
    @staticmethod
    def forward(ctx: Context, x0, x1):
        ctx.save_for_backward(x0._shape, x1._shape)
        y = ops.sub(x0, x1)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x0_shape, x1_shape = ctx.saved_tensors
        gx0 = gy
        gx1 = -gy
        if x0_shape != x1_shape:  # for broadcast
            gx0 = sum_to(gx0, x0_shape)
            gx1 = sum_to(gx1, x1_shape)
        return gx0, gx1



def sub(x0, x1):
    x1 = ensure_tensor(x1)
    return Sub.apply(x0, x1)


def rsub(x0, x1):
    x1 = ensure_tensor(x1)
    return sub(x1, x0)


class Div(Function):
    @staticmethod
    def forward(ctx: Context, x0, x1):
        y = ops.div(x0, x1)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x0, x1 = ctx.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return gx0, gx1
        # gx0, gx1 = fused_div_grad(x0.data, x1.data, gy.data)
        # return tensor(gx0), tensor(gx1)

def div(x0, x1):
    x1 = ensure_tensor(x1)
    return Div.apply(x0, x1)


def rdiv(x0, x1):
    x1 = ensure_tensor(x1)
    return div(x1, x0)

class Sqrt(Function):
    @staticmethod
    def forward(ctx: Context, x):
        y = ops.sqrt(x)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        y = ctx.outputs[0]()
        _sqrt_grad = ops.Primitive('SqrtGrad')
        gx = _sqrt_grad(y.data, gy.data)
        return tensor(gx, requires_grad=gy.requires_grad)

def sqrt(x):
    x = ensure_tensor(x)
    return Sqrt.apply(x)

class Erf(Function):
    @staticmethod
    def forward(ctx: Context, x):
        y = ops.erf(x)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, = ctx.inputs

        half_root_pi = 2 / sqrt(np.pi)
        x_square = square(x)
        gx = gy * half_root_pi * exp(neg(x_square))

        return gx

def erf(x):
    return Erf.apply(x)


class Pow(Function):
    @staticmethod
    def forward(ctx: Context, x, c):
        ctx.save_for_backward(c)
        y = ops.pow(x, c)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, = ctx.inputs
        c, = ctx.saved_tensors

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow.apply(x, c=c)

# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================
class Sin(Function):
    @staticmethod
    def forward(ctx: Context, x):
        y = ops.sin(x)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, = ctx.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin.apply(x)


class Cos(Function):
    @staticmethod
    def forward(ctx: Context, x):
        y = ops.cos(x)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, = ctx.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos.apply(x)


class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, x):
        y = ops.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        y, = ctx.saved_tensors  # weakref
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh.apply(x)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, x):
        y = ops.exp(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        y, = ctx.saved_tensors  # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp.apply(x)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, x):
        y = ops.log(x)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, = ctx.inputs
        gx = gy / x
        return gx


def log(x):
    return Log.apply(x)
