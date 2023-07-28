from mindtorch import Tensor, tensor
from mindtorch.autograd import Function, Context
from mindtorch._operations import raw_mul, raw_square, raw_add, raw_neg, raw_sub, \
    raw_div, raw_pow, raw_sin, raw_cos, raw_tanh, raw_exp, raw_log
from .array import sum_to

def ensure_tensor(tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return tensor(tensorable)

class Square(Function):
    @staticmethod
    def forward(ctx: Context, x):
        y = raw_square(x)
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
        y = raw_add(x0, x1)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x0_shape, x1_shape = ctx.saved_tensors
        gx0, gx1 = gy, gy
        if x0_shape != x1_shape:  # for broadcaset
            gx0 = sum_to(gx0, x0_shape)
            gx1 = sum_to(gx1, x1_shape)
        return gx0, gx1

def add(x0, x1):
    x1 = ensure_tensor(x1)
    return Add.apply(x0, x1)

class Mul(Function):
    @staticmethod
    def forward(ctx: Context, x0, x1):
        y = raw_mul(x0, x1)
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
        return raw_neg(x)

    @staticmethod
    def backward(ctx: Context, gy):
        return -gy


def neg(x):
    return Neg.apply(x)


class Sub(Function):
    @staticmethod
    def forward(ctx: Context, x0, x1):
        ctx.save_for_backward(x0._shape, x1._shape)
        y = raw_sub(x0, x1)
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
        y = raw_div(x0, x1)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x0, x1 = ctx.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = sum_to(gx0, x0._shape)
            gx1 = sum_to(gx1, x1._shape)
        return gx0, gx1

def div(x0, x1):
    x1 = ensure_tensor(x1)
    x0 = x0.float()
    x1 = x1.float()
    return Div.apply(x0, x1)


def rdiv(x0, x1):
    x1 = ensure_tensor(x1)
    x0 = x0.float()
    x1 = x1.float()
    return div(x1, x0)


class Pow(Function):
    @staticmethod
    def forward(ctx: Context, x, c):
        ctx.save_for_backward(c)
        y = raw_pow(x, c)
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
        y = raw_sin(x)
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
        y = raw_cos(x)
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
        y = raw_tanh(x)
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
        y = raw_exp(x)
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
        y = raw_log(x)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, = ctx.inputs
        gx = gy / x
        return gx


def log(x):
    return Log.apply(x)
