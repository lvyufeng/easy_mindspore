from mindtorch import Tensor
from mindtorch.autograd import Function
from mindtorch._operations import raw_mul, raw_square, raw_add, raw_neg, raw_sub, \
    raw_div, raw_pow, raw_sin, raw_cos, raw_tanh, raw_exp, raw_log
from .array import sum_to

def ensure_tensor(tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)

class Square(Function):
    def forward(self, x):
        y = raw_square(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)

class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0._shape, x1._shape
        y = raw_add(x0, x1)
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:  # for broadcaset
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1

def add(x0, x1):
    x1 = ensure_tensor(x1)
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0, x1):
        y = raw_mul(x0, x1)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = sum_to(gx0, x0._shape)
            gx1 = sum_to(gx1, x1._shape)
        return gx0, gx1


def mul(x0, x1):
    x1 = ensure_tensor(x1)
    return Mul()(x0, x1)

class Neg(Function):
    def forward(self, x):
        return raw_neg(x)

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0._shape, x1._shape
        y = raw_sub(x0, x1)
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1



def sub(x0, x1):
    x1 = ensure_tensor(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = ensure_tensor(x1)
    return sub(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = raw_div(x0, x1)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = sum_to(gx0, x0._shape)
            gx1 = sum_to(gx1, x1._shape)
        return gx0, gx1

def div(x0, x1):
    x1 = ensure_tensor(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = ensure_tensor(x1)
    return div(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = raw_pow(x, self.c)
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)

# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================
class Sin(Function):
    def forward(self, x):
        y = raw_sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = raw_cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y = raw_tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Exp(Function):
    def forward(self, x):
        y = raw_exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        y = raw_log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)
