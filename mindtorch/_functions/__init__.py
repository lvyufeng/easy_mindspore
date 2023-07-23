from mindtorch import Tensor
from mindtorch.autograd import Function
from mindtorch._operations import raw_mul, raw_square, raw_add, raw_neg, raw_sub, \
    raw_div, raw_pow

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
        y = raw_add(x0, x1)
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = ensure_tensor(x1)
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0, x1):
        y = raw_mul(x0, x1)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0


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
        y = raw_sub(x0, x1)
        return y

    def backward(self, gy):
        return gy, -gy


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