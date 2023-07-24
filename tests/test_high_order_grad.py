import numpy as np
from mindtorch import Tensor

def f(x):
    return x ** 4 - 2 * x ** 2

def test_two_order():
    x = Tensor(np.array(2.0), requires_grad=True)
    y = f(x)
    y.backward(create_graph=True)
    print(x.grad)
    gx = x.grad
    x.zero_grad()
    gx.backward()
    print(x.grad)
