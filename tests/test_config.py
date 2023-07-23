import numpy as np
from mindtorch import Tensor
from mindtorch._functions import add, square
from mindtorch.config import using_config, no_grad

def test_weak_ref():
    x0 = Tensor(np.array(1.0), requires_grad=True)
    x1 = Tensor(np.array(1.0), requires_grad=True)
    t = add(x0, x1)
    y = add(x0, t)
    y.backward()

    assert y.grad is None
    assert t.grad is None
    print(y.grad, t.grad)  # None None
    print(x0.grad, x1.grad)  # 2.0 1.0


def test_enable_backprop_false():
    with using_config('enable_backprop', True):
        x = Tensor(np.array(2.0), requires_grad=True)
        y = square(x)
    print(y.requires_grad)

def test_no_grad():
    with no_grad():
        x = Tensor(np.array(2.0))
        y = square(x)
    print(y.requires_grad)
