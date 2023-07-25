import numpy as np
from mindtorch import Tensor

def test_mul_add():
    a = Tensor(np.array(3.0), requires_grad=True)
    b = Tensor(np.array(2.0), requires_grad=True)
    c = Tensor(np.array(1.0), requires_grad=True)

    # y = add(mul(a, b), c)
    y = a * b + c
    print(y)
    y.backward()

    print(y)
    print(a.grad)
    print(b.grad)