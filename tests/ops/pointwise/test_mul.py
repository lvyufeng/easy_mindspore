import numpy as np
from easy_mindspore import Tensor

def test_mul_add():
    a = Tensor(np.array(3.0))
    b = Tensor(np.array(2.0))
    c = Tensor(np.array(1.0))

    print(a, b, c)
    # y = add(mul(a, b), c)
    y = a * b + c
    print(y)
