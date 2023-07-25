import numpy as np
from mindtorch import Tensor

def test_rsub():
    x = Tensor(np.array(2.0))
    y = 2.0 - x
    print(y)

def test_sub():
    x = Tensor(np.array(2.0))
    y = x - 1.0
    print(y)
