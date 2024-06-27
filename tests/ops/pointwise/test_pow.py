import numpy as np
from easy_mindspore import Tensor
from easy_mindspore._tensor import Tensor

def test_pow():
    x = Tensor(np.array(2.0))
    y = x ** 3
    print(y)  # Tensor(8.0)
