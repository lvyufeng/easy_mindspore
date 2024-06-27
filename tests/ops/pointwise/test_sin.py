import numpy as np
from easy_mindspore import Tensor
from easy_mindspore.ops import sin

def test_sin():
    x = Tensor(np.array(1.0))
    y = sin(x)
    print(y)
