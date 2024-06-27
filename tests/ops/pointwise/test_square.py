import numpy as np
from easy_mindspore import Tensor
from easy_mindspore.ops import square

def test_forward():
    x = Tensor(np.array(2.0))
    y = square(x)
    expected = np.array(4.0)
    assert y.numpy() == expected
