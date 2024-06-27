import numpy as np
import easy_mindspore
from easy_mindspore import tensor

def test_radd():
    x = tensor(np.array(2.0))
    y = x + 3.0
    print(y)

def test_add():
    x = tensor(np.array(2.0))
    y = 3.0 + x
    print(y)
