import numpy as np
from mindtorch import Tensor

def test_pow():
    x = Tensor(np.array(2.0), requires_grad=True)
    y = x ** 3
    y.backward()
    print(y)  # Tensor(8.0)
