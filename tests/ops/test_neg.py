import numpy as np
from mindtorch import Tensor

def test_neg():
    x = Tensor(np.array(2.0))
    y = -x
    print(y)  # Tensor(-2.0)