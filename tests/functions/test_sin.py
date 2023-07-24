import numpy as np
from mindtorch import Tensor
from mindtorch._functions import sin

def test_sin():
    x = Tensor(np.array(1.0), requires_grad=True)
    y = sin(x)
    y.backward(create_graph=True)

    for i in range(3):
        gx = x.grad
        x.zero_grad()
        gx.backward(create_graph=True)
        print(x.grad)

