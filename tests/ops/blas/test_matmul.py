import unittest
import numpy as np
from easy_mindspore import tensor
from easy_mindspore.ops import matmul


import mindspore

class TestMatmul(unittest.TestCase):

    def test_forward1(self):
        x = tensor(np.array([[1., 2., 3.], [4., 5., 6.]]))
        w = tensor(x.numpy().T)
        y = matmul(x, w)
        res = y.numpy()
        expected = np.array([[14., 32.], [32., 77.]])
        self.assertTrue(np.allclose(res, expected))

    def test_backward(self):
        x = tensor(np.array([[1., 2., 3.], [4., 5., 6.]]))
        w = tensor(x.numpy().T)
        def forward(x, w):
            return matmul(x, w)

        grad_fn = mindspore.grad(forward, None, (w,))
        grad = grad_fn(x, w)
        print(grad)
