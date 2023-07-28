import unittest
import numpy as np
from mindtorch import tensor
from mindtorch._functions import matmul
from .. import gradient_check


class TestMatmul(unittest.TestCase):

    def test_forward1(self):
        x = tensor(np.array([[1., 2., 3.], [4., 5., 6.]]), requires_grad=True)
        w = tensor(x.numpy().T, requires_grad=True)
        y = matmul(x, w)
        res = y.numpy()
        expected = np.array([[14., 32.], [32., 77.]])
        self.assertTrue(np.allclose(res, expected))

    def test_backward1(self):
        x = tensor(np.random.randn(3, 2), requires_grad=True)
        w = tensor(np.random.randn(2, 3), requires_grad=True)
        f = lambda x: matmul(x, w)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x = tensor(np.random.randn(10, 1), requires_grad=True)
        w = tensor(np.random.randn(1, 5), requires_grad=True)
        f = lambda w: matmul(x, w)
        self.assertTrue(gradient_check(f, w))