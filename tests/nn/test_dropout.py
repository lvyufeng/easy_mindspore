import unittest
import numpy as np
from mindtorch import tensor
from mindtorch.nn import functional as F
from .. import gradient_check


class TestDropout(unittest.TestCase):

    def test_forward1(self):
        x = tensor(np.random.randn(100, 100))
        y = F.dropout(x, p=0.0)
        res = x == y
        self.assertTrue(res)

    def test_forward2(self):
        x = tensor(np.random.randn(100, 100))
        y = F.dropout(x, p=0.5)
        drop_num = (y == 0).sum()
        print(drop_num, x.size())
    # def test_backward1(self):
    #     x_data = np.random.randn(10, 10)

    #     def f(x):
    #         np.random.seed(0)
    #         return F.dropout(x, 0.5)

    #     self.assertTrue(gradient_check(f, x_data))

    # def test_backward2(self):
    #     x_data = np.random.randn(10, 20)

    #     def f(x):
    #         np.random.seed(0)
    #         return F.dropout(x, 0.99)

    #     self.assertTrue(gradient_check(f, x_data))

    # def test_backward3(self):
    #     x_data = np.random.randn(10, 10)

    #     def f(x):
    #         np.random.seed(0)
    #         return F.dropout(x, 0.0)

    #     self.assertTrue(gradient_check(f, x_data))