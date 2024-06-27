import unittest
import numpy as np
from easy_mindspore import tensor
from easy_mindspore.nn import functional as F


class TestDropout(unittest.TestCase):

    def test_forward1(self):
        x = tensor(np.random.randn(100, 100))
        y = F.dropout(x, p=0.0)
        res = x == y
        self.assertTrue(res)

    def test_forward2(self):
        x = tensor(np.random.randn(100, 100))
        y = F.dropout(x, p=0.5)
        print(y == 0)
        drop_num = (y == 0).sum()
        print(drop_num, x.size())
