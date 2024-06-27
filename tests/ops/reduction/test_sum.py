import unittest
import numpy as np
from easy_mindspore import tensor, dtype
from easy_mindspore.ops import sum

class TestSum(unittest.TestCase):

    def test_datatype(self):
        x = tensor(np.random.rand(10))
        y = sum(x)
        self.assertFalse(np.isscalar(y.numpy()))

    def test_forward1(self):
        x = tensor(np.array(2.0))
        print(x.dtype)
        y = sum(x)
        expected = np.sum(x.numpy())
        self.assertTrue(np.allclose(y.numpy(), expected))

    def test_forward2(self):
        x = tensor(np.random.rand(10, 20, 30))
        y = sum(x, dim=1)
        expected = np.sum(x.numpy(), axis=1)
        self.assertTrue(np.allclose(y.numpy(), expected))

    def test_forward3(self):
        x = tensor(np.random.rand(10, 20, 30))
        y = sum(x, dim=1, keepdim=True)
        expected = np.sum(x.numpy(), axis=1, keepdims=True)
        self.assertTrue(np.allclose(y.numpy(), expected))
