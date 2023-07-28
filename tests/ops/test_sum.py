import unittest
import numpy as np
from mindtorch import tensor, dtype
from mindtorch._functions import sum
from .. import gradient_check

class TestSum(unittest.TestCase):

    def test_datatype(self):
        x = tensor(np.random.rand(10))
        y = sum(x)
        self.assertFalse(np.isscalar(y.numpy()))

    def test_forward1(self):
        x = tensor(np.array(2.0))
        y = sum(x)
        expected = np.sum(x.numpy())
        self.assertTrue(np.allclose(y.numpy(), expected))

    def test_forward2(self):
        x = tensor(np.random.rand(10, 20, 30))
        y = sum(x, axis=1)
        expected = np.sum(x.numpy(), axis=1)
        self.assertTrue(np.allclose(y.numpy(), expected))

    def test_forward3(self):
        x = tensor(np.random.rand(10, 20, 30))
        y = sum(x, axis=1, keepdims=True)
        expected = np.sum(x.numpy(), axis=1, keepdims=True)
        self.assertTrue(np.allclose(y.numpy(), expected))

    def test_backward1(self):
        x_data = tensor(np.random.rand(10), requires_grad=True)
        f = lambda x: sum(x)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = tensor(np.random.rand(10, 10), requires_grad=True)
        f = lambda x: sum(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = tensor(np.random.rand(10, 20, 20), requires_grad=True)
        f = lambda x: sum(x, axis=2)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward4(self):
        x_data = tensor(np.random.rand(10, 20, 20), requires_grad=True)
        f = lambda x: sum(x, axis=None)
        self.assertTrue(gradient_check(f, x_data))


# class TestSumTo(unittest.TestCase):

#     def test_forward1(self):
#         x = Variable(np.random.rand(10))
#         y = F.sum_to(x, (1,))
#         expected = np.sum(x.data)
#         self.assertTrue(array_allclose(y.data, expected))

#     def test_forward2(self):
#         x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
#         y = F.sum_to(x, (1, 3))
#         expected = np.sum(x.data, axis=0, keepdims=True)
#         self.assertTrue(array_allclose(y.data, expected))

#     def test_forward3(self):
#         x = Variable(np.random.rand(10))
#         y = F.sum_to(x, (10,))
#         expected = x.data  # 同じ形状なので何もしない
#         self.assertTrue(array_allclose(y.data, expected))

#     def test_backward1(self):
#         x_data = np.random.rand(10)
#         f = lambda x: F.sum_to(x, (1,))
#         self.assertTrue(gradient_check(f, x_data))

#     def test_backward2(self):
#         x_data = np.random.rand(10, 10) * 10
#         f = lambda x: F.sum_to(x, (10,))
#         self.assertTrue(gradient_check(f, x_data))

#     def test_backward3(self):
#         x_data = np.random.rand(10, 20, 20) * 100
#         f = lambda x: F.sum_to(x, (10,))
#         self.assertTrue(gradient_check(f, x_data))

#     def test_backward4(self):
#         x_data = np.random.rand(10)
#         f = lambda x: F.sum_to(x, (10,)) + 1
#         self.assertTrue(gradient_check(f, x_data))