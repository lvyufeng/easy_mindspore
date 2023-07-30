import unittest
import mindtorch
import mindtorch.nn.functional as F
import numpy as np
import torch
from .. import gradient_check

class TestConv2d(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 15, 15
        o, k, s, p = 8, (3, 3), (1, 1), (1, 1)
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        y = F.conv2d(mindtorch.tensor(x), mindtorch.tensor(W), b, s, p)
        expected = torch.nn.functional.conv2d(torch.tensor(x), torch.tensor(W), b, s, p)
        assert np.allclose(y.numpy(), expected.numpy())

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        o, k, s, p = 8, (3, 3), (2, 1), (2, 1)
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        expected = torch.nn.functional.conv2d(torch.tensor(x), torch.tensor(W), b, s, p)
        y = F.conv2d(mindtorch.tensor(x), mindtorch.tensor(W), b, s, p)
        assert np.allclose(y.numpy(), expected.numpy())

    def test_forward3(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        expected = torch.nn.functional.conv2d(torch.tensor(x), torch.tensor(W), b, s, p)
        y = F.conv2d(mindtorch.tensor(x), mindtorch.tensor(W), b, s, p)
        assert np.allclose(y.numpy(), expected.numpy())

    def test_forward4(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = np.random.randn(o).astype('f')
        expected = torch.nn.functional.conv2d(torch.tensor(x), torch.tensor(W), torch.tensor(b), s, p)
        y = F.conv2d(mindtorch.tensor(x), mindtorch.tensor(W), mindtorch.tensor(b), s, p)
        assert np.allclose(y.numpy(), expected.numpy())

    def test_backward1(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = mindtorch.tensor(np.random.randn(n, c, h, w), requires_grad=True)
        W = mindtorch.tensor(np.random.randn(o, c, k[0], k[1]))
        b = mindtorch.tensor(np.random.randn(o))
        f = lambda x: F.conv2d(x, W, b, s, p)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = mindtorch.tensor(np.random.randn(n, c, h, w))
        W = mindtorch.tensor(np.random.randn(o, c, k[0], k[1]))
        b = mindtorch.tensor(np.random.randn(o), requires_grad=True)
        f = lambda b: F.conv2d(x, W, b, s, p)
        self.assertTrue(gradient_check(f, b))

    def test_backward3(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = mindtorch.tensor(np.random.randn(n, c, h, w))
        W = mindtorch.tensor(np.random.randn(o, c, k[0], k[1]), requires_grad=True)
        b = mindtorch.tensor(np.random.randn(o))
        f = lambda W: F.conv2d(x, W, b, s, p)
        self.assertTrue(gradient_check(f, W))
