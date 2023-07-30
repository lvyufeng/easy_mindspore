import unittest
import numpy as np
from mindtorch.nn import functional as F
from mindtorch import tensor
import torch
from .. import gradient_check


class TestPooling(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = F.max_pool2d(tensor(x), ksize, stride, pad)
        expected = torch.nn.functional.max_pool2d(torch.tensor(x), ksize, stride, pad)

        assert np.allclose(y.numpy(), expected.numpy())

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = F.max_pool2d(tensor(x), ksize, stride, pad, ceil_mode=True)
        expected = torch.nn.functional.max_pool2d(torch.tensor(x), ksize, stride, pad, ceil_mode=True)

        assert np.allclose(y.numpy(), expected.numpy())

    def test_forward3(self):
        n, c, h, w = 1, 5, 15, 15
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y, yi = F.max_pool2d(tensor(x), ksize, stride, pad, ceil_mode=True, return_indices=True)
        e, ei = torch.nn.functional.max_pool2d(torch.tensor(x), ksize, stride, pad, ceil_mode=True, return_indices=True)

        assert np.allclose(y.numpy(), e.numpy())
        assert np.allclose(yi.numpy(), ei.numpy())

    def test_backward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = tensor(np.random.randn(n, c, h, w).astype('f') * 1000, requires_grad=True)
        f = lambda x: F.max_pool2d(x, ksize, stride, pad)
        self.assertTrue(gradient_check(f, x))

    def test_backward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = tensor(np.random.randn(n, c, h, w).astype('f') * 1000, requires_grad=True)
        f = lambda x: F.max_pool2d(x, ksize, stride, pad)
        self.assertTrue(gradient_check(f, x))