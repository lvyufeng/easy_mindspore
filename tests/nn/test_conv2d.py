import unittest
import easy_mindspore
import easy_mindspore.nn.functional as F
import numpy as np
import torch

class TestConv2d(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 15, 15
        o, k, s, p = 8, (3, 3), (1, 1), (1, 1)
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        y = F.conv2d(easy_mindspore.tensor(x), easy_mindspore.tensor(W), b, s, p)
        expected = torch.nn.functional.conv2d(torch.tensor(x), torch.tensor(W), b, s, p)
        assert np.allclose(y.numpy(), expected.numpy(), 1e-5, 1e-5)

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        o, k, s, p = 8, (3, 3), (2, 1), (2, 1)
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        expected = torch.nn.functional.conv2d(torch.tensor(x), torch.tensor(W), b, s, p)
        y = F.conv2d(easy_mindspore.tensor(x), easy_mindspore.tensor(W), b, s, p)
        assert np.allclose(y.numpy(), expected.numpy())

    def test_forward3(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = None
        expected = torch.nn.functional.conv2d(torch.tensor(x), torch.tensor(W), b, s, p)
        y = F.conv2d(easy_mindspore.tensor(x), easy_mindspore.tensor(W), b, s, p)
        assert np.allclose(y.numpy(), expected.numpy())

    def test_forward4(self):
        n, c, h, w = 1, 5, 20, 15
        o, k, s, p = 3, (5, 3), 1, 3
        x = np.random.randn(n, c, h, w).astype('f')
        W = np.random.randn(o, c, k[0], k[1]).astype('f')
        b = np.random.randn(o).astype('f')
        expected = torch.nn.functional.conv2d(torch.tensor(x), torch.tensor(W), torch.tensor(b), s, p)
        y = F.conv2d(easy_mindspore.tensor(x), easy_mindspore.tensor(W), easy_mindspore.tensor(b), s, p)
        assert np.allclose(y.numpy(), expected.numpy())
