import unittest
import numpy as np
from mindtorch.nn import functional as F
from mindtorch import tensor
import torch
from .. import gradient_check


class TestNLLLoss(unittest.TestCase):

    def test_forward1(self):
        n, c = 32, 10
        x = np.random.randn(n, c).astype('f')
        y = np.random.randint(0, c, (n,))

        expected = torch.nn.functional.nll_loss(torch.tensor(x), torch.tensor(y))
        y = F.nll_loss(tensor(x), tensor(y))

        assert np.allclose(y.numpy(), expected.numpy())

    def test_forward2(self):
        n, c = 32, 10
        x = np.random.randn(n, c).astype('f')
        y = np.random.randint(0, c, (n,))

        expected = torch.nn.functional.nll_loss(torch.tensor(x), torch.tensor(y), ignore_index=1)
        y = F.nll_loss(tensor(x), tensor(y), ignore_index=1)

        assert np.allclose(y.numpy(), expected.numpy())

    def test_forward3(self):
        n, c = 32, 10
        x = np.random.randn(n, c).astype('f')
        y = np.random.randint(0, c, (n,))

        expected = torch.nn.functional.nll_loss(torch.tensor(x), torch.tensor(y), reduction='none')
        y = F.nll_loss(tensor(x), tensor(y), reduction='none')

        assert np.allclose(y.numpy(), expected.numpy())

    def test_backward1(self):
        n, c = 32, 10
        x_np = np.random.randn(n, c).astype('f')
        y = np.random.randint(0, c, (n,))

        x = tensor(x_np, requires_grad=True)
        x_e = torch.tensor(x_np, requires_grad=True)
        expected = torch.nn.functional.nll_loss(x_e, torch.tensor(y))
        expected.backward()

        y = F.nll_loss(x, tensor(y))
        y.backward()

        assert np.allclose(x.grad.numpy(), x_e.grad.numpy())

    # def test_backward1(self):
    #     n, c, h, w = 1, 5, 16, 16
    #     ksize, stride, pad = 2, 2, 0
    #     x = tensor(np.random.randn(n, c, h, w).astype('f') * 1000, requires_grad=True)
    #     f = lambda x: F.max_pool2d(x, ksize, stride, pad)
    #     self.assertTrue(gradient_check(f, x))