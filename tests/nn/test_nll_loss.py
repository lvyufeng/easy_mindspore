import unittest
import numpy as np
from easy_mindspore.nn import functional as F
from easy_mindspore import tensor
import torch


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

