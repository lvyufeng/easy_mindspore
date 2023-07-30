import unittest
import numpy as np
from mindtorch import tensor
from mindtorch.nn import functional as F
import torch
from .. import gradient_check

class TestLogSoftmax(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        y2 = torch.nn.functional.log_softmax(torch.tensor(x))
        y = F.log_softmax(tensor(x))

        assert np.allclose(y.numpy(), y2.numpy())

    def test_backward2(self):
        x = tensor(np.random.randn(10, 10), requires_grad=True)
        f = lambda x: F.log_softmax(x)
        self.assertTrue(gradient_check(f, x, rtol=0.01, atol=0.01))