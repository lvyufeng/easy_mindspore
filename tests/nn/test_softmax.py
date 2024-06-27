import unittest
import numpy as np
from easy_mindspore import tensor
from easy_mindspore.nn import functional as F
import torch

class TestLogSoftmax(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        y2 = torch.nn.functional.log_softmax(torch.tensor(x))
        y = F.log_softmax(tensor(x))

        assert np.allclose(y.numpy(), y2.numpy())
