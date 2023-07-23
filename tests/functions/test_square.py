import unittest
import numpy as np
from mindtorch import Tensor
from mindtorch._functions import square
from mindtorch._operations import raw_sub, raw_add, raw_div

def numerical_diff(f, x, eps=1e-4):
    x0 = Tensor(raw_sub(x.data, eps))
    x1 = Tensor(raw_add(x.data, eps))
    y0 = f(x0)
    y1 = f(x1)
    return raw_div(raw_sub(y1.data, y0.data), (2 * eps))


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Tensor(np.array(2.0), requires_grad=True)
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data.asnumpy(), expected)

    def test_backward(self):
        x = Tensor(np.array(3.0), requires_grad=True)
        y = square(x)
        y.backward(Tensor(np.ones(x.shape)))
        expected = np.array(6.0)
        self.assertEqual(x.grad.asnumpy(), expected)

    def test_gradient_check(self):
        x = Tensor(np.random.rand(1), requires_grad=True)
        y = square(x)
        y.backward(Tensor(np.ones(x.shape)))
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad.asnumpy(), num_grad.asnumpy())
        self.assertTrue(flg)
