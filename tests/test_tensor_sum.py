import unittest
from mindtorch.tensor import Tensor
import time

class TestTensorSum(unittest.TestCase):
    def test_simple_sum(self):
        s = time.time()
        t1 = Tensor([1., 2., 3.], requires_grad=True)
        t2 = t1.sum()

        t2.backward()
        t = time.time()
        print(t - s)
        assert t1.grad.tolist() == [1., 1., 1.]

    def test_sum_with_grad(self):
        t1 = Tensor([1., 2., 3.], requires_grad=True)
        s = time.time()
        t2 = t1.sum()

        t2.backward(Tensor(3.))
        t = time.time()
        print(t - s)
        t1 = Tensor([1., 2., 3.], requires_grad=True)
        s = time.time()
        t2 = t1.sum()

        t2.backward(Tensor(3.))
        t = time.time()
        print(t - s)
        assert t1.grad.tolist() == [3., 3., 3.]

    def test_sum_with_grad_torch(self):
        import torch
        t1 = torch.tensor([1., 2., 3.], requires_grad=True)
        s = time.time()
        t2 = t1.sum()

        t2.backward(torch.tensor(3.))
        t = time.time()
        print(t - s)
        t1 = torch.tensor([1., 2., 3.], requires_grad=True)
        s = time.time()
        t2 = t1.sum()

        t2.backward(torch.tensor(3.))
        t = time.time()
        print(t - s)
        assert t1.grad.tolist() == [3., 3., 3.]
