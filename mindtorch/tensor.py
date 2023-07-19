import numpy as np
import warnings
from typing import List, NamedTuple, Callable, Optional, Union
from mindspore._c_expression import TensorNode
from mindspore._c_expression import Tensor as Array  # pylint: disable=E0611
from mindspore.common import dtype as mstype

from mindtorch import BACKEND
from ._operations import raw_sum, raw_zeros_like, raw_add, raw_mul, raw_neg, raw_matmul, raw_strided_slice, \
    raw_strided_slice_grad, raw_tile
from .utils import slice_helper, ASCEND_DTYPE_MAP

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[Array], Array]


Arrayable = Union[float, list, int, Array]

def ensure_array(arrayable: Arrayable, dtype) -> Array:
    if isinstance(arrayable, Tensor):
        return arrayable.data
    if isinstance(arrayable, (Array, TensorNode)):
        return arrayable
    if dtype is None:
        if BACKEND == 'Ascend':
            if isinstance(arrayable, (list, tuple)):
                arrayable = np.array(arrayable)

            if isinstance(arrayable, (int, float)):
                origin_dtype = type(arrayable)
                dtype = ASCEND_DTYPE_MAP[origin_dtype]
                warnings.warn(f'Tensor dtype will auto change from system type {origin_dtype} to tensor dtype {dtype} on Ascend.')
            elif isinstance(arrayable, np.ndarray):
                origin_dtype = str(arrayable.dtype)
                dtype = ASCEND_DTYPE_MAP.get(origin_dtype, None)
                warnings.warn(f'Tensor dtype will auto change from numpy dtyensure_tensorpe {origin_dtype} to tensor dtype {dtype} on Ascend.')
            return Array(arrayable, dtype)
        return Array(arrayable)
    return Array(arrayable, dtype)


Tensorable = Union['Tensor', float, np.ndarray]


def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor:
    """mindtorch defined Tensor."""

    def __init__(self,
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None,
                 dtype=None
                 ) -> None:
        self.data = ensure_array(data, dtype)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []

        self.shape = self.data.shape if isinstance(self.data, Array) else self.data.get_shape()
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        self.grad = Tensor(raw_zeros_like(self.data))

    def __repr__(self) -> str:
        return f"Tensor({self.data.asnumpy()}, requires_grad={self.requires_grad})"

    def __add__(self, other) -> 'Tensor':
        """
        geys called if I do t + other
        """
        return add(self, ensure_tensor(other))

    def __radd__(self, other) -> 'Tensor':
        """ gets called if I do other + t """
        return add(ensure_tensor(other), self)

    def __iadd__(self, other) -> 'Tensor':
        """
        when we do t += other
        """
        self.data = raw_add(self.data, ensure_tensor(other).data)
        # Invalidate the gradient
        self.grad = None
        return self

    def __isub__(self, other) -> 'Tensor':
        self.data = sub(self, ensure_tensor(other)).data
        # Invalidate the gradient
        self.grad = None
        return self

    def __imul__(self, other) -> 'Tensor':
        self.data *= ensure_tensor(other).data
        # Invalidate the gradient
        self.grad = None
        return self

    def __mul__(self, other) -> 'Tensor':
        return mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> 'Tensor':
        return mul(ensure_tensor(other), self)

    def __neg__(self) -> 'Tensor':
        return neg(self)

    def __sub__(self, other) -> 'Tensor':
        return sub(self, ensure_tensor(other))

    def __rsub__(self, other) -> 'Tensor':
        return sub(ensure_tensor(other), self)

    def __matmul__(self, other) -> 'Tensor':
        return matmul(self, other)

    def __getitem__(self, idxs) -> 'Tensor':
        return slice(self, idxs)

    def sum(self) -> 'Tensor':
        # raise NotImplementedError
        return tensor_sum(self)

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == []:
                grad = Tensor(1, dtype=self.dtype)
            else:
                raise RuntimeError("grad must specified for non-0-tensor")

        self.grad.data = raw_add(self.grad.data, grad.data)

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def tolist(self):
        return self.data.asnumpy().tolist()

    @property
    def dtype(self):
        return self.data.dtype


def tensor_sum(t: Tensor) -> Tensor:
    """
    Takes a tensor and returns the 0-tensor
    that's the sum of all its elements.
    """
    data = raw_sum(t.data)
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad: Array) -> Array:
            """
            grad is necessarily a 0-tensor, so each input element
            contributes that much
            """
            # print(grad.shape, t.data.shape)
            # return _mul(grad, _ones_like(t.data))
            return raw_mul(grad, Array(np.ones(t.data.shape)))

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def add(t1: Tensor, t2: Tensor) -> Tensor:
    data = raw_add(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: Array) -> Array:
            # Idea: [1,2,3] + [4,5,6] => [5,7,9]
            # Handle the broadcasting properly
            # Sum out added dims
            ndims_added = len(grad.shape) - len(t1.data.shape)
            for _ in range(ndims_added):
                grad = raw_sum(grad, axis=0)

            # Sum across broadcasted (but non-added dims)
            # (2,3) + (1,3) => (2,3) grad(2,3)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = raw_sum(grad, axis=i, keepdims=True)

            return grad
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: Array) -> Array:
            ndims_added = len(grad.shape) - len(t2.data.shape)
            for _ in range(ndims_added):
                grad = raw_sum(grad, axis=0)
             # Sum across broadcasted (but non-added dims)
            # (2,3) + (1,3) => (2,3) grad(2,3)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = _sum(grad, axis=i, keepdims=True)

            return grad
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)


def mul(t1: Tensor, t2: Tensor) -> Tensor:
    """
    y = (a + eps) * b = a * b + (eps * b * dL/dy)
    gradient_y = 5
    have dL/dy
    dL/da = dL/dy * dy/da(b)
    """
    data = raw_mul(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: Array) -> Array:
            grad = raw_mul(grad, t2.data)

            ndims_added = len(grad.shape) - len(t1.data.shape)
            for _ in range(ndims_added):
                grad = raw_sum(grad, axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = raw_sum(grad, axis=i, keepdims=True)

            return grad
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: Array) -> Array:
            grad = raw_mul(grad, t1.data)
            ndims_added = len(grad.shape) - len(t2.data.shape)
            for _ in range(ndims_added):
                grad = raw_sum(grad, axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = raw_sum(grad, axis=i, keepdims=True)

            return grad
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)


def neg(t: Tensor) -> Tensor:
    data = raw_neg(t.data)
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda x: raw_neg(x))]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + -t2


def matmul(t1: Tensor, t2: Tensor) -> Tensor:
    """
    if t1 is (n1,m1) t2 is (m1,m2) then t1 @ t2 is (n1,m2)
    so grad3 is (n1,m2)

    if t3 = t1 @ t2 and grad3 is the gradient of some function wrt t3, then
        grad1 = grad @ t2.T
        grad2 = t1.T @ grad
    """
    data = raw_matmul(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: Array) -> Array:
            return raw_matmul(grad, t2.data, transpose_b=True)
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: Array) -> Array:
            return raw_matmul(t1.data, grad, transpose_a=True)
        depends_on.append(Dependency(t2, grad_fn2))
    return Tensor(data, requires_grad, depends_on)


def slice(t: Tensor, idx) -> Tensor:
    """
    t2 = t1[3:4,4:4]
    """
    args = slice_helper(idx)
    data = raw_strided_slice(t.data, *args)
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad: Array) -> Array:
            return raw_strided_slice_grad(grad, data.shape, *args)
        depends_on = Dependency(t, grad_fn)
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)
