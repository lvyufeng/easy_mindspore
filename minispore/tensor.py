import numpy as np
import mindspore
from typing import List, NamedTuple, Callable, Optional, Union
from mindspore._c_expression import Tensor as _Tensor  # pylint: disable=E0611
from .ops import _sum, _ones_like, _zeros_like, _add, _mul, _neg, _matmul


class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], _Tensor]


Arrayable = Union[float, list, np.ndarray, _Tensor]


def ensure_array(arrayable: Arrayable, dtype) -> _Tensor:
    if isinstance(arrayable, Tensor):
        return arrayable.data
    if isinstance(arrayable, _Tensor):
        return arrayable
    if dtype is None:
        return _Tensor(arrayable)
    return _Tensor(arrayable, dtype)


Tensorable = Union['Tensor', float, np.ndarray]


def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor:
    """minispore defined Tensor."""

    def __init__(self,
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None,
                 dtype=None
                 ) -> None:
        self.data = ensure_array(data, dtype)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []

        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        self.grad = Tensor(_zeros_like(self.data))

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
        self.data += ensure_tensor(other).data
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

        self.grad.data = _add(self.grad.data, grad.data)

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
    data = _sum(t.data)
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad: _Tensor) -> _Tensor:
            """
            grad is necessarily a 0-tensor, so each input element
            contributes that much
            """
            return _mul(grad, _ones_like(t.data))

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def add(t1: Tensor, t2: Tensor) -> Tensor:
    data = _add(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: _Tensor) -> _Tensor:
            # Idea: [1,2,3] + [4,5,6] => [5,7,9]
            # Handle the broadcasting properly
            # Sum out added dims
            ndims_added = len(grad.shape) - len(t1.data.shape)
            for _ in range(ndims_added):
                grad = _sum(grad, axis=0)

            # Sum across broadcasted (but non-added dims)
            # (2,3) + (1,3) => (2,3) grad(2,3)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = _sum(grad, axis=i, keepdims=True)

            return grad
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: _Tensor) -> _Tensor:
            ndims_added = len(grad.shape) - len(t2.data.shape)
            for _ in range(ndims_added):
                grad = _sum(grad, axis=0)
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
    data = _mul(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: _Tensor) -> _Tensor:
            grad = _mul(grad, t2.data)

            ndims_added = len(grad.shape) - len(t1.data.shape)
            for _ in range(ndims_added):
                grad = _sum(grad, axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = _sum(grad, axis=i, keepdims=True)

            return grad
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: _Tensor) -> _Tensor:
            grad = _mul(grad, t1.data)
            ndims_added = len(grad.shape) - len(t2.data.shape)
            for _ in range(ndims_added):
                grad = _sum(grad, axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = _sum(grad, axis=i, keepdims=True)

            return grad
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)


def neg(t: Tensor) -> Tensor:
    data = _neg(t.data)
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda x: _neg(x))]
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
    data = _matmul(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return _matmul(grad, t2.data, transpose_b=True)
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return _matmul(t1.data, grad, transpose_a=True)
        depends_on.append(Dependency(t2, grad_fn2))
    return Tensor(data, requires_grad, depends_on)


def slice(t: Tensor, *idx) -> Tensor:
    """
    t2 = t1[3:4,4:4]
    """
    data = t.data[idx]
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(data)
            bigger_grad[idx] = grad
            return bigger_grad
        depends_on = Dependency(t, grad_fn)
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)
