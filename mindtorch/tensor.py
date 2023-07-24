import numpy as np
import warnings
from typing import List, NamedTuple, Callable, Optional, Union
from mindspore._c_expression import TensorNode
from mindspore._c_expression import Tensor as Array  # pylint: disable=E0611

import mindtorch
from mindtorch import BACKEND
from mindtorch.config import using_config

from .utils import ASCEND_DTYPE_MAP

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[Array], Array]


Arrayable = Union[float, list, int, Array]

def ensure_array(arrayable: Arrayable, dtype) -> Array:
    if isinstance(arrayable, Tensor):
        return arrayable.data
    if isinstance(arrayable, Array):
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
                warnings.warn(f'Tensor dtype will auto change from numpy dtype {origin_dtype} to tensor dtype {dtype} on Ascend.')
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
                 dtype=None
                 ) -> None:
        self.data = ensure_array(data, dtype)
        self.requires_grad = requires_grad

        self.grad: Optional['Tensor'] = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data._shape

    @property
    def ndim(self):
        return self.data.dim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def grad_fn(self):
        return self.creator

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'Tensor(None)'
        p = str(self.data.asnumpy()).replace('\n', '\n' + ' ' * 9)
        return f'Tensor({p})'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self, gradient: 'Tensor' = None, retain_graph=None, create_graph=False) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if gradient is None:
            if self.shape == ():
                gradient = Tensor(1, dtype=self.dtype)
            else:
                raise RuntimeError("grad must specified for non-0-tensor")

        if self.grad is None:
            self.grad = gradient

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

                if not retain_graph:
                    for y in f.outputs:
                        y().grad = None  # y is weakref


    def tolist(self):
        return self.data.asnumpy().tolist()

    def numpy(self):
        return self.data.asnumpy()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return mindtorch._functions.reshape(self, shape)

    def sum(self, axis=None, keepdims=False):
        return mindtorch._functions.sum(self, axis, keepdims)

    def cuda(self):
        return self

    def __iadd__(self, other) -> 'Tensor':
        """
        when we do t += other
        """
        self.data = mindtorch._operations.raw_add(self.data, ensure_tensor(other).data)
        # Invalidate the gradient
        self.grad = None
        return self

    def __isub__(self, other) -> 'Tensor':
        self.data = mindtorch._operations.raw_sub(self.data, ensure_tensor(other).data)
        # Invalidate the gradient
        self.grad = None
        return self

    def __imul__(self, other) -> 'Tensor':
        self.data = mindtorch._operations.raw_mul(self.data, ensure_tensor(other).data)
        # Invalidate the gradient
        self.grad = None
        return self

def setup_tensor():
    from mindtorch._functions import add, mul, neg, sub, rsub, div, rdiv, pow, \
        matmul, get_item
    Tensor.__add__ = add
    Tensor.__radd__ = add
    Tensor.__mul__ = mul
    Tensor.__rmul__ = mul
    Tensor.__neg__ = neg
    Tensor.__sub__ = sub
    Tensor.__rsub__ = rsub
    Tensor.__truediv__ = div
    Tensor.__rtruediv__ = rdiv
    Tensor.__pow__ = pow
    Tensor.__matmul__ = matmul
    Tensor.__getitem__ = get_item
