import numpy as np
from typing import List, NamedTuple, Callable, Optional, Union
from mindspore._c_expression import Tensor as Array  # pylint: disable=E0611
from mindspore.common.api import _pynative_executor as executor
import mindtorch
from mindtorch.config import using_config
from mindtorch import dtype
from .utils import NORMAL_DTYPE_MAP, _get_unfold_indices

def _uniform(self, a, b):
    dtype = self.dtype
    data = np.random.uniform(a, b, self._shape)
    self.assign_value_cpp(Array.from_numpy(data))
    self.set_dtype(dtype)

def _fill(self, value):
    dtype = self.dtype
    data = np.full(self._shape, value)
    self.assign_value_cpp(Array.from_numpy(data))
    self.set_dtype(dtype)

def _zero(self):
    dtype = self.dtype
    data = np.zeros(self._shape)
    self.assign_value_cpp(Array.from_numpy(data))
    self.set_dtype(dtype)
    

Array.uniform_ = _uniform
Array.fill_ = _fill
Array.zero_ = _zero

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[Array], Array]


Arrayable = Union[float, list, int, Array, np.ndarray]

def ensure_array(arrayable: Arrayable, dtype) -> Array:
    if isinstance(arrayable, Tensor):
        return arrayable.data
    if isinstance(arrayable, Array):
        return arrayable
    if dtype is None:
        if isinstance(arrayable, (list, tuple)):
            arrayable = np.array(arrayable)

        if isinstance(arrayable, float):
            origin_dtype = type(arrayable)
            dtype = NORMAL_DTYPE_MAP[origin_dtype]
        elif isinstance(arrayable, np.ndarray):
            origin_dtype = str(arrayable.dtype)
            dtype = NORMAL_DTYPE_MAP.get(origin_dtype, None)

        return Array(arrayable, dtype)
    return Array(arrayable, dtype)

Tensorable = Union['Tensor', float, np.ndarray]

def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return tensor(tensorable)

def ensure_tuple_int(args):
    for i in args:
        if not isinstance(i, int):
            return False
    return True


class Tensor:
    """mindtorch defined Tensor."""
    def __init__(self,
                 *args,
                 **kwargs
                 ) -> None:
        dtype = kwargs.get('dtype', mindtorch.float32)
        # object data
        if isinstance(args[0], (list, Tensor, Array, np.ndarray)):
            if len(args) == 1:
                self.data = ensure_array(args[0], dtype)
            else:
                raise ValueError(f'only support one arrayable data as input of Tensor, but got {len(args)}')
        # tuple of ints size
        elif isinstance(args[0], tuple):
            if ensure_tuple_int(args[0]):
                self.data = Array(shape=args[0], dtype=dtype)
            else:
                raise ValueError(f'only support tuple of ints for Tensor size.')
        elif isinstance(args[0], int):
            if ensure_tuple_int(args):
                self.data = Array(shape=args, dtype=dtype)
            else:
                raise ValueError(f'only support tuple of ints for Tensor size.')
        else:
            raise ValueError(f'only support `tuple of ints size` and `object data`.')

        # Tensor
        self.requires_grad = kwargs.get('requires_grad', False)

        self.grad: Optional['Tensor'] = None
        self.creator = None
        self.generation = 0

    @property
    def T(self):
        return mindtorch._functions.transpose(self, 0, 1)

    @property
    def shape(self):
        return self.data._shape

    @property
    def ndim(self):
        return self.data.dim()

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def dim(self):
        return self.ndim

    def numel(self):
        return self.data._size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def grad_fn(self):
        return self.creator

    def __len__(self):
        if self.shape == ():
            return 1
        return self.shape[0]

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
                gradient = tensor(1, dtype=self.dtype)
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
            gys = [output().grad for output in f.ctx.outputs]  # output is weakref
            with using_config('enable_backprop', create_graph):
                gxs = f.last_fn._backward(f.ctx, *gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.ctx.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)
                if not retain_graph:
                    for y in f.ctx.outputs:
                        y().grad = None  # y is weakref

    def tolist(self):
        return self.data.asnumpy().tolist()

    def numpy(self):
        return self.data.asnumpy()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return mindtorch._functions.reshape(self, shape)

    def view_as(self, other):
        return self.reshape(other.shape)

    def sum(self, dim=None, keepdims=False):
        return mindtorch._functions.sum(self, dim, keepdims)

    def mean(self, dim=None, keepdims=False):
        return mindtorch._functions.mean(self, dim, keepdims)

    def flatten(self, start_dim=0, end_dim=-1):
        return mindtorch._functions.flatten(self, start_dim, end_dim)

    def unflatten(self, dim, sizes):
        return mindtorch._functions.unflatten(self, dim, sizes)

    def argmax(self, dim=None, keepdim=False):
        out = mindtorch._functions.argmax(self, dim)
        if keepdim:
            return out.unsqueeze(dim)
        return out

    def unsqueeze(self, dim):
        return mindtorch._functions.expand_dims(self, dim)

    def squeeze(self, dim):
        return mindtorch._functions.squeeze(self, dim)

    def unfold(self, dimension, size, step):
        _indices, _dimension = _get_unfold_indices(self.shape, dimension, size, step)
        output = mindtorch._functions.gather(self, _indices, axis=_dimension)
        return mindtorch._functions.transpose(output, _dimension + 1, -1)

    def permute(self, *dims):
        return mindtorch._functions.permute(self, dims)

    def transpose(self, dim0, dim1):
        return mindtorch._functions.transpose(self, dim0, dim1)


    def view(self, *size):
        return self.reshape(*size)

    def contiguous(self):
        return self

    def cuda(self):
        return self

    def cpu(self):
        return self

    def detach(self):
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

    def __itruediv__(self, other) -> 'Tensor':
        self.data = mindtorch._operations.raw_div(self.float().data, ensure_tensor(other).data)
        # Invalidate the gradient
        self.grad = None
        return self

    def __format__(self, format_spec):
        return np.ndarray.__format__(self.numpy(), format_spec)

    def mul_(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        self.data = mindtorch._operations.raw_mul(self.data, other_data)
        return self

    def addcmul_(self, tensor1, tensor2, value=1):
        self.data = mindtorch._operations.raw_addcmul(self.data, tensor1.data, tensor2.data, Array(value))
        return self

    def addcdiv_(self, tensor1, tensor2, value=1):
        self.data = mindtorch._operations.raw_addcdiv(self.data, tensor1.data, tensor2.data, Array(value))
        return self

    def sqrt_(self):
        self.data = mindtorch._operations.raw_sqrt(self.data)
        return self

    def div_(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        self.data = mindtorch._operations.raw_div(self.data, other_data)
        return self

    def add_(self, other, alpha=1):
        other_data = other.data if isinstance(other, Tensor) else other
        if alpha == 1:
            self.data = mindtorch._operations.raw_add(self.data, other_data)
        else:
            other_data = mindtorch._operations.raw_mul(other_data, alpha)
            self.data = mindtorch._operations.raw_add(self.data, other_data)
        return self

    def uniform_(self, a, b):
        self.data.uniform_(a, b)
        return self

    def fill_(self, value):
        self.data.fill_(value)
        return self

    def zero_(self):
        self.data.zero_()
        return self

    def masked_fill_(self, mask, value):
        self.data = mindtorch._operations.raw_masked_fill(self.data, mask, value)
        return self

    def float(self):
        return mindtorch._functions.cast(self, dtype.float)

    def double(self):
        return mindtorch._functions.cast(self, dtype.double)
    
    def int(self):
        return mindtorch._functions.cast(self, dtype.int)

    def long(self):
        return mindtorch._functions.cast(self, dtype.long)

    def bool(self):
        return mindtorch._functions.cast(self, dtype.bool)

    def to(self, target):
        if isinstance(target, dtype.typing.Type):
            return mindtorch._functions.cast(self, target)
        return self

    def item(self):
        return self.data.asnumpy().item()

def setup_tensor():
    from mindtorch._functions import add, mul, neg, sub, rsub, div, rdiv, pow, \
        matmul, get_item, equal, less, le, greater, ge, sqrt
    Tensor.add = add
    Tensor.eq = equal
    Tensor.sqrt = sqrt
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
    Tensor.__eq__ = equal
    Tensor.__lt__ = less
    Tensor.__le__ = le
    Tensor.__gt__ = greater
    Tensor.__ge__ = ge

def tensor(data, dtype=None, requires_grad=False):
    return Tensor(ensure_array(data, dtype), dtype=dtype, requires_grad=requires_grad)

def is_tensor(x):
    return isinstance(x, Tensor)
