import numpy as np
from mindspore import ops as _ops
from ..executor import execute
import easy_mindspore

# bernoulli
_bernoulli = _ops.Bernoulli()
def bernoulli(input):
    return execute(_bernoulli, input)

# multinomial
_multinomial = _ops.Multinomial()
def multinomial(input, num_samples, replacement=False):
    if replacement:
        return execute(_multinomial, input, num_samples)
    else:
        raise ValueError("not support replacement=False")

# normal
def normal(mean, std, size):
    return easy_mindspore.Tensor(np.random.normal(mean, std, size), easy_mindspore.float32)

# poisson


# rand


# rand_like


# randint
_ops.randint
def randint(low, high, size, *, dtype=None):
    return easy_mindspore.Tensor(np.random.randint(low, high, size), dtype)

# randint_like
def ranint_like(input, low, high, *, dtype=None):
    if dtype is None:
        dtype = input.dtype
    return randint(low, high, input.shape, dtype=dtype)

# randn
def randn(*size, dtype=None):
    return easy_mindspore.Tensor(np.random.randn(*size), dtype)

# randn_like


# randperm
