from typing import overload
import math
import numpy as np
from mindspore import ops as _ops
from ..executor import execute
import easy_mindspore

# from_numpy
def from_numpy(ndarray):
    return easy_mindspore.Tensor(ndarray)

# frombuffer


# zeros
def zeros(*size, dtype=None):
    if isinstance(size[0], tuple):
        size = size[0]
    return easy_mindspore.Tensor(np.zeros(size), dtype)

# zeros_like
def zeros_like(input, *, dtype=None):
    return zeros(*input.shape, dtype=input.dtype if dtype is None else dtype)

# ones
_ones = _ops.Ones()
def ones(*size, dtype=None):
    # if dtype is None:
    #     dtype = easy_mindspore.float32
    # if isinstance(size[0], tuple):
    #     size = size[0]
    # return execute(_ones, size, dtype)
    if isinstance(size[0], tuple):
        size = size[0]
    return easy_mindspore.Tensor(np.ones(size), dtype)

# ones_like
def ones_like(input, *, dtype=None):
    return ones(*input.shape, dtype=input.dtype if dtype is None else dtype)

# arange
def arange(start=0, end=None, step=1, dtype=None):
    if dtype is None:
        dtype = easy_mindspore.float32
    if end is None:
        start, end = 0, start
    return easy_mindspore.Tensor(np.arange(start, end, step), dtype)

# range
def range(start=0, end=None, step=1, dtype=None):
    if dtype is None:
        dtype = easy_mindspore.float32
    if end is None:
        start, end = 0, start
    num_steps = int((end - start) / step) + 1
    result = linspace(start, end, num_steps, dtype=dtype)
    return result

# linspace
def linspace(start, end, steps, *, dtype=None):
    return easy_mindspore.Tensor(np.linspace(start, end, steps), dtype)

# logspace
def logspace(start, end, steps, base=10.0, *, dtype=None):
    return easy_mindspore.Tensor(np.logspace(start, end, steps, base=base), dtype)

# eye
def eye(n, m=None, *, dtype=None):
    return easy_mindspore.Tensor(np.eye(n, m), dtype)

# empty


# empty_like


# empty_strided


# full
def full(size, fill_value, *, dtype=None):
    return easy_mindspore.Tensor(np.full(size, fill_value), dtype)

# full_like
def full_like(input, fill_value, *, dtype=None):
    return full(input.shape, fill_value, dtype=dtype)

# quantize_per_tensor


# quantize_per_channel


# dequantize


# complex
_complex = _ops.Complex()
def complex(real, imag):
    return execute(_complex, real, imag)

# polar
_polar = _ops.Polar()
def polar(abs, angle):
    return execute(_polar, abs, angle)

# heaviside
_heaviside = _ops.Heaviside()
def heaviside(input, values):
    return execute(_heaviside, input, values)
