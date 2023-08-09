import mindtorch
from functools import lru_cache
from mindspore._c_expression import typing
import mindtorch.utils.data as data

NORMAL_DTYPE_MAP = {
    float: typing.Float(32),
    "float64": typing.Float(32),
}

def _get_unfold_indices(input_shape, dimension, size, step):
    if dimension < 0:
        dimension += len(input_shape)
    indices = []
    for i in range(0, input_shape[dimension] - size + 1, step):
        indices.append(list(range(i, i + size)))

    indices = mindtorch.tensor(indices)
    return indices, dimension