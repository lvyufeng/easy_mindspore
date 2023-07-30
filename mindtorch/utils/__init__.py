from mindspore._c_expression import typing
import mindtorch.utils.data as data

NORMAL_DTYPE_MAP = {
    float: typing.Float(32),
    "float64": typing.Float(32),
}

