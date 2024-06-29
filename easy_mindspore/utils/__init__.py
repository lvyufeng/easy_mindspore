from mindspore._c_expression import typing
import easy_mindspore.utils.data as data

NORMAL_DTYPE_MAP = {
    float: typing.Float(32),
    "float64": typing.Float(32),
}

