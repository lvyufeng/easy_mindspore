from mindspore._c_expression import typing

NORMAL_DTYPE_MAP = {
    float: typing.Float(32),
    "float64": typing.Float(32),
}

ASCEND_DTYPE_MAP = {
    int: typing.Int(32),
    "int64": typing.Int(32),
}
