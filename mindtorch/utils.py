from mindspore._c_expression import typing

ASCEND_DTYPE_MAP = {
    float: typing.Float(32),
    int: typing.Int(32),
    "float64": typing.Float(32),
    "int64": typing.Int(32),    
}
