from mindspore._c_expression import typing

float32 = typing.Float(32)
float = float32
float64 = typing.Float(64)
double = float64
float16 = typing.Float(16)
# bfloat16: dtype = ...
half = float16
uint8 = typing.UInt(8)
int8 = typing.Int(8)
int16 = typing.Int(16)
short = int16
int32 = typing.Int(32)
int = int32
int64 = typing.Int(64)
long = int64
bool = typing.Bool()

def is_floating_point(x):
    return isinstance(x.dtype, typing.Float)