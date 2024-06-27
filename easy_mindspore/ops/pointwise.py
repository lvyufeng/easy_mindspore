import math
from mindspore import ops as _ops
from mindspore.ops import Primitive
from mindspore._c_expression import typing
from ..executor import execute
from .. import dtype

import easy_mindspore

# abs
_abs = _ops.Abs()
def abs(input):
    return execute(_abs, input)

# absolute
def absolute(input):
    return abs(input)

# acos
_acos = _ops.ACos()
def acos(input):
    return execute(_acos, input)

# arccos
def arrcos(input):
    return acos(input)

# acosh
_acosh = _ops.Acosh()
def acosh(input):
    return execute(_acosh, input)

# arccosh
def arccosh(input):
    return acosh(input)

# add
_add = _ops.Add()
def add(input, other, *, alpha=1):
    if alpha == 1:
        return execute(_add, input, other)
    return execute(_add, input, alpha * other)

# addcdiv
_addcdiv = _ops.Addcdiv()
def addcdiv(input, tensor1, tensor2, *, value=1):
    return execute(_addcdiv, input, tensor1, tensor2, value)

# addcmul
_addcmul = _ops.Addcmul()
def addcmul(input, tensor1, tensor2, *, value=1):
    return execute(_addcmul, input, tensor1, tensor2, value)

# angle
_angle = _ops.Angle()
def angle(input):
    return execute(_angle, input)

# asin
_asin = _ops.Asin()
def asin(input):
    return execute(_asin, input)

# arcsin
def arcsin(input):
    return asin(input)

# asinh
_asinh = _ops.Asinh()
def asinh(input):
    return execute(_asinh, input)

# arcsinh
def arcsinh(input):
    return asinh(input)

# atan
_atan = _ops.Atan()
def atan(input):
    return execute(_atan, input)

# arctan
def arctan(input):
    return atan(input)

# atanh
_atanh = _ops.Atanh()
def atanh(input):
    return execute(_atanh, input)

# arctanh
def arctanh(input):
    return atanh(input)

# atan2
_atan2 = _ops.Atan2()
def atan2(input):
    return execute(_atan2, input)

# arctan2
def arctan2(input):
    return atan2(input)

# bitwise_not

# bitwise_and
_bitwise_and = _ops.BitwiseAnd()
def bitwise_and(input, other):
    return execute(_bitwise_and, input, other)

# bitwise_or
_bitwise_or = _ops.BitwiseOr()
def bitwise_or(input, other):
    return execute(_bitwise_or, input, other)

# bitwise_xor
_bitwise_xor = _ops.BitwiseXor()
def bitwise_xor(input, other):
    return execute(_bitwise_xor, input, other)

# bitwise_left_shift
_bitwise_left_shift = _ops.LeftShift()
def bitwise_left_shift(input, other):
    return execute(_bitwise_left_shift, input, other)

# bitwise_right_shift
_bitwise_right_shift = _ops.RightShift()
def bitwise_right_shift(input, other):
    return execute(_bitwise_right_shift, input, other)

# ceil
_ceil = _ops.Ceil()
def ceil(input):
    return execute(_ceil, input)

# clamp


# clip


# conj_physical


# copysign


# cos
_cos = _ops.Cos()
def cos(input):
    return execute(_cos, input)

# cosh
_cosh = _ops.Cosh()
def cosh(input):
    return execute(_cosh, input)

# deg2rad
def deg2rad(input):
    return input * easy_mindspore.Tensor(math.pi / 180.0, input.dtype)

# div
_div = _ops.Div()
def div(input, other, *, rounding_mode=None):
    if rounding_mode is not None and rounding_mode not in ['floor', 'trunc']:
        raise ValueError("For _ops.div, rounding_mode value should be None, 'floor' or 'trunc'.")

    if rounding_mode == 'floor':
        return floor_divide(input, other)
    output = execute(_div, input, other)
    if rounding_mode == 'trunc':
        output = trunc(output)
    return output


# divide
def divide(input, other):
    return div(input, other)

# digamma
_digamma = _ops.Digamma()
def digamma(input):
    return execute(_digamma, input)

# erf
_erf = _ops.Erf()
def erf(input):
    return execute(_erf, input)

# erfc
_erfc = _ops.Erfc()
def erfc(input):
    return execute(_erfc, input)

# erfinv
_erfinv = _ops.Erfinv()
def erfinv(input):
    return execute(_erfinv, input)

# exp
_exp = _ops.Exp()
def exp(input):
    return execute(_exp, input)

# exp2
def exp2(input):
    return pow(2, input)

# expm1
_expm1 = _ops.Expm1()
def expm(input):
    return execute(_expm1, input)

# fake_quantize_per_channel_affine


# fake_quantize_per_tensor_affine


# fix


# float_power
def float_power(input, exponent):
    return pow(input.to(dtype.float64), exponent.to(dtype.float64))

# floor
_floor = _ops.Floor()
def floor(input):
    return execute(_floor, input)

# floor_divide
_floor_divide = _ops.FloorDiv()
def floor_divide(input, other):
    return execute(_floor_divide, input, other)

# fmod
_fmod = _ops.Mod()
def fmod(input, other):
    return execute(_fmod, input, other)

# frac
def frac(input):
    return fmod(input, 1)

# frexp


# imag
_imag = _ops.Imag()
def imag(input):
    return execute(_imag, input)

# ldexp


# lerp
_lerp = _ops.Lerp()
def lerp(input, end, weight):
    return execute(_lerp, input, end, weight)

# lgamma
_lgamma = _ops.Lgamma()
def lgamma(input):
    return execute(_lgamma, input)

# log
_log = _ops.Log()
def log(input):
    return execute(_log, input)

# log10

# log1p
_log1p = _ops.Log1p()
def log1p(input):
    return execute(log1p, input)

# log2


# logaddexp


# logaddexp2


# logical_and
_logical_and = _ops.LogicalAnd()
def logical_and(input, other):
    return execute(_logical_and, input, other)

# logical_not
_logical_not = _ops.LogicalNot()
def logical_not(input, other):
    return execute(_logical_not, input, other)

# logical_or
_logical_or = _ops.LogicalOr()
def logical_or(input, other):
    return execute(_logical_or, input, other)

# logical_xor
_logical_xor = _ops.LogicalXor()
def logical_xor(input, other):
    return execute(_logical_xor, input, other)

# logit
_logit = Primitive('Logit')
_logit.init_prim_io_names(inputs=['x'], outputs=['y'])
def logit(input, eps=None):
    _logit.add_prim_attr("eps", eps)
    if eps is None:
        eps = -1
    return execute(_logit, input)

# hypot
_hypot = _ops.Hypot()
def hypot(input, other):
    return execute(_hypot, input, other)

# i0

# igamma
_igamma = _ops.Igamma()
def igamma(input, other):
    return execute(_igamma, input, other)

# igammac
_igammac = _ops.Igammac()
def igammac(input, other):
    return execute(_igammac, input, other)

# mul
_mul = _ops.Mul()
def mul(input, other):
    return execute(_mul, input, other)

# multiply
def multiply(input, other):
    return mul(input, other)

# mvlgamma
_mvlgamma = Primitive('Mvlgamma')
_mvlgamma.init_prim_io_names(inputs=['x'], outputs=['y'])
def mvlgamma(input, p):
    _mvlgamma.add_prim_attr('p', p)
    return execute(_mvlgamma, input)
    
# nan_to_num
_nan_to_num = Primitive('NanToNum')
def nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    _nan_to_num.add_prim_attr('nan', nan)
    _nan_to_num.add_prim_attr('posinf', posinf)
    _nan_to_num.add_prim_attr('neginf', neginf)
    if nan is None:
        _nan_to_num.add_prim_attr("nan_none", True)
    if posinf is None:
        _nan_to_num.add_prim_attr("posinf_none", True)
    if neginf is None:
        _nan_to_num.add_prim_attr("neginf_none", True)
    return execute(_nan_to_num, input)

# neg
_neg = Primitive('Neg')
def neg(input):
    return execute(_neg, input)

# negative
def negative(input):
    return neg(input)

# nextafter
_nextafter = _ops.NextAfter()
def nextafter(input, other):
    return execute(_nextafter, input, other)

# polygamma
_polygamma = _ops.Polygamma()
def polygamma(n, input):
    return execute(_polygamma, n, input)

# positive
def positive(input):
    return input

# pow
_pow = _ops.Pow()
def pow(input, exponent):
    return execute(_pow, input, exponent)

# quantized_batch_norm


# quantized_max_pool1d


# quantized_max_pool2d


# rad2deg
def rad2deg(input):
    return input * easy_mindspore.Tensor(180.0 / math.pi, input.dtype)

# real
_real = _ops.Real()
def real(input):
    return execute(_real, input)

# reciprocal
_reciprocal = _ops.Reciprocal()
def reciprocal(input):
    return execute(_reciprocal, input)

# remainder


# round
_round = _ops.Round()
def round(input):
    return execute(_round, input)

# rsqrt
_rsqrt = _ops.Rsqrt()
def rsqrt(input):
    return execute(_rsqrt, input)

# sigmoid
_sigmoid = _ops.Sigmoid()
def sigmoid(input):
    return execute(_sigmoid, input)

# sign
_sign = _ops.Sign()
def sign(input):
    return execute(_sign, input)

# sgn

# signbit

# sin
_sin = _ops.Sin()
def sin(input):
    return execute(_sin, input)

# sinc
_sinc = _ops.Sinc()
def sinc(input):
    return execute(_sinc, input)

# sinh
_sinh = _ops.Sinh()
def sinh(input):
    return execute(_sinh, input)

# softmax
_softmax = Primitive('Softmax')
_softmax.init_prim_io_names(inputs=['x'], outputs=['output'])
def softmax(input, dim=-1):
    _softmax.add_prim_attr('axis', (dim,))
    return execute(_softmax, input)

# sqrt
_sqrt = _ops.Sqrt()
def sqrt(input):
    return execute(_sqrt, input)

# square
_square = _ops.Square()
def square(input):
    return execute(_square, input)

# sub
_sub = _ops.Sub()
def sub(input, other):
    return execute(_sub, input, other)

# subtract
def subtract(input, other):
    return sub(input, other)

# tan
_tan = _ops.Tan()
def tan(input):
    return execute(_tan, input)

# tanh
_tanh = _ops.Tanh()
def tanh(input):
    return execute(_tanh, input)

# true_divide
def true_divide(input, other):
    return div(input, other)

# trunc
_trunc = _ops.Trunc()
def trunc(input):
    return execute(_trunc, input)

# xlogy
_xlogy = _ops.Xlogy()
def xlogy(input, other):
    return execute(_xlogy, input, other)