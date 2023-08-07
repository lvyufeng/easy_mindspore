import math
from mindspore import ops
from mindspore.ops import Primitive
from mindspore.common.api import _pynative_executor as executor
from mindspore.ops._tracefunc import PackFunc

_relu = Primitive('ReLU')
_relu.init_prim_io_names(inputs=['x'], outputs=['output'])
def raw_relu(x):
    return executor.real_run_op(_relu, 'ReLU', (x,))

_relu_grad = Primitive('ReluGrad')
_relu_grad.init_prim_io_names(inputs=['y_backprop', 'x'], outputs=['output'])
def raw_relu_grad(dout, out):
    return executor.real_run_op(_relu_grad, 'ReluGrad', (dout, out))

_gelu = Primitive('GeLU')
_gelu.init_prim_io_names(inputs=['x'], outputs=['output'])
def raw_gelu(x):
    return executor.real_run_op(_gelu, 'GeLU', (x,))


_gelu_grad = Primitive('GeLUGrad')
_gelu_grad.init_prim_io_names(inputs=['dy', 'x', 'y'], outputs=['z'])
def raw_gelu_grad(dout, x, out):
    return executor.real_run_op(_gelu_grad, 'GeLUGrad', (dout, x, out))

_softmax_crossentropy = Primitive('SparseSoftmaxCrossEntropyWithLogits')
_softmax_crossentropy.init_prim_io_names(inputs=['features', 'labels'], outputs=['output'])
_softmax_crossentropy.add_prim_attr('sens', 1.0)
def raw_softmax_crossentropy(logits, labels, grad=False):
    _softmax_crossentropy.add_prim_attr('is_grad', grad)
    return executor.real_run_op(_softmax_crossentropy, 'SparseSoftmaxCrossEntropyWithLogits', (logits, labels))

_softmax_crossentropy_ascend = Primitive('SparseSoftmaxCrossEntropyWithLogitsV2')
_softmax_crossentropy_ascend.init_prim_io_names(inputs=['features', 'labels'], outputs=['loss', 'backprop'])
def raw_softmax_crossentropy_ascend(logits, labels):
    return executor.real_run_op(_softmax_crossentropy_ascend, 'SparseSoftmaxCrossEntropyWithLogitsV2', (logits, labels))

_conv2d = Primitive('Conv2D')
_conv2d.init_prim_io_names(inputs=['x', 'w'], outputs=['output'])
_conv2d.add_prim_attr('mode', 1) # only support mode=1
def raw_conv2d(x, w, out_channel, kernel_size, pad_mode="valid", pad=0, stride=1, dilation=1, groups=1, data_format="NCHW"):
    _conv2d.add_prim_attr("out_channel", out_channel)
    _conv2d.add_prim_attr("kernel_size", kernel_size)
    _conv2d.add_prim_attr("pad_mode", pad_mode)
    _conv2d.add_prim_attr("pad", pad)
    _conv2d.add_prim_attr('stride', stride)
    _conv2d.add_prim_attr('dilation', dilation)
    _conv2d.add_prim_attr('group', groups)
    _conv2d.add_prim_attr('groups', groups)
    _conv2d.add_prim_attr('data_format', data_format)
    return executor.real_run_op(_conv2d, 'Conv2D', (x, w))

_conv2d_gx = Primitive('Conv2DBackpropInput')
_conv2d_gx.init_prim_io_names(inputs=['out_backprop', 'filter', 'input_sizes'], outputs=['output'])
_conv2d_gx.add_prim_attr('mode', 1) # only support mode=1
def raw_conv2d_gx(gy, w, x_shape, out_channel, kernel_size, pad_mode="valid", pad=0, pad_list=None,
               stride=1, dilation=1, groups=1, data_format="NCHW"):
    _conv2d_gx.add_prim_attr("out_channel", out_channel)
    _conv2d_gx.add_prim_attr("kernel_size", kernel_size)
    _conv2d_gx.add_prim_attr("pad_mode", pad_mode)
    _conv2d_gx.add_prim_attr("pad", pad)
    _conv2d_gx.add_prim_attr("pad_list", pad_list)
    _conv2d_gx.add_prim_attr('stride', stride)
    _conv2d_gx.add_prim_attr('dilation', dilation)
    _conv2d_gx.add_prim_attr('group', groups)
    _conv2d_gx.add_prim_attr('groups', groups)
    _conv2d_gx.add_prim_attr('data_format', data_format)
    return executor.real_run_op(_conv2d_gx, 'Conv2DBackpropInput', (gy, w, x_shape))

_conv2d_gw = Primitive('Conv2DBackpropFilter')
_conv2d_gw.init_prim_io_names(inputs=['out_backprop', 'input', 'filter_sizes'], outputs=['output'])
_conv2d_gw.add_prim_attr('mode', 1) # only support mode=1
def raw_conv2d_gw(gy, x, w_shape, out_channel, kernel_size, pad_mode="valid", pad=0, pad_list=None,
               stride=1, dilation=1, groups=1, data_format="NCHW"):
    _conv2d_gw.add_prim_attr("out_channel", out_channel)
    _conv2d_gw.add_prim_attr("kernel_size", kernel_size)
    _conv2d_gw.add_prim_attr("pad_mode", pad_mode)
    _conv2d_gw.add_prim_attr("pad", pad)
    _conv2d_gw.add_prim_attr("pad_list", pad_list)
    _conv2d_gw.add_prim_attr('stride', stride)
    _conv2d_gw.add_prim_attr('dilation', dilation)
    _conv2d_gw.add_prim_attr('group', groups)
    _conv2d_gw.add_prim_attr('groups', groups)
    _conv2d_gw.add_prim_attr('data_format', data_format)
    return executor.real_run_op(_conv2d_gw, 'Conv2DBackpropFilter', (gy, x, w_shape))

_bias_add = Primitive('BiasAdd')
_bias_add.init_prim_io_names(inputs=['x', 'b'], outputs=['output'])
_bias_add.add_prim_attr('data_format', 'NCHW')
def raw_bias_add(x, y):
    return executor.real_run_op(_bias_add, 'BiasAdd', [x, y])

_bias_add_grad = Primitive('BiasAddGrad')
_bias_add_grad.init_prim_io_names(inputs=['dout'], outputs=['output'])
_bias_add_grad.add_prim_attr('data_format', "NCHW")
def raw_bias_add_grad(gy):
    return executor.real_run_op(_bias_add_grad, 'BiasAddGrad', [gy])

_dropout = Primitive('Dropout')
_dropout.add_prim_attr('Seed0', 1)
_dropout.add_prim_attr('Seed1', 1)
def raw_dropout(x, dropout):
    _dropout.add_prim_attr('keep_prob', 1 - dropout)
    return executor.real_run_op(_dropout, 'Dropout', [x])

_dropout_grad = Primitive('DropoutGrad')
def raw_dropout_grad(x, mask, dropout):
    _dropout_grad.add_prim_attr('keep_prob', 1 - dropout)
    return executor.real_run_op(_dropout_grad, 'DropoutGrad', [x, mask])

_maxpool = Primitive('MaxPoolWithArgmaxV2')
_maxpool.init_prim_io_names(inputs=["x"], outputs=["output", "argmax"])
_maxpool.add_prim_attr("argmax_type", 4)
def raw_maxpool(x, kernel_size, strides=None, pads=0, dilation=(1, 1), ceil_mode=False):
    _maxpool.add_prim_attr("kernel_size", kernel_size)
    _maxpool.add_prim_attr("strides", strides)
    _maxpool.add_prim_attr("pads", pads)
    _maxpool.add_prim_attr("dilation", dilation)
    _maxpool.add_prim_attr("ceil_mode", ceil_mode)
    return executor.real_run_op(_maxpool, 'MaxPoolWithArgmaxV2', [x])

_maxpool_grad = Primitive('MaxPoolGradWithArgmaxV2')
_maxpool_grad.init_prim_io_names(inputs=['x', 'grad', 'argmax'], outputs=['y'])
_maxpool_grad.add_prim_attr("argmax_type", 4)
def raw_maxpool_grad(x, grad, argmax, kernel_size, strides=None, pads=0, dilation=(1, 1), ceil_mode=False):
    _maxpool_grad.add_prim_attr("kernel_size", kernel_size)
    _maxpool_grad.add_prim_attr("strides", strides)
    _maxpool_grad.add_prim_attr("pads", pads)
    _maxpool_grad.add_prim_attr("dilation", dilation)
    _maxpool_grad.add_prim_attr("ceil_mode", ceil_mode)

    return executor.real_run_op(_maxpool_grad, 'MaxPoolGradWithArgmaxV2', [x, grad, argmax])


_nll_loss = Primitive('NLLLoss')
_nll_loss.init_prim_io_names(inputs=['x', 'target', "weight"], outputs=['loss', 'total_weight'])
def raw_nll_loss(input, target, weight, ignore_index=-100, reduction='mean'):
    _nll_loss.add_prim_attr("ignore_index", ignore_index)
    _nll_loss.add_prim_attr("reduction", reduction)
    return executor.real_run_op(_nll_loss, 'NLLLoss', [input, target, weight])

_nll_loss_grad = Primitive('NLLLossGrad')
_nll_loss_grad.init_prim_io_names(inputs=['x', 'loss_grad', 'target', 'weight', 'total_weight'], outputs=['x_grad'])
def raw_nll_loss_grad(x, loss_grad, target, weight, total_weight, ignore_index=-100, reduction='mean'):
    _nll_loss_grad.add_prim_attr("ignore_index", ignore_index)
    _nll_loss_grad.add_prim_attr("reduction", reduction)
    return executor.real_run_op(_nll_loss_grad, 'NLLLossGrad', [x, loss_grad, target, weight, total_weight])


_layer_norm = Primitive("LayerNorm")
def raw_layer_norm(input, weight, bias, begin_norm_axis=1, begin_params_axis=1, epsilon=1e-7):
    _layer_norm.add_prim_attr("begin_norm_axis", begin_norm_axis)
    _layer_norm.add_prim_attr("begin_params_axis", begin_params_axis)
    _layer_norm.add_prim_attr("epsilon", epsilon)

    return executor.real_run_op(_layer_norm, 'LayerNorm', [input, weight, bias])


_layer_norm_grad = Primitive("LayerNormGrad")
def raw_layer_norm_grad(input, gy, mean, var, weight, begin_norm_axis=1, begin_params_axis=1):
    # x, dout[0], out[2], out[1], gamma
    _layer_norm_grad.add_prim_attr("begin_norm_axis", begin_norm_axis)
    _layer_norm_grad.add_prim_attr("begin_params_axis", begin_params_axis)
    return executor.real_run_op(_layer_norm_grad, 'LayerNormGrad', [input, gy, var, mean, weight])

_unfold = Primitive('Im2Col')
_unfold.init_prim_io_names(inputs=['x'], outputs=['y'])
def raw_unfold(x, ksizes, strides=1, dilations=1, pads=0):
    _unfold.add_prim_attr('ksizes', ksizes)
    _unfold.add_prim_attr('strides', strides)
    _unfold.add_prim_attr('dilations', dilations)
    _unfold.add_prim_attr('pads', pads)
    _unfold.add_prim_attr('padding_mode', "CALCULATED")
    return executor.real_run_op(_unfold, 'Im2Col', [x])


_fold = Primitive('Col2Im')
_fold.init_prim_io_names(inputs=['x', 'output_size'], outputs=['y'])
def raw_fold(x, output_size, kernel_size, dilation=1, padding=0, stride=1):
    _fold.add_prim_attr('kernel_size', kernel_size)
    _fold.add_prim_attr('dilation', dilation)
    _fold.add_prim_attr('padding', padding)
    _fold.add_prim_attr('stride', stride)
    return executor.real_run_op(_fold, 'Col2Im', [x, output_size])

_softmax = Primitive('Softmax')
_softmax.init_prim_io_names(inputs=['x'], outputs=['output'])
def raw_softmax(input, axis):
    if not isinstance(axis, tuple):
        axis = (axis,)
    _softmax.add_prim_attr('axis', axis)
    return executor.real_run_op(_softmax, 'Softmax', [input])

_fused_linear_matmul = ops.MatMul(transpose_b=True)
_fused_linear_add = ops.BiasAdd()
def _pack_linear(x, y, b):
    x_shape = x.shape
    if x.ndim != 2:
        x = x.reshape(-1, x_shape[-1])
    out = _fused_linear_matmul(x, y)
    out = _fused_linear_add(out, b)
    out = out.reshape(*x_shape[:-1], out.shape[-1])
    return out

_fused_linear = PackFunc(_pack_linear, str(id(_pack_linear)), None, True)
def fused_linear(x, w, b):
    return executor.real_run_op(_fused_linear, 'PackFuc', [x, w, b])

_grad_matmul = ops.MatMul()
_grad_matmul_1 = ops.MatMul(transpose_a=True)
def _pack_linear_grad(x, w, b, gy):
    ndim = len(b.shape)
    lead = gy.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(b.shape) if sx == 1])
    gb = gy.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        gb = gb.squeeze(lead_axis)

    x_shape = x.shape
    if gy.ndim != 2:
        gy = gy.reshape(-1, gy.shape[-1])
    if x.ndim != 2:
        x = x.reshape(-1, x_shape[-1])
    gx = _grad_matmul(gy, w)
    gx = gx.reshape(*x_shape[:-1], gx.shape[-1])
    gW = _grad_matmul_1(x, gy)
    return gx, gW.T, gb


_fused_linear_grad = PackFunc(_pack_linear_grad, str(id(_pack_linear_grad)), None, True)
def fused_linear_grad(x, w, b, gy):
    return executor.real_run_op(_fused_linear_grad, 'PackFuc', [x, w, b, gy])

def _pack_gelu_erf(x):
    return 0.5 * x * (1 + ops.erf(x / ops.sqrt(ops.cast(2, x.dtype))))

_fused_gelu_erf = PackFunc(_pack_gelu_erf, str(id(_pack_gelu_erf)), None, True)
def fused_gelu_erf(x):
    return executor.real_run_op(_fused_gelu_erf, 'PackFuc', [x])

def _pack_gelu_erf_grad(x, dy):
    gelu_derivative =  0.5 * (1 + ops.erf(x / ops.sqrt(ops.cast(2, x.dtype)))) + \
        0.5 * x * ops.exp(-x ** 2 / 2) / ops.sqrt(ops.cast(2 * math.pi, x.dtype))
    return dy * gelu_derivative

_fused_gelu_erf_grad = PackFunc(_pack_gelu_erf_grad, str(id(_pack_gelu_erf_grad)), None, True)
def fused_gelu_erf_grad(x, dy):
    return executor.real_run_op(_fused_gelu_erf_grad, 'PackFuc', [x, dy])

def _pack_softmax_grad(y, gy, axis):
    gx = y * gy
    sumdx = gx.sum(axis=axis, keepdims=True)
    gx -= y * sumdx
    return gx

_fused_softmax_grad = PackFunc(_pack_softmax_grad, str(id(_pack_softmax_grad)), None, True)
def fused_softmax_grad(y, gy, axis):
    return executor.real_run_op(_fused_softmax_grad, 'PackFuc', [y, gy, axis])

def _pack_dropout(x, p):
    mask = ops.randn(x.shape) > p
    scale = 1 - p
    y = x * mask / scale
    return y, mask

_fused_dropout = PackFunc(_pack_dropout, str(id(_pack_dropout)), None, True)
def fused_dropout(x, p):
    return executor.real_run_op(_fused_dropout, 'PackFuc', [x, p])

def _pack_dropout_grad(gy, mask, p):
    dx = gy * mask / (1 - p)
    return dx
_fused_dropout_grad = PackFunc(_pack_dropout_grad, str(id(_pack_dropout_grad)), None, True)
def fused_dropout_grad(gy, mask, p):
    return executor.real_run_op(_fused_dropout_grad, 'PackFuc', [gy, mask, p])

