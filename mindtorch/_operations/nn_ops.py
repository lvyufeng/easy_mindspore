from mindspore import ops
from mindspore.ops import Primitive
from mindspore.common.api import _pynative_executor as executor

_relu = Primitive('ReLU')
_relu.init_prim_io_names(inputs=['x'], outputs=['output'])
def raw_relu(x):
    return executor.real_run_op(_relu, 'ReLU', (x,))

_relu_grad = Primitive('ReluGrad')
_relu_grad.init_prim_io_names(inputs=['y_backprop', 'x'], outputs=['output'])
def raw_relu_grad(dout, out):
    return executor.real_run_op(_relu_grad, 'ReluGrad', (dout, out))

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
def raw_dropout(x, dropout):
    _dropout.add_prim_attr('keep_prob', 1 - dropout)
    _dropout.add_prim_attr('Seed0', 1)
    _dropout.add_prim_attr('Seed1', 1)
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
