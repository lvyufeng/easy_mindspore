import math
from mindspore import ops, Tensor
from mindspore.ops import Primitive
from mindspore.ops.operations._grad_ops import LayerNormGrad, Conv2DBackpropFilter, BiasAddGrad, \
    DropoutGrad, NLLLossGrad, MaxPoolGradWithArgmaxV2

from mindtorch import tensor
from mindtorch.autograd import Function, Context

from .math import matmul
from .creation import zeros_like
from .utils import sum_to

# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================
class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, x):
        y = ops.relu(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        y, = ctx.saved_tensors
        _relu_grad = Primitive('ReluGrad')
        gx = _relu_grad(gy.data, y)
        return tensor(gx)

class GELU(Function):
    @staticmethod
    def forward(ctx: Context, x):
        y = ops.gelu(x)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, y, = ctx.saved_tensors
        _gelu_grad = Primitive('GeLUGrad')
        gx = _gelu_grad(gy.data, x, y)
        return tensor(gx)

def _pack_gelu_erf_grad(x, dy):
    gelu_derivative =  0.5 * (1 + ops.erf(x / ops.sqrt(ops.cast(2, x.dtype)))) + \
        0.5 * x * ops.exp(-x ** 2 / 2) / ops.sqrt(ops.cast(2 * math.pi, x.dtype))
    return dy * gelu_derivative

class GELUErf(Function):
    @staticmethod
    def forward(ctx: Context, x):
        y = 0.5 * x * (1 + ops.erf(x / ops.sqrt(ops.cast(2, x.dtype))))
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, = ctx.inputs
        gx = _pack_gelu_erf_grad(x.data, gy.data)
        return tensor(gx)

class Softmax(Function):
    @staticmethod
    def forward(ctx: Context, input, axis):
        ctx.save_for_backward(axis)
        return ops.softmax(input, axis)

    @staticmethod
    def backward(ctx: Context, gy):
        axis, = ctx.saved_values
        y = ctx.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(dim=axis, keepdims=True)
        gx -= y * sumdx
        return gx
        # gx = fused_softmax_grad(y.data, gy.data, axis)
        # return tensor(gx)

class SoftmaxCrossEntropy(Function):
    @staticmethod
    def forward(ctx:Context, logits, labels):
        loss = ops.SparseSoftmaxCrossEntropyWithLogits()(logits, labels)
        # loss.data_sync(True)
        return loss

    @staticmethod
    def backward(ctx: Context, gy):
        logits, labels = ctx.inputs
        requires_grad = logits.requires_grad | labels.requires_grad
        grad = ops.SparseSoftmaxCrossEntropyWithLogits(True)(logits.data, labels.data)
        grad = ops.mul(grad, gy.data)
        return tensor(grad, requires_grad=requires_grad), None

class SoftmaxCrossEntropyAscend(Function):
    @staticmethod
    def forward(ctx:Context, logits, labels):
        _softmax_crossentropy_ascend = Primitive('SparseSoftmaxCrossEntropyWithLogitsV2')
        loss, grads = _softmax_crossentropy_ascend(logits, labels)
        ctx.save_for_backward(grads)
        return loss

    @staticmethod
    def backward(ctx:Context, gy):
        _, labels = ctx.inputs
        grads, = ctx.saved_tensors
        grad = grads * gy.reshape(-1, 1)
        return grad, zeros_like(labels)

def _pack_linear_grad(x, w, b, gy):
    # print(type(x), type(w), type(b), type(gy))
    ndim = len(b.shape)
    lead = len(gy.shape) - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(b.shape) if sx == 1])
    gb = ops.ReduceSum(True)(gy, lead_axis + axis)
    if lead > 0:
        gb = gb.squeeze(lead_axis)

    x_shape = x.shape
    if len(gy.shape) != 2:
        gy = ops.reshape(gy, (-1, gy.shape[-1]))
    if len(x.shape) != 2:
        x = ops.reshape(x, (-1, x_shape[-1]))
    gx = ops.MatMul()(gy, w)
    gx = gx.reshape(*x_shape[:-1], gx.shape[-1])
    gW = ops.MatMul(True)(x, gy)
    return gx, gW.T, gb

class Linear(Function):
    @staticmethod
    def forward(ctx: Context, x, w, b):
        if len(x.shape) == 1:
            x = ops.unsqueeze(x, 0)
            y = ops.matmul(x, Tensor(w).T)
            y = ops.squeeze(y, 0)
        elif len(w.shape) == 1:
            w = ops.unsqueeze(x, 1)
            y = ops.matmul(x, Tensor(w).T)
            y = ops.squeeze(y, 1)
        else:
            y = ops.matmul(x, Tensor(w).T)
        if b is not None:
            y = ops.add(y, b)
        # y = fused_linear(x, w, b)
        # y = raw_linear(x, w, b)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, W, b = ctx.inputs
        # gb = None if b.data is None else sum_to(gy.data, b.shape)
        # gx = matmul(gy, W)
        # gW = matmul(x, gy, transpose_a=True)
        # return gx, gW.T, tensor(gb)
        gx, gw, gb = _pack_linear_grad(x.data, W.data, b.data, gy.data)
        # gx, gw, _ = raw_linear_grad(x.data, W.data, gy.data)
        return tensor(gx), tensor(gw), tensor(gb)

class Conv2d(Function):
    @staticmethod
    def forward(ctx: Context, x, w, out_channel, kernel_size, pad_mode="valid", pad=0, stride=1, dilation=1, groups=1):
        ctx.save_for_backward(out_channel, kernel_size, pad_mode, pad, stride, dilation, groups)
        conv2d = ops.Conv2D(out_channel, kernel_size, 1, pad_mode, pad, stride, dilation, groups)
        y = conv2d(x, w)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, w = ctx.inputs
        out_channel, kernel_size, pad_mode, pad, stride, dilation, groups = ctx.saved_tensors
        gx = conv2d_gx(gy, w, x.shape, out_channel, kernel_size, pad_mode, pad, pad, stride, dilation, groups)
        gw = conv2d_gw(gy, x, w.shape, out_channel, kernel_size, pad_mode, pad, pad, stride, dilation, groups)
        return gx, gw

def _conv2d(x, w, out_channel, kernel_size, pad_mode="valid", pad=0, stride=1, dilation=1, groups=1):
    return Conv2d.apply(x, w, out_channel=out_channel, kernel_size=kernel_size, pad_mode=pad_mode, pad=pad,
                        stride=stride, dilation=dilation, groups=groups)

class Conv2dGx(Function):
    @staticmethod
    def forward(ctx: Context, gy, w, x_shape, out_channel, kernel_size, pad_mode="valid", pad=0, pad_list=None,
               stride=1, dilation=1, groups=1):

        ctx.save_for_backward(out_channel, kernel_size, pad_mode, pad, pad_list, stride, dilation, groups)
        _conv2d_gx = ops.Conv2DBackpropInput(out_channel, kernel_size, pad_mode, pad, pad_list, 1, stride, dilation, groups)
        y = _conv2d_gx(gy, w, x_shape)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        x, w = ctx.inputs
        out_channel, kernel_size, pad_mode, pad, pad_list, stride, dilation, groups = ctx.saved_tensors
        gx = _conv2d(gy, w, out_channel, kernel_size, pad_mode, pad, stride, dilation, groups)
        gw = conv2d_gw(x, gy, w.shape, out_channel, kernel_size, pad_mode, pad, pad_list, stride, dilation, groups)
        return gx, gw

def conv2d_gx(gy, w, x_shape, out_channel, kernel_size, pad_mode="valid", pad=0, pad_list=None,
              stride=1, dilation=1, groups=1):
    return Conv2dGx.apply(gy, w, x_shape=x_shape, out_channel=out_channel, kernel_size=kernel_size, pad_mode=pad_mode,
                          pad=pad, pad_list=pad_list, stride=stride, dilation=dilation, groups=groups)

class Conv2dGw(Function):
    @staticmethod
    def forward(ctx: Context, gy, x, w_shape, out_channel, kernel_size, pad_mode="valid", pad=0, pad_list=None,
               stride=1, dilation=1, groups=1):

        ctx.save_for_backward(out_channel, kernel_size, pad_mode, pad, pad_list, stride, dilation, groups)
        _conv2d_gw = Conv2DBackpropFilter(out_channel, kernel_size, pad_mode, pad, pad_list, 1, stride, dilation, groups)
        y = _conv2d_gw(gy, x, w_shape)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        dy, x = ctx.inputs
        out_channel, kernel_size, pad_mode, pad, pad_list, stride, dilation, groups = ctx.saved_tensors
        gx = _conv2d(x, gy, out_channel, kernel_size, pad_mode, pad, stride, dilation, groups)
        gw = conv2d_gx(dy, gy, x.shape, out_channel, kernel_size, pad_mode, pad, pad_list, stride, dilation, groups)
        return gx, gw

def conv2d_gw(gy, x, w_shape, out_channel, kernel_size, pad_mode="valid", pad=0, pad_list=None,
              stride=1, dilation=1, groups=1):
    return Conv2dGw.apply(gy, x, w_shape=w_shape, out_channel=out_channel, kernel_size=kernel_size, pad_mode=pad_mode.upper(),
                          pad=pad, pad_list=pad_list, stride=stride, dilation=dilation, groups=groups)


class BiasAdd(Function):
    @staticmethod
    def forward(ctx: Context, x0, x1):
        y = ops.bias_add(x0, x1)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        _bias_add_grad = BiasAddGrad()
        gb = _bias_add_grad(gy.data)
        return gy, tensor(gb)

def _bias_add(x, b):
    return BiasAdd.apply(x, b)

class Dropout(Function):
    @staticmethod
    def forward(ctx: Context, x, dropout):
        y, mask = ops.Dropout(1-dropout)(x)
        ctx.save_for_backward(mask, dropout)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        mask, dropout = ctx.saved_tensors
        _dropout_grad = DropoutGrad(1-dropout)
        gx = _dropout_grad(gy.data, mask)
        return tensor(gx)


# class DropoutGrad(Function):
#     @staticmethod
#     def forward(ctx: Context, x, mask, dropout):
#         gx = raw_dropout_grad(x, mask, dropout)
#         ctx.save_for_backward(mask, dropout)
#         return gx

#     @staticmethod
#     def backward(ctx: Context, gy):
#         mask, dropout = ctx.saved_tensors
#         mask = tensor(mask)
#         gx = _dropout_grad(gy, mask, dropout)
#         return gx, zeros_like(mask)

# def _dropout_grad(x, mask, dropout):
#     return DropoutGrad.apply(x, mask, dropout=dropout)


class MaxPool(Function):
    @staticmethod
    def forward(ctx: Context, x, kernel_size, strides=None, pads=0, dilation=(1, 1), ceil_mode=False, return_indices=False):
        out, indices = ops.MaxPoolWithArgmaxV2(kernel_size, strides, pads, dilation, ceil_mode)(x)
        ctx.save_for_backward(kernel_size, strides, pads, dilation, ceil_mode, indices)
        if return_indices:
            return out, indices
        return out

    @staticmethod
    def backward(ctx: Context, gy, indices=None):
        kernel_size, strides, pads, dilation, ceil_mode, indices = ctx.saved_tensors
        x, = ctx.inputs
        gx = _maxpool_grad(x, gy, tensor(indices), kernel_size, strides, pads, dilation, ceil_mode)
        return gx

def _maxpool(x, kernel_size, strides=None, pads=0, dilation=(1, 1), ceil_mode=False, return_indices=False):
    return MaxPool.apply(x, kernel_size=kernel_size, strides=strides, pads=pads,
                         dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)

class MaxPoolGrad(Function):
    @staticmethod
    def forward(ctx: Context, x, grad, argmax, kernel_size, strides=None, pads=0, dilation=(1, 1), ceil_mode=False):
        _maxpool_grad = MaxPoolGradWithArgmaxV2(kernel_size, strides, pads, dilation, ceil_mode)
        return _maxpool_grad(x, grad, argmax)

def _maxpool_grad(x, grad, argmax, kernel_size, strides=None, pads=0, dilation=(1, 1), ceil_mode=False):
    return MaxPoolGrad.apply(x, grad, argmax, kernel_size=kernel_size, strides=strides, pads=pads,
                             dilation=dilation, ceil_mode=ceil_mode)

class NLLLoss(Function):
    @staticmethod
    def forward(ctx: Context, input, target, weight, ignore_index, reduction):
        out, total_weight = ops.NLLLoss(reduction, ignore_index)(input, target, weight)
        ctx.save_for_backward(ignore_index, reduction, total_weight)        
        return out

    @staticmethod
    def backward(ctx: Context, gy):
        input, target, weight = ctx.inputs
        ignore_index, reduction, total_weight = ctx.saved_tensors
        _nll_loss_grad = NLLLossGrad(reduction, ignore_index)
        gx = _nll_loss_grad(input.data, gy.data, target.data, weight.data, total_weight)
        return tensor(gx), zeros_like(target), zeros_like(weight)

class LayerNorm(Function):
    @staticmethod
    def forward(ctx: Context, input, weight, bias, begin_norm_axis=1, begin_params_axis=1, epsilon=1e-7):
        _layer_norm = ops.LayerNorm(begin_norm_axis, begin_params_axis, epsilon)
        out, mean, var = _layer_norm(input, weight, bias)
        ctx.save_for_backward(mean, var, begin_norm_axis, begin_params_axis)
        return out

    @staticmethod
    def backward(ctx: Context, gy):
        input, weight, _ = ctx.inputs
        mean, var, begin_norm_axis, begin_params_axis = ctx.saved_tensors
        _layer_norm_grad = LayerNormGrad(begin_norm_axis, begin_params_axis)
        gx, gw, gb = _layer_norm_grad(input.data, gy.data, mean, var, weight.data)
        return tensor(gx, requires_grad=gy.requires_grad), \
               tensor(gw, requires_grad=gy.requires_grad), \
               tensor(gb, requires_grad=gy.requires_grad)

# class Unfold(Function):
#     @staticmethod
#     def forward(ctx: Context, input, kernel_size, dilation=1, padding=0, stride=1):
#         return raw_unfold(input, kernel_size, stride, dilation, padding)

#     @staticmethod
#     def backward(ctx: Context, gy):
#         pass