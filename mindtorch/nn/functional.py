import mindtorch
from mindtorch import BACKEND
from .._tensor import Tensor, Dependency
from mindtorch._functions import ReLU, GELU, SoftmaxCrossEntropy, Linear, SoftmaxCrossEntropyAscend, LogSoftmax, \
    ones, matmul, uniform
from mindtorch._functions.nn import _conv2d, _bias_add, Dropout, _maxpool, NLLLoss, LayerNorm

def make_tuple(inp):
    if isinstance(inp, tuple):
        return (1, 1, inp[0], inp[1])
    elif isinstance(inp, int):
        return (1, 1, inp, inp)

def linear(x, W, b=None):
    # return Linear.apply(x, W, b)
    return _bias_add(matmul(x, W, transpose_b=True), b)

def relu(x):
    return ReLU.apply(x)

def gelu(x, approximate):
    if approximate == 'tanh':
        return GELU.apply(x)
    else:
        return x * 0.5 * (1.0 + mindtorch.erf(x / mindtorch.sqrt(2.0)))

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    weight_shape = weight.shape
    out_channel = weight_shape[0]
    kernel_size = weight_shape[2:4]
    
    pad_mode = 'pad'
    pad = padding
    if isinstance(padding, tuple):
        pad = (padding[0], padding[0], padding[1], padding[1])
    elif isinstance(padding, int):
        pad = (padding,) * 4
    if not isinstance(padding, (int, tuple)):
        pad_mode = padding
        pad = (0,) * 4

    stride = make_tuple(stride)
    dilation = make_tuple(dilation)
    output = _conv2d(input, weight, out_channel, kernel_size, pad_mode, pad, stride, dilation, groups)
    if bias is not None:
        output = _bias_add(output, bias)
    return output


def softmax_cross_entropy(logits, labels):
    if BACKEND == 'Ascend':
        outputs = SoftmaxCrossEntropyAscend.apply(logits, labels)
        return outputs.mean()
    return SoftmaxCrossEntropy.apply(logits, labels)

def dropout(x: Tensor, dropout:int=0.5, training:bool=True) -> Tensor:
    """
    http://arxiv.org/abs/1207.0580
    """
    if training and dropout != 0:
        return Dropout.apply(x, dropout=dropout)
        # mask = uniform(x.shape) > dropout
        # scale = 1 - dropout
        # y = x * mask / scale
        # return y
    else:
        return x

def max_pool2d(input, kernel_size, strides=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):

    if strides is None:
        strides = kernel_size
    kernel_size = make_tuple(kernel_size)
    stride = make_tuple(strides)
    pads = make_tuple(padding)
    dilation = make_tuple(dilation)
    
    return _maxpool(input, kernel_size, stride, pads, dilation, ceil_mode, return_indices)

def log_softmax(input, dim=None, dtype=None):
    if dim is None:
        dim = -1
    out = LogSoftmax.apply(input, axis=dim)
    if dtype is not None:
        out = out.to(dtype)
    return out


def nll_loss(input, target, weight=None, ignore_index=- 100, reduction='mean'):
    if weight is None:
        weight = ones(input.shape[-1])
    return NLLLoss.apply(input, target, weight, ignore_index=ignore_index, reduction=reduction)

def tanh(tensor: Tensor) -> Tensor:
    '''
    tanh = 
    '''
    data = np.tanh(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)

def sigmoid(tensor: Tensor) -> Tensor:
    data = 1 / (1 + np.exp(-tensor.data))
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (data * (1 - data))
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def cross_entropy(input:Tensor, target:Tensor) -> Tensor:
    y = input.data
    t = target.data
    if y.dim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    if y.size == t.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return Tensor(-np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size)

def binary_cross_entropy():
    pass

def mean_squard_error():
    pass

def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    norm_ndim = len(normalized_shape)
    begin_norm_axis = input.ndim - norm_ndim,
    begin_params_axis = input.ndim - norm_ndim
    return LayerNorm.apply(input, weight, bias, begin_norm_axis=begin_norm_axis, 
                           begin_params_axis=begin_params_axis, epsilon=eps)
