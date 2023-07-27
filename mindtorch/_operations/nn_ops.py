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
