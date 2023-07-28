from mindtorch import tensor
from mindtorch.autograd import Function, Context
from mindtorch._operations import raw_relu, raw_relu_grad, raw_softmax_crossentropy, raw_mul, \
    raw_softmax_crossentropy_ascend
from mindtorch._functions import utils
from .creation import zeros_like

# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================
class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, x):
        y = raw_relu(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: Context, gy):
        y, = ctx.saved_tensors
        gx = raw_relu_grad(gy.data, y)
        return tensor(gx)

class SoftmaxCrossEntropy(Function):
    @staticmethod
    def forward(ctx:Context, logits, labels):
        loss = raw_softmax_crossentropy(logits, labels)
        return loss

    @staticmethod
    def backward(ctx: Context, gy):
        logits, labels = ctx.inputs
        requires_grad = logits.requires_grad | labels.requires_grad
        grad = raw_softmax_crossentropy(logits.data, labels.data, True)
        grad = raw_mul(grad, gy.data)
        return tensor(grad, requires_grad=requires_grad), zeros_like(labels)

class SoftmaxCrossEntropyAscend(Function):
    @staticmethod
    def forward(ctx:Context, logits, labels):
        loss, grads = raw_softmax_crossentropy_ascend(logits, labels)
        ctx.save_for_backward(grads)
        return loss

    @staticmethod
    def backward(ctx:Context, gy):
        _, labels = ctx.inputs
        grads, = ctx.save_for_backward
        grad = grads * gy.reshape(-1, 1)
        return grad, zeros_like(labels)
