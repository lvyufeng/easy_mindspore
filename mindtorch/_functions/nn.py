from mindtorch import tensor
from mindtorch.autograd import Function
from mindtorch._operations import raw_relu, raw_relu_grad, raw_softmax_crossentropy, raw_mul, \
    raw_softmax_crossentropy_ascend
from mindtorch._functions import utils
from .creation import zeros_like

# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================
class ReLU(Function):
    def forward(self, x):
        y = raw_relu(x)
        return y

    def backward(self, gy):
        y, = self.outputs
        gx = raw_relu_grad(gy.data, y().data)
        return tensor(gx, requires_grad=y().requires_grad)

class SoftmaxCrossEntropy(Function):
    def forward(self, logits, labels):
        loss = raw_softmax_crossentropy(logits, labels)
        return loss

    def backward(self, gy):
        logits, labels = self.inputs
        requires_grad = logits.requires_grad | labels.requires_grad
        grad = raw_softmax_crossentropy(logits.data, labels.data, True)
        grad = raw_mul(grad, gy.data)
        return tensor(grad, requires_grad=requires_grad), zeros_like(labels)

class SoftmaxCrossEntropyAscend(Function):
    def forward(self, logits, labels):
        loss, grads = raw_softmax_crossentropy_ascend(logits, labels)
        self.grads = grads
        return loss

    def backward(self, gy):
        _, labels = self.inputs
        grad = self.grads * gy.reshape(-1, 1)
        return grad, zeros_like(labels)
