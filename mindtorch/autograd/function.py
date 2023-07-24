import weakref
from mindtorch import Tensor
from mindtorch.config import Config

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        requires_grad = any([x.requires_grad for x in inputs]) and Config.enable_backprop
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Tensor(y, requires_grad) for y in ys]
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()