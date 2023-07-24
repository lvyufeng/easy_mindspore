from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, lr: float = 0.01) -> None:
        self.lr = lr
        super(SGD, self).__init__(parameters)

    def step(self) -> None:
        for parameter in self.parameters:
            parameter -= parameter.grad * self.lr
