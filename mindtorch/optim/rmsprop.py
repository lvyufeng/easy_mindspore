from .optimizer import Optimizer
from .._tensor import Tensor

class RMSprop(Optimizer):
    '''
    RMSprop
    '''
    def __init__(self, parameters, lr=0.01, decay_rate = 0.99):
        super().__init__(parameters)
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def step(self):
        if self.h is None:
            self.h = []
            for parameter in self.parameters:
                self.h.append(Tensor(_zeros_like(parameter.data)))

        for idx, parameter in enumerate(self.parameters):
            self.h[idx] *= self.decay_rate
            self.h[idx] += (1 - self.decay_rate) * (parameter.grad * parameter.grad)
            parameter -= self.lr * parameter.grad * (1 / (np.sqrt(self.h[idx].data) + 1e-7))