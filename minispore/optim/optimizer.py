"""
Optimizers go there
"""
from minispore.nn import Module


class Optimizer:
    def __init__(self, parameters):
        self.parameters = []
        self.add_param_group(parameters)

    def step(self):
        raise NotImplementedError

    def add_param_group(self, param_group):
        for group in param_group:
            self.parameters.append(group)

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()
