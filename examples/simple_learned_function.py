import numpy as np
import sys
sys.path.append('./')
from minispore import Tensor, Parameter, Module
from minispore.optim import SGD

x_data = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([[-1.], [+3.], [-2.]]))
y_data = x_data @ coef + 5.


class Model(Module):
    def __init__(self) -> None:
        # tensor (3,) requires_grad = True,  random values
        self.w = Parameter(3, 1)
        self.b = Parameter(1)

    def predict(self, inputs: Tensor) -> Tensor:
        return inputs @ self.w + self.b


batch_size = 32
model = Model()
optimizer = SGD(model.parameters(), 0.001)

for epoch in range(100):

    epoch_loss = 0.0
    for start in range(0, 100, batch_size):
        end = start + batch_size
        model.w.zero_grad()
        model.b.zero_grad()
        inputs = x_data[start:end]
        # TODO: implement batching
        # TODO: implement matrix multiplication
        predicted = model.predict(inputs)
        actual = y_data[start:end]

        errors = predicted - actual
        loss = (errors * errors).sum()

        loss.backward()
        epoch_loss += loss

        optimizer.step()

    print(epoch, epoch_loss)
