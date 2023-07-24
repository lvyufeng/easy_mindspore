import numpy as np
import sys
import time
sys.path.append('./')
from mindtorch import Tensor
from mindtorch.nn import Parameter, Module
from mindtorch.optim import SGD
# from torch import Tensor
# from torch.nn import Parameter, Module
# from torch.optim import SGD

x_data = Tensor(np.random.randn(100, 1024)).cuda()
coef = Tensor(np.random.randn(1024, 2048)).cuda()
# x_data = Tensor(np.random.randn(100, 3))
# coef = Tensor(np.array([-1., +3., -2.]))
# coef = Tensor(np.array([[-1.], [+3.], [-2.]]))
y_data = x_data @ coef + 5.

class Model(Module):
    def __init__(self) -> None:
        super().__init__()
        # tensor (3,) requires_grad = True,  random values
        self.w = Parameter(Tensor(np.random.randn(1024, 2048)))
        self.b = Parameter(Tensor(np.random.randn(2048)))
        # self.w = Parameter(Tensor(np.random.randn(3, 1)))
        # self.b = Parameter(Tensor(np.random.randn(1)))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs @ self.w + self.b

batch_size = 32
model = Model().cuda()
optimizer = SGD(model.parameters(), 0.001)

for epoch in range(100):

    s = time.time()
    epoch_loss = 0.0
    for start in range(0, 100, batch_size):
        end = start + batch_size
        optimizer.zero_grad()
        inputs = x_data[start:end]
        actual = y_data[start:end]
        # TODO: implement batching
        # TODO: implement matrix multiplication
        predicted = model(inputs)

        errors = predicted - actual
        loss = (errors * errors).sum()

        loss.backward()
        epoch_loss += loss

        optimizer.step()
    t = time.time()
    print(t - s)

    print(epoch, epoch_loss)
