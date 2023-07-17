import sys
sys.path.append('./')
import numpy as np
from minispore.tensor import Tensor

x_data = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([[-1.], [3.], [-2.]]))
y_data = x_data @ coef + 5.

learning_rate = 0.001

w = Tensor(np.random.randn(3, 1), requires_grad=True)
b = Tensor(np.random.randn(), requires_grad=True)

batch_size = 32
for epoch in range(100):
    epoch_loss = 0.0
    for start in range(0, 100, batch_size):
        end = start + batch_size
        w.zero_grad()
        b.zero_grad()
        inputs = x_data[start:end]
        # TODO: implement batching
        # TODO: implement matrix multiplication
        predicted = inputs @ w + b
        actual = y_data[start:end]

        errors = predicted - actual
        loss = (errors * errors).sum()

        loss.backward()
        epoch_loss += loss

        w -= w.grad * learning_rate
        b -= b.grad * learning_rate
    print(epoch, epoch_loss)
