# mindtorch

A torch-like frontend for [MindSpore](https://github.com/mindspore-ai/mindspore), which use `Define-by-run` method to achieve dynamic graph.


## Installation

### Dependency

- mindspore >= 2.1.0 (which optimized operator excution)

### Install from source

To install MindTorch from source, please run:

```bash
pip install git+https://github.com/lvyufeng/mindtorch.git
```


## Simple demo

```python
import numpy as np
from mindtorch import Tensor
from mindtorch.nn import Parameter, Module
from mindtorch.optim import SGD

x_data = Tensor(np.random.randn(100, 1024))
coef = Tensor(np.random.randn(1024, 2048))
y_data = x_data @ coef + 5.

class Model(Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = Parameter(Tensor(np.random.randn(1024, 2048)))
        self.b = Parameter(Tensor(np.random.randn(2048)))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs @ self.w + self.b

batch_size = 32
model = Model()
optimizer = SGD(model.parameters(), 0.001)

for epoch in range(100):

    epoch_loss = 0.0
    for start in range(0, 100, batch_size):
        end = start + batch_size
        optimizer.zero_grad()
        inputs = x_data[start:end]
        actual = y_data[start:end]
        predicted = model(inputs)

        errors = predicted - actual
        loss = (errors * errors).sum()

        loss.backward()
        epoch_loss += loss

        optimizer.step()

    print(epoch, epoch_loss)
```

## Acknowledgement

This repository refers to the implementation of many deep learning frameworks. Thanks to their inspiration for this project, the following are their links:

- [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3)
- [Autograd](https://github.com/joelgrus/autograd)
- [MiniTorch](https://github.com/minitorch/minitorch)
- [Pytorch](https://github.com/pytorch/pytorch)
- [MindSpore](https://github.com/mindspore-ai/mindspore)