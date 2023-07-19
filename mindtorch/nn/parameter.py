from mindtorch.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data=None, requires_grad=True) -> None:
        super().__init__(data, requires_grad)
