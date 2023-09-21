from mindtorch._tensor import Tensor
from mindspore import Parameter as _Parameter

PARAM_COUNT = 0

class Parameter(Tensor):
    def __init__(self, data=None, requires_grad=True, name=None) -> None:
        super().__init__(data, requires_grad=requires_grad)
        global PARAM_COUNT
        self.data = _Parameter(self.data, requires_grad=requires_grad, name=str(PARAM_COUNT))
        PARAM_COUNT += 1
