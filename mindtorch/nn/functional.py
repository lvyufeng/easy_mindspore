from .._tensor import Tensor, Dependency
from mindtorch._functions import linear

def dropout(tensor: Tensor, dropout_ratio:int=0.5, training:bool=True) -> Tensor:
    """
    http://arxiv.org/abs/1207.0580
    """
    requires_grad = tensor.requires_grad
    mask = np.random.rand(*tensor.shape) > dropout_ratio
    if training:
        data = tensor.data * mask
    else:
        data = tensor.data * (1.0 - dropout_ratio)

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * mask
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)
    
def tanh(tensor: Tensor) -> Tensor:
    '''
    tanh = 
    '''
    data = np.tanh(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)

def sigmoid(tensor: Tensor) -> Tensor:
    data = 1 / (1 + np.exp(-tensor.data))
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (data * (1 - data))
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def cross_entropy(input:Tensor, target:Tensor) -> Tensor:
    y = input.data
    t = target.data
    if y.dim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    if y.size == t.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return Tensor(-np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size)

def binary_cross_entropy():
    pass

def mean_squard_error():
    pass
