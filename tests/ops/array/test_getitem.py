from easy_mindspore import ops, tensor
from mindspore.ops import assign

def test_simple_getitem():
    x = ops.randn(3, 5)
    print(x, x.shape)
    y = x[1]
    print(y, y.shape)
    # y[0] = 5
    assign(y[0], 5)
    print(x)
    print(y)

def test_shrink():
    x = ops.randn(3, 5)
    x = x[None, :, :]
    print(x.shape)
