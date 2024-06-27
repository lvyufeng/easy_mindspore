from easy_mindspore import ops, tensor

def test_simple_setitem():
    x = ops.randn(3, 5)
    print(x, x.shape)
    x[1] = 5
    print(x)

def test_simple_setitem_array():
    x = ops.randn(3, 5)
    print(x, x.shape)
    x[1] = ops.randn(5,)
    print(x)

def test_setitem_4d_with_ellipsis():
    x = ops.randn(3, 4, 5, 6)
    print(x)
    x[:,:,:, 2:] = x[:,:,:, 2:] + 1
    print(x)
