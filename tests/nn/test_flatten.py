from easy_mindspore import nn
from easy_mindspore.ops import ones

def test_flatten():
    input = ones(32, 1, 5, 5)
    # With default parameters
    m = nn.Flatten()
    output = m(input)
    print(output.size())
    
    # With non-default parameters
    m = nn.Flatten(0, 2)
    output = m(input)
    print(output.size())


def test_unflatten():
    input = ones(2, 50)
    m = nn.Unflatten(1, (2, 5, 5))
    output = m(input)
    print(output.size())
