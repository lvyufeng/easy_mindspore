import math
import easy_mindspore
from easy_mindspore.nn import Linear, Parameter
from easy_mindspore.nn.functional import linear
from easy_mindspore.nn.init import initializer, Normal, HeUniform
from easy_mindspore.autograd import value_and_grad

def test_functional_linear_backward():
    input = easy_mindspore.ops.randn(1, 10)
    weight = Parameter(initializer(HeUniform(math.sqrt(5)), (20, 10)))
    bias = Parameter(initializer(Normal(), (20,)))
    # print(weight.name, bias.name)
    class Net:
        def __init__(self):
            self.weight = weight
            self.bias = bias

        def __call__(self, input):
            return linear(input, weight, bias)

    net = Net()

    # def forward(input, weight, bias):
    #     output = linear(input, weight, bias)
    #     return output

    def forward(input):
        output = linear(input, net.weight, net.bias)
        return output

    print(id(net.weight), id(net.bias))
    # grad_fn = mindspore.value_and_grad(forward, None, (net.weight, net.bias))
    grad_fn = value_and_grad(forward, (net.weight, net.bias))
    # grad_fn = mindspore.value_and_grad(forward, (1, 2))
    print(grad_fn(input))
    # print(grad_fn(input, weight, bias))


def test_linear_backward():
    linear = Linear(10, 20)
    input = easy_mindspore.ops.randn(1, 10)
    def forward(input, *args):
        output = linear(input)
        return output

    grad_fn = value_and_grad(forward, tuple(linear.parameters()))
    # grad_fn = mindspore.value_and_grad(forward, 0)
    print(grad_fn(input))
