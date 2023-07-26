"""
The idea here is that we'd like to use our library
to minimize a function, say ** 2
"""
import sys
sys.path.append('./')
import torch
import mindtorch as torch

x = torch.tensor([10., -10., 10., -5., 6., 3., 1.], requires_grad=True)

# we want to minimize the sum of squares
for i in range(100):
    # print(x)
    sum_of_squares = (x * x).sum() # is a 0-tensor
    sum_of_squares.backward()

    # what i would like to do
    # ugly b/c we haven't implemented the stuff yet
    delta_x = 0.1 * x.grad
    # print(delta_x.data,x.data)
    x = torch.tensor(x - delta_x, requires_grad=True)

    print(i, sum_of_squares)
