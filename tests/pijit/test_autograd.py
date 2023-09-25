import copy
import torch
import mindtorch as torch

class Function(torch.nn.Module):
    def __init__(self):
        super(Function, self).__init__()
        self.Linear = torch.nn.Linear(1,1)

    def forward(self, input):
        output = self.Linear(input)
        return output

def test_normal_train():
    x = torch.tensor([2.0])
    y = torch.tensor([4.0])
    func = Function()
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.SGD(func.parameters(), lr=0.01)

    w_grad_list = []
    for _ in range(3):
        optim.zero_grad()
        y_hat = func(copy.deepcopy(x))
        loss = loss_fn(y_hat, y)
        loss.backward()
        # optim.step() each step different if update parameter
        w_grad_list.append(copy.deepcopy(func.Linear.weight.grad))

    assert w_grad_list[1].numpy() == w_grad_list[0].numpy()
    assert w_grad_list[2].numpy() == w_grad_list[0].numpy()

def test_grad_accumulate():
    x = torch.tensor([2.0])
    y = torch.tensor([4.0])
    func = Function()
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.SGD(func.parameters(), lr=0.01)

    w_grad_list = []
    optim.zero_grad()
    for _ in range(3):
        y_hat = func(copy.deepcopy(x))
        loss = loss_fn(y_hat, y)
        loss.backward()
        w_grad_list.append(copy.deepcopy(func.Linear.weight.grad))

    optim.step()

    assert w_grad_list[1].numpy() == (2 * w_grad_list[0].numpy())
    assert w_grad_list[2].numpy() == (3 * w_grad_list[0].numpy())


def test_intermediate_values():
    func = Function()
    x = torch.tensor([1.0])
    y = func(x)
    y_hat =  y ** 2

    y_hat.backward()
    assert y.grad is None
    assert y_hat.grad is None    

def test_retain_graph():
    func = Function()

    x = torch.tensor([1.0])
    x.requires_grad=True
    y = func(x) ** 2
    print(y.shape)

    y.backward(retain_graph=True)
    w_grad_0 = copy.deepcopy(func.Linear.weight.grad)
    y.backward()
    w_grad_1 = func.Linear.weight.grad

    # print(func.Linear.weight.grad)
    print(w_grad_0, w_grad_1)
    assert w_grad_1.numpy() == (2 * w_grad_0).numpy()

def test_create_grad():
    # for high order
    pass

def test_multi_loss():
    x = torch.tensor([2.0])
    y0 = torch.tensor([4.0])
    y1 = torch.tensor([4.0])
    func = Function()
    loss_fn = torch.nn.MSELoss()
    
    w_grad_list = []
    y_hat = func(copy.deepcopy(x))
    loss0 = loss_fn(y_hat, y0)
    loss0.backward(retain_graph=True)
    w_grad_list.append(copy.deepcopy(func.Linear.weight.grad))

    loss1 = loss_fn(y_hat, y1)
    loss1.backward()
    w_grad_list.append(copy.deepcopy(func.Linear.weight.grad))

    assert w_grad_list[1].numpy() == (2 * w_grad_list[0].numpy())


def test_joint_loss():
    x = torch.tensor([2.0])
    y0 = torch.tensor([4.0])
    y1 = torch.tensor([4.0])
    func = Function()
    loss_fn = torch.nn.MSELoss()
    
    y_hat = func(copy.deepcopy(x))
    assert func.Linear.weight.grad is None
    loss0 = loss_fn(y_hat, y0)    
    loss1 = loss_fn(y_hat, y1)
    (loss1 + loss0).backward()

    assert func.Linear.weight.grad is not None


def test_two_net_connect_with_detach():
    x = torch.tensor([1.0])
    y = torch.tensor([2.0])

    func_0 = Function()
    func_1 = Function()
    loss_fn = torch.nn.MSELoss()

    y_0 = func_0(x)
    y_0 = y_0.detach()
    y_1 = func_1(y_0)
    loss = loss_fn(y_1, y)
    loss.backward()
    
    assert func_0.Linear.weight.grad is None
    assert func_0.Linear.bias.grad is None

    assert func_1.Linear.weight.grad is not None
    assert func_1.Linear.bias.grad is not None

def test_two_net_connect_without_detach():
    x = torch.tensor([1.0])
    y = torch.tensor([2.0])

    func_0 = Function()
    func_1 = Function()
    loss_fn = torch.nn.MSELoss()

    y_0 = func_0(x)
    y_1 = func_1(y_0)
    loss = loss_fn(y_1, y)
    loss.backward()
    
    assert func_0.Linear.weight.grad is not None
    assert func_0.Linear.bias.grad is not None

    assert func_1.Linear.weight.grad is not None
    assert func_1.Linear.bias.grad is not None

def test_share_weight():
    x = torch.tensor([1.0])
    y = torch.tensor([2.0])

    func_0 = Function()
    func_1 = Function()
    loss_fn = torch.nn.MSELoss()
    # not share weight
    y_0 = func_0(x)
    y_1 = func_1(y_0)
    loss = loss_fn(y_1, y)
    loss.backward()
    
    print(func_0.Linear.weight.grad.data)
    print(func_1.Linear.weight.grad.data)

    assert func_0.Linear.weight.grad != func_1.Linear.weight.grad
    
    func_0_weight_not_shared = copy.deepcopy(func_0.Linear.weight.grad)
    func_1_weight_not_shared = copy.deepcopy(func_1.Linear.weight.grad)
    print(func_0_weight_not_shared, func_1_weight_not_shared)
    # zero_grad
    func_0.zero_grad()
    func_1.zero_grad()
    # share weight
    func_1.Linear.weight = func_0.Linear.weight
    y_0 = func_0(x)
    y_1 = func_1(y_0)
    loss = loss_fn(y_1, y)
    loss.backward()

    print(func_0.Linear.weight.grad, func_1.Linear.weight.grad)
    assert func_0.Linear.weight == func_1.Linear.weight
    assert func_0.Linear.weight.grad == func_1.Linear.weight.grad
    assert func_0.Linear.weight.grad != func_0_weight_not_shared
    assert func_0.Linear.weight.grad != func_1_weight_not_shared

def test_vanilla_backward():
    x = torch.tensor([1.0], requires_grad=True)
    y = x * 2
    z = y + x
    z.backward()
    
    assert x.grad is not None
    assert x.grad.numpy() == [3]
