import numpy as np
import mindtorch as torch
from mindtorch import nn
# import torch
# from torch import nn

# def test_for_loop():
#     """use conv layer"""    
#     def corr2d(X, K):  
#         """Compute 2D cross-correlation."""
#         h, w = K.shape
#         Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
#         for i in range(Y.shape[0]):
#             for j in range(Y.shape[1]):
#                 Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
#         return Y
    
#     X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
#     K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
#     out = corr2d(X, K)
#     assert (out == torch.tensor([[19., 25.], [37., 43.]])).numpy().all()

def test_while_grad():
    """https://github.com/mindspore-ai/mindspore/blob/master/tests/st/control/test_while_grad.py#L45"""
    def test_fn(x, y):
        while x < y:
            x = x * x + 1
        return x
    
    x = torch.tensor([2.0], requires_grad=True, dtype=torch.float)
    y = torch.tensor([5.0], requires_grad=True, dtype=torch.float)
    
    z = test_fn(x, y)
    z.backward()
    
    assert x.grad.numpy() == [4.]

def test_if_else():
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(10, 1)
            self.relu = nn.ReLU()
        
        def forward(self, x, step):
            x = self.fc(x)
            if step % 2 == 0:
                x = self.relu(x)
            return x
    
    net = Net()
    x = torch.ones(10)
    w_grads = []
    for i in range(10):
        net.zero_grad()
        out = net(x, i)
        out.backward()
        w_grads.append(net.fc.weight.grad)
    
    for idx, grad in enumerate(w_grads):
        if idx % 2 == 0:
            assert (grad == w_grads[0]).numpy().all()
        else:
            assert (grad == w_grads[1]).numpy().all()

def test_rnn_from_scratch():
    """rnn from scratch with for loop"""
    def init_rnn_state(batch_size, num_hiddens):
        return (torch.zeros((batch_size, num_hiddens)), )
    
    def rnn(inputs, state, params):
        W_xh, W_hh, b_h, W_hq, b_q = params
        H, = state
        outputs = []
        for X in inputs:
            H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
            Y = torch.mm(H, W_hq) + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H,)
    
    class RNNModelScratch(nn.Module): 
        """A RNN Model implemented from scratch."""
        def __init__(self, vocab_size, num_hiddens, init_state, forward_fn):
            super().__init__()
            num_inputs = num_outputs = vocab_size
            def normal(shape):
                return torch.randn(shape) * 0.01            
            self.W_xh = nn.Parameter(normal((num_inputs, num_hiddens)))
            self.W_hh = nn.Parameter(normal((num_hiddens, num_hiddens)))
            self.b_h = nn.Parameter(torch.zeros(num_hiddens))
            self.W_hq = nn.Parameter(normal((num_hiddens, num_outputs)))
            self.b_q = nn.Parameter(torch.zeros(num_outputs))

            self.params = [self.W_xh, self.W_hh, self.b_h, self.W_hq, self.b_q]

            self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
            self.init_state, self.forward_fn = init_state, forward_fn

        def forward(self, X, state):
            return self.forward_fn(X, state, self.params)

        def begin_state(self, batch_size):
            return self.init_state(batch_size, self.num_hiddens)
    
    num_hiddens = 512
    net = RNNModelScratch(32, num_hiddens, init_rnn_state, rnn)
    X = torch.randn((5, 2, 32))
    state = net.begin_state(X.shape[1])
    Y, new_state = net(X, state)
    print(Y.shape, len(new_state), new_state[0].shape)
    
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(net.parameters(), 0.01)
    
    for i in range(32):
        optim.zero_grad()
        x = torch.randn((5, 2, 32))
        y = torch.ones((10), dtype=torch.int)
        y_hat, state = net(x, state)
        loss = loss_fn(y_hat, y)
        loss.backward()
        print(loss)
        optim.step()
        