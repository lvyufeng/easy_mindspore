import easy_mindspore
from .optimizer import Optimizer


class Adadelta(Optimizer):
    """Implements Adadelta algorithm.

    It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    __ https://arxiv.org/abs/1212.5701
    """

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super(Adadelta, self).__init__(params, defaults)

    def step(self, grads):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        start = 0
        for group in self.param_groups:
            end = start + len(group['params'])
            for (p, grad) in zip(group['params'], grads[start: end]):
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = easy_mindspore.ops.zeros(grad.shape)
                    state['acc_delta'] = easy_mindspore.ops.zeros(grad.shape)

                square_avg, acc_delta = state['square_avg'], state['acc_delta']
                rho, eps = group['rho'], group['eps']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # square_avg.mul_(rho).addcmul_(grad, grad, value=1-rho)
                # std = square_avg.add(eps).sqrt_()
                # delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
                # p.add_(delta, alpha=-group['lr'])
                # acc_delta.mul_(rho).addcmul_(delta, delta, value=1-rho)
                param_, square_avg_, acc_delta_ = easy_mindspore.ops.optim.raw_adadelta(
                    p, square_avg.data, acc_delta.data, group['lr'], rho, eps, grad
                )
                p.data = param_
                square_avg.data = square_avg_
                acc_delta.data = acc_delta_

        return loss
