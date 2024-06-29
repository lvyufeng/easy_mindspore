from mindspore._c_expression import GradOperation_
from mindspore.common.api import _pynative_executor
from mindspore.ops import stop_gradient, GradOperation
from typing import Generator
from easy_mindspore import Tensor, MS_23

grad_cell = GradOperation(False, True, False)
def value_and_grad(fn, params, has_aux=False):
    if isinstance(params, Generator):
        params = tuple(params)

    grad_ = grad_cell

    def fn_aux(*args):
        outputs = fn(*args)
        no_grad_outputs = ()
        for out in outputs[1:]:
            no_grad_outputs += (stop_gradient(out),)
        return outputs[0], no_grad_outputs

    if has_aux:
        fn_ = fn_aux
    else:
        fn_ = fn

    def value_and_grad_f(*args):
        _pynative_executor.set_grad_flag(True)
        _pynative_executor.new_graph(fn, *args)
        values = fn_(*args)
        _pynative_executor.end_graph(fn, values, *args)

        if MS_23:
            grads = _pynative_executor.grad(fn_, grad_, params, None, *args)
            # grads = grad_(fn_, params)(*args, *params)
        else:
            _pynative_executor.grad(fn_, grad_, params, None, *args)
            grads = _pynative_executor()
        grads = tuple(Tensor(grad) for grad in grads)
        return values, grads
    return value_and_grad_f

def grad(fn, params=None, has_aux=False):
    value_and_grad_f = value_and_grad(fn, params, has_aux)
    def grad_f(*args):
        _, g = value_and_grad_f(*args)
        return g
    return grad_f
