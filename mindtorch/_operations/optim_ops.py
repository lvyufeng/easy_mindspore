from mindspore import ops
from mindspore.ops import Primitive
from mindspore.common.api import _pynative_executor as executor

_adadelta = Primitive('ApplyAdadelta')
def raw_adadelta(param, square_avg, acc_delta, lr, rho, eps, grad):
    return executor.real_run_op(_adadelta, 'ApplyAdadelta', [param, square_avg, acc_delta, lr, rho, eps, grad])