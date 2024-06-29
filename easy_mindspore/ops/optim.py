from mindspore.ops import Primitive
from ..executor import execute

_adadelta = Primitive('ApplyAdadelta')
def raw_adadelta(param, square_avg, acc_delta, lr, rho, eps, grad):
    return execute(_adadelta, param, square_avg, acc_delta, lr, rho, eps, grad)

_adam = Primitive('Adam')
_adam.add_prim_attr('side_effect_mem', True)
_adam.add_prim_attr('use_locking', False)
_adam.add_prim_attr('use_nesterov', False)
def raw_adam(param, exp_avg, exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
    # var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad
    return execute(_adam, param, exp_avg, exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)

_adam_amsgrad = Primitive('ApplyAdamWithAmsgradV2')
_adam_amsgrad.add_prim_attr('use_locking', False)
def raw_adam_amsgrad(param, exp_avg, exp_avg_sq, max_exp_avg_sq, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
    # var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad
    return execute(_adam_amsgrad, param, exp_avg, exp_avg_sq, max_exp_avg_sq,
                      beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)
