from dataclasses import dataclass
from typing import Tuple, Any, Optional, Type, Sequence
import weakref
from mindtorch import Tensor
from mindtorch.config import Config
from mindspore.common._stub_tensor import StubTensor

def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)

def wrap_stub_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return tuple([i.stub_sync() if isinstance(i, StubTensor) else i for i in x])
    return (x.stub_sync() if isinstance(x, StubTensor) else x,)


@dataclass(unsafe_hash=True)
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values

# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, *grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, *grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor, **kwargs) -> Tensor:
        return wrap_stub_tuple(cls.forward(ctx, *inps, **kwargs))  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor, **kwargs):
        requires_grad = kwargs.pop('requires_grad', True) # for creation ops

        raw_vals = [x.data for x in vals]

        requires_grad = any([x.requires_grad for x in vals]) \
            and Config.enable_backprop and requires_grad  # whether inputs requires grad

        # Create the context.
        ctx = Context(not requires_grad)
        ctx.inputs = vals

        # Call forward with the variables.
        ys = cls._forward(ctx, *raw_vals, **kwargs)

        outputs = [Tensor(y, requires_grad=requires_grad) for y in ys]

        if requires_grad: # cut useless nodes
            generation = max([x.generation for x in vals])
            ctx.outputs = [weakref.ref(output) for output in outputs]
            back = History(cls, ctx, generation)
            for output in outputs:
                output.set_creator(back)

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

@dataclass(unsafe_hash=True)
class History:
    """
    `History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    generation: int = 0
