from mindspore._c_expression import TensorNode, SequenceNode, NoneTypeNode, AnyTypeNode
from mindspore.common.api import _pynative_executor as executor
from ._tensor import tensor

def _convert_stub(stub):
    "convert stub to StubNode or Value"
    if isinstance(stub, TensorNode):
        return tensor(stub)
    if isinstance(stub, tuple):
        return tuple(_convert_stub(e) for e in stub)
    if isinstance(stub, SequenceNode):
        elements = stub.get_elements()
        return tuple(_convert_stub(e) for e in elements)
    if isinstance(stub, NoneTypeNode):
        val = stub.get_real_value()
        return tensor(val)
    if isinstance(stub, AnyTypeNode):
        val = stub.get_real_node()
        return _convert_stub(val)
    return tensor(stub)

def execute(op, *args):
    out = executor.run_op_async(op, op.name, args)
    return _convert_stub(out)

__all__ = ['execute']
