import easy_mindspore
from easy_mindspore import ops

def test_bitwise_and():
    out1 = ops.bitwise_and(easy_mindspore.tensor([-1, -2, 3], dtype=easy_mindspore.int8),
                            easy_mindspore.tensor([1, 0, 3], dtype=easy_mindspore.int8))
    out2 = ops.bitwise_and(easy_mindspore.tensor([True, True, False]),
                            easy_mindspore.tensor([False, True, False]))
    print(out1, out2)
