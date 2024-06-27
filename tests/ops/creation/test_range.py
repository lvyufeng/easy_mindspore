import easy_mindspore
from easy_mindspore.ops import range

def test_range():
    x = range(1, 4)
    print(x)
    y = range(1, 4, 0.5)
    print(y)

def test_range_int():
    x = range(1, 4, dtype=easy_mindspore.int)
    print(x)
    y = range(1, 4, 1.2, dtype=easy_mindspore.int)
    print(y)
