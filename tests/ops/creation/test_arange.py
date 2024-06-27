from easy_mindspore.ops import arange

def test_arange():
    x = arange(2, 5, 1)
    print(x)
    y = arange(2)
    print(y)
