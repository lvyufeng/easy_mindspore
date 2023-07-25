from mindtorch import zeros, ones

def test_zeros_scalar():
    x = zeros(3)
    print(x.shape)

def test_zeros_two_scalar():
    x = zeros(3, 3)
    print(x.shape)

def test_zeros_two_tuple():
    x = zeros((3, 3))
    print(x.shape)
