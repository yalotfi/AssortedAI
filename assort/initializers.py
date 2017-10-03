import numpy as np


class Initializer(object):
    """docstring for Initializer."""
    def __call__(self, arg):
        super(Initializer, self).__init__()
        self.arg = arg


class RandomNorm(Initializer):
    """docstring for RandomNorm."""
    def __call__(self, shape):
        return np.random.rand(shape)


class Zeros(Initializer):
    """docstring for Zeros."""
    def __call__(self, shape):
        self.shape = shape
