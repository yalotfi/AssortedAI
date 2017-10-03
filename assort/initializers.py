import numpy as np


class Initializer(object):
    """Base initializer class is parent of all initializers."""
    def __call__(self):
        raise NotImplementedError


class Zeros(Initializer):
    """Initialize parameters as an array of zeros."""
    def __call__(self, shape):
        return np.zeros(shape)


class Ones(Initializer):
    """Initialize parameters as an array of ones."""
    def __call__(self, shape):
        return np.ones(shape)


class RandomNormal(Initializer):
    """Initialize parameters from a normal random distribution."""
    def __init__(self, mean=0., stdv=1., seed=None):
        self.mean = mean
        self.stdv = stdv
        self.seed = seed

    def __call__(self, shape, scaling_factor=0.1):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.normal(self.mean, self.stdv, shape) * scaling_factor
