import numpy as np


class GradientDescent(object):
    """docstring for GradientDescent."""
    def __init__(self, learning_rate, epochs):
        super(GradientDescent, self).__init__()
        self.alpha = learning_rate
        self.epochs = epochs

    def _update_params(self, parameters, gradient):
        return parameters - self.alpha * gradient
