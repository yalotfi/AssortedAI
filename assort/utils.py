import numpy as np


def sigmoid(z):
    """
    Implement the sigmoid logistic function
    """
    return 1 / (1 + np.exp(-z))


def log_loss(y_hat, y):
    m = y.shape[0]  # Number of training examples
    loss = -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return loss
