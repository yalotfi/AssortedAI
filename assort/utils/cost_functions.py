import numpy as np

def log_loss(y_hat, y):
    """Negative log-liklihood or categorical cross entropy"""
    assert(y_hat.shape == y.shape)
    assert(y_hat.shape[0] == y.shape[0])
    m = y.shape[0]  # Number of training examples
    loss = -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return loss
